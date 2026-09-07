#!/usr/bin/env python3
"""Offline validation of bounded original verifier decisions, not retokenization."""
import argparse
import importlib.util
import json
import math
from pathlib import Path


def harness_module():
    path = Path(__file__).resolve().parents[1] / 'evidence/clean-mtp-ab/run-guarded.py'
    spec = importlib.util.spec_from_file_location('verifier_harness', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_row(row):
    for key in ('selected_id', 'raw_argmax_id', 'raw_runner_up_id', 'task_id', 'slot_id',
                'verifier_row', 'verifier_rows', 'logits_row', 'assembled_batch_tokens'):
        if type(row.get(key)) is not int or row[key] < 0:
            raise ValueError('invalid integer: ' + key)
    for key in ('raw_top_logit', 'raw_second_logit', 'raw_top_margin'):
        if type(row.get(key)) not in (int, float) or not math.isfinite(row[key]):
            raise ValueError('invalid raw float: ' + key)
    if (row['raw_argmax_id'] == row['raw_runner_up_id'] or row['raw_top_margin'] < 0
            or not math.isclose(row['raw_top_logit'] - row['raw_second_logit'], row['raw_top_margin'], rel_tol=1e-9, abs_tol=1e-9)):
        raise ValueError('invalid top-two summary')
    if (not 0 <= row['verifier_row'] < row['verifier_rows']
            or not 0 <= row['logits_row'] < row['assembled_batch_tokens']):
        raise ValueError('invalid verifier/logit row')
    proposal = row.get('proposal_id')
    if proposal is None:
        if row.get('raw_proposal_logit') is not None:
            raise ValueError('unexpected proposal logit')
    elif (type(proposal) is not int or proposal < 0
            or type(row.get('raw_proposal_logit')) not in (int, float)
            or not math.isfinite(row['raw_proposal_logit'])):
        raise ValueError('invalid proposal logit or ID')
    decision = row.get('decision')
    if decision == 'target_only' and (row['verifier_row'] != 0 or row['verifier_rows'] != 1):
        raise ValueError('invalid single-token mapping')
    if decision == 'bonus' and row['verifier_row'] != row['verifier_rows'] - 1:
        raise ValueError('invalid bonus mapping')
    if decision in ('accepted', 'rejected') and row['verifier_row'] >= row['verifier_rows'] - 1:
        raise ValueError('invalid proposal mapping')


def read_trace(text):
    tasks = {}
    if 'EXPERTPIN_VERIFIER_LIMIT' in text:
        raise ValueError('process trace cap reached')
    for line in text.splitlines():
        if 'EXPERTPIN_VERIFIER ' not in line:
            continue
        row = json.loads(line.split('EXPERTPIN_VERIFIER ', 1)[1])
        validate_row(row)
        if row['kind'] != 'target_verifier_decision_v1' or not row['raw_valid'] or row['temperature'] != 0:
            raise ValueError('not a valid raw greedy trace')
        pos = row['output_position_1based']
        if type(pos) is not int or not 1 <= pos <= 128:
            raise ValueError('position out of trace window')
        selected, proposal, decision = row['selected_id'], row['proposal_id'], row['decision']
        if decision not in ('target_only', 'bonus', 'accepted', 'rejected'):
            raise ValueError('unknown decision')
        if decision in ('target_only', 'bonus'):
            if proposal is not None:
                raise ValueError('nonproposal row with proposal')
        elif type(proposal) is not int or (proposal == selected) != (decision == 'accepted'):
            raise ValueError('inconsistent acceptance')
        if row['stage'] != 'sampled_before_commit_and_stop_handling':
            raise ValueError('unexpected trace stage')
        group = tasks.setdefault(row['task_id'], [])
        if pos != len(group) + 1:
            raise ValueError('duplicate, missing or unordered position')
        group.append(row)
    if len(tasks) != 2 or any(len(rows) != 128 for rows in tasks.values()):
        raise ValueError('expected two complete first-128 request traces')
    return list(tasks.values())


def first_difference(left, right):
    for a, b in zip(left, right):
        if a['output_position_1based'] != b['output_position_1based']:
            raise ValueError('unaligned traces')
        if a['selected_id'] != b['selected_id']:
            return {'position_1based': a['output_position_1based'], 'target': a, 'mtp': b}
    return None


def analyze(target_dir, mtp_dir):
    harness = harness_module()
    output = {'kind': 'original_verifier_prefix_comparison', 'runs': []}
    all_rows = []
    for directory, depth in ((target_dir, 0), (mtp_dir, 4)):
        summary = json.loads((directory / 'summary.json').read_text())
        if (summary['status'] != 'completed' or summary['startup_n_max'] != depth
                or not summary['verifier_trace'] or summary['cleanup_errors']
                or not summary['scope_stopped_verified'] or summary['qli_status'] != 'active'):
            raise ValueError('unverified run/lifecycle')
        if len(summary['requests']) != 2:
            raise ValueError('expected exactly two request artifacts')
        for index, request in enumerate(summary['requests'], 1):
            label = f'{index:02d}-nmax{depth}'
            if request['label'] != label or request['n_max'] != depth:
                raise ValueError('unexpected request order or depth')
            payload = json.loads((directory / (label + '-request.json')).read_text())
            if payload != harness.payload(depth):
                raise ValueError('unmatched source payload')
            response = json.loads((directory / (label + '-response.json')).read_text())
            validated = harness.validate_completion(response, depth)
            if any(request[key] != validated[key] for key in validated):
                raise ValueError('summary differs from source response')
        rows = read_trace((directory / 'server.log').read_text())
        for group in rows:
            if depth == 0 and any(r['decision'] != 'target_only' for r in group):
                raise ValueError('draft decisions in target run')
            if depth and not any(r['decision'] in ('accepted', 'rejected') for r in group):
                raise ValueError('no proposal verification in MTP prefix')
        all_rows.append(rows)
        output['runs'].append({'directory': str(directory), 'depth': depth,
            'tokens_per_request_traced': 128,
            'repeat_ids_identical': [r['selected_id'] for r in rows[0]] == [r['selected_id'] for r in rows[1]],
            'raw_argmax_mismatch_count': sum(r['selected_id'] != r['raw_argmax_id'] for group in rows for r in group),
            'requests': summary['requests']})
    output['first_differences'] = [first_difference(a, b) for a, b in zip(*all_rows)]
    output['limitations'] = ['Decisions precede commit and stop handling; not delivery proof.',
        'Raw logits precede sampling transforms; margins are not transformed acceptance margins.',
        'Assembled batch size is not a kernel microbatch size.',
        'Equal output prefix does not imply equal hidden/recurrent/KV state.',
        'No identical-prefix replay; numerical versus state cause remains open.',
        'Tracing changes CPU work; throughput is diagnostic, not a clean optimization A/B.']
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', type=Path)
    parser.add_argument('mtp', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.target, args.mtp), indent=2, allow_nan=False) + '\n')
