#!/usr/bin/env python3
"""Synthetic trace fixtures only, no GPU/model operations."""
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('trace', Path(__file__).resolve().parents[1] / 'scripts/analyze-verifier-trace.py')
assert spec is not None and spec.loader is not None
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


def rows():
    return [dict(kind='target_verifier_decision_v1', raw_valid=True, temperature=0,
        output_position_1based=p, selected_id=p, proposal_id=None, decision='target_only',
        stage='sampled_before_commit_and_stop_handling', task_id=t, slot_id=0,
        raw_argmax_id=p, raw_runner_up_id=999, raw_top_logit=2.0, raw_second_logit=1.0,
        raw_top_margin=1.0, raw_proposal_logit=None, verifier_row=0, verifier_rows=1,
        logits_row=0, assembled_batch_tokens=1)
        for t in (10, 20) for p in range(1, 129)]


def text(records):
    return '\n'.join('EXPERTPIN_VERIFIER ' + json.dumps(r) for r in records)


class TraceTests(unittest.TestCase):
    def test_complete(self):
        data = trace.read_trace(text(rows()))
        self.assertEqual(len(data), 2)
        self.assertIsNone(trace.first_difference(data[0], data[1]))
        other = copy.deepcopy(data[1]); other[33]['selected_id'] = 999
        self.assertEqual(trace.first_difference(data[0], other)['position_1based'], 34)

    def test_rejects_missing_duplicate_and_cap(self):
        base = rows()
        for invalid in (text(base[:-1]), text(base + [base[-1]]), text(base) + '\nEXPERTPIN_VERIFIER_LIMIT 4096'):
            with self.assertRaises(ValueError): trace.read_trace(invalid)

    def test_rejects_bad_decision_invalid_raw_and_non_greedy(self):
        for mutation in ({'decision': 'accepted'}, {'raw_valid': False}, {'temperature': 1},
                         {'decision': 'rejected', 'proposal_id': 1}, {'stage': 'delivered'}):
            base = rows(); base[0].update(mutation)
            with self.assertRaises(ValueError): trace.read_trace(text(base))

    def test_accept_reject_bonus(self):
        base = rows()
        base[0].update(decision='accepted', proposal_id=1, raw_proposal_logit=2.0, verifier_rows=2, assembled_batch_tokens=2)
        base[1].update(decision='rejected', proposal_id=99, raw_proposal_logit=0.0, verifier_rows=2, assembled_batch_tokens=2)
        base[2].update(decision='bonus')
        self.assertEqual(len(trace.read_trace(text(base))[0]), 128)

    def test_strict_raw_and_row_invariants(self):
        for mutation in ({'selected_id': True}, {'raw_top_logit': float('nan')},
                         {'raw_runner_up_id': 1}, {'raw_top_margin': -1},
                         {'raw_top_margin': 2}, {'verifier_row': 1}, {'logits_row': 1},
                         {'assembled_batch_tokens': 0}, {'raw_proposal_logit': 1}):
            base = rows(); base[0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                trace.read_trace(text(base))

    def test_source_artifact_validation(self):
        # Completed summaries alone must never prove matched workloads.
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'target'; mtp = Path(tmp) / 'mtp'
            for directory, depth in ((target, 0), (mtp, 4)):
                directory.mkdir()
                summary = dict(status='completed', startup_n_max=depth, verifier_trace=True,
                    cleanup_errors=[], scope_stopped_verified=True, qli_status='active', requests=[])
                (directory / 'summary.json').write_text(json.dumps(summary))
                (directory / 'server.log').write_text(text(rows()))
            with self.assertRaises(ValueError): trace.analyze(target, mtp)

    def test_invalid_mtp_depth(self):
        for depth in (0, 1, 32, True, 8.0, '8'):
            with self.subTest(depth=depth), self.assertRaises(ValueError):
                trace.analyze(Path('missing-target'), Path('missing-mtp'), mtp_depth=depth)

    def test_full_artifacts_and_negative_mutations(self):
        for mtp_depth, mutation in ((d, m) for d in (4, 8, 16)
                                  for m in ('none', 'payload', 'missing_response', 'no_mtp', 'wrong_depth')):
            with self.subTest(depth=mtp_depth, mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                dirs = [Path(tmp) / 'target', Path(tmp) / 'mtp']
                harness = trace.harness_module()
                for directory, depth in zip(dirs, (0, mtp_depth)):
                    directory.mkdir()
                    requests = []
                    for index in (1, 2):
                        label = f'{index:02d}-nmax{depth}'
                        payload = harness.payload(depth)
                        if mutation == 'payload' and depth: payload['seed'] = 99
                        (directory / (label + '-request.json')).write_text(json.dumps(payload))
                        timings = dict(predicted_n=256, prompt_ms=20, predicted_ms=5000, predicted_per_second=51.2)
                        if depth:
                            timings.update(draft_n=2, draft_n_accepted=1,
                                draft_by_depth=[dict(depth=1, draft_n=2, draft_n_accepted=1)])
                        response = dict(usage={'completion_tokens': 256}, timings=timings,
                            choices=[dict(finish_reason='length', message={'content': 'synthetic fixture'})])
                        if not (mutation == 'missing_response' and depth and index == 1):
                            (directory / (label + '-response.json')).write_text(json.dumps(response))
                        requests.append(dict(harness.validate_completion(response, depth), label=label, n_max=depth))
                    summary = dict(status='completed', startup_n_max=depth, verifier_trace=True,
                        cleanup_errors=[], scope_stopped_verified=True, qli_status='active', requests=requests)
                    if mutation == 'wrong_depth' and depth:
                        summary['startup_n_max'] = 16 if depth != 16 else 8
                    (directory / 'summary.json').write_text(json.dumps(summary))
                    records = rows()
                    if depth and mutation != 'no_mtp':
                        for offset in (0, 128):
                            records[offset].update(decision='accepted', proposal_id=1,
                                raw_proposal_logit=2.0, verifier_rows=2, assembled_batch_tokens=2)
                    (directory / 'server.log').write_text(text(records))
                if mutation == 'none':
                    result = trace.analyze(*dirs, mtp_depth=mtp_depth)
                    self.assertEqual(result['first_differences'], [None, None])
                    self.assertEqual(result['runs'][1]['depth'], mtp_depth)
                    if mtp_depth == 4:
                        self.assertEqual(trace.analyze(*dirs), result)
                else:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        trace.analyze(*dirs, mtp_depth=mtp_depth)


if __name__ == '__main__': unittest.main()
