#!/usr/bin/env python3
"""CPU-only counter-order evidence analysis; never starts a model or changes defaults."""
import argparse
import difflib
from datetime import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path


def text_fields(message):
    """Preserve content and reasoning independently, including null versus empty."""
    if not isinstance(message, dict):
        raise ValueError('missing message')
    result = {k: message.get(k) for k in ('content', 'reasoning_content')}
    if any(v is not None and not isinstance(v, str) for v in result.values()):
        raise ValueError('non-text output')
    if not any(isinstance(v, str) and v.strip() for v in result.values()):
        raise ValueError('empty output')
    return result


def canonical(fields):
    return json.dumps(fields, sort_keys=True, ensure_ascii=False, indent=2) + '\n'


def compare(left, right):
    a, b = canonical(left), canonical(right)
    prefix = 0
    for x, y in zip(a, b):
        if x != y:
            break
        prefix += 1
    return {'equal': a == b, 'left_sha256': hashlib.sha256(a.encode()).hexdigest(),
            'right_sha256': hashlib.sha256(b.encode()).hexdigest(),
            'canonical_common_prefix_chars': prefix,
            'diff': ''.join(difflib.unified_diff(a.splitlines(True), b.splitlines(True),
                                               fromfile='left', tofile='right'))}


def load_run(directory):
    s = json.loads((directory / 'summary.json').read_text())
    if (s['status'] != 'completed' or s['cleanup_errors'] or not s['scope_stopped_verified']
            or s['qli_status'] != 'active' or len(s['requests']) != 2):
        raise ValueError('incomplete/unclean run: ' + str(directory))
    nmax = s['startup_n_max']
    if [r['label'] for r in s['requests']] != [f'01-nmax{nmax}', f'02-nmax{nmax}']:
        raise ValueError('duplicate/unexpected request labels')
    environment = json.loads((directory / 'environment.json').read_text())
    if environment.pop('DRAFT_NMAX') != str(nmax) or environment.pop('EXPERTPIN_SCOPE_UNIT') != s['scope']:
        raise ValueError('launch depth/scope mismatch')
    samples = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    if not samples:
        raise ValueError('no budget samples')
    for p in samples:
        events = dict(line.split() for line in p['memory.events'].splitlines())
        if (int(p['memory.max']) != 36 * 1024**3 or not 0 <= int(p['memory.peak']) <= 40 * 1024**3
                or not 0 <= p['vram_used_mib'] < 28 * 1024
                or any(int(events[k]) for k in ('max', 'oom', 'oom_kill'))):
            raise ValueError('budget evidence failed')
    requests = []
    for r in s['requests']:
        label = r['label']
        payload = json.loads((directory / (label + '-request.json')).read_text())
        if payload.pop('speculative.n_max') != nmax:
            raise ValueError('depth mismatch')
        response = json.loads((directory / (label + '-response.json')).read_text())
        t = response['timings']
        if (response['usage']['completion_tokens'] != 256 or t['predicted_n'] != 256
                or response['choices'][0]['finish_reason'] != 'length'
                or not math.isfinite(t['predicted_ms']) or t['predicted_ms'] <= 0
                or not math.isclose(t['predicted_per_second'], 256000 / t['predicted_ms'], rel_tol=1e-9)
                or t != r['timings']):
            raise ValueError('invalid completion/timing')
        rows = t['draft_by_depth']
        if (sum(x['draft_n'] for x in rows) != t['draft_n']
                or sum(x['draft_n_accepted'] for x in rows) != t['draft_n_accepted']):
            raise ValueError('acceptance totals mismatch')
        metadata = {'model': response['model'], 'n_ctx': t['n_ctx'], 'prompt_n': t['prompt_n'],
                    'prompt_tokens': response['usage']['prompt_tokens'],
                    'cached_tokens': response['usage']['prompt_tokens_details']['cached_tokens']}
        requests.append({'label': label, 'payload': payload, 'metadata': metadata, 'text': text_fields(response['choices'][0]['message']),
                         'timings': t})
    return {'directory': str(directory.resolve()), 'n_max': nmax, 'started': s['started'],
            'finished': s['finished'], 'scope': s['scope'], 'environment': environment,
            'requests': requests, 'ram_peak_gib': max(int(p['memory.peak']) for p in samples) / 1024**3,
            'vram_sample_peak_gib': max(p['vram_used_mib'] for p in samples) / 1024,
            'swap_sample_peak_gib': max(int(p['memory.swap.current']) for p in samples) / 1024**3,
            'samples': len(samples), 'guard': s['guard'],
            'qli_stop_requested': s['qli_stop_requested'], 'qli_start_requested': s['qli_start_requested']}


def analyze(forward, reverse):
    runs = [load_run(p) for p in forward + reverse]
    if [r['n_max'] for r in runs] != [4, 8, 16, 16, 8, 4]:
        raise ValueError('expected chronological forward 4/8/16 and reverse 16/8/4')
    for key in ('directory', 'scope'):
        if len({r[key] for r in runs}) != len(runs):
            raise ValueError('duplicate run ' + key)
    intervals = [(datetime.fromisoformat(r['started']), datetime.fromisoformat(r['finished'])) for r in runs]
    if (any(a.tzinfo is None or b.tzinfo is None or a >= b for a, b in intervals)
            or any(a[1] >= b[0] for a, b in zip(intervals, intervals[1:]))):
        raise ValueError('run intervals overlap or are not chronological')
    if any(r['environment'] != runs[0]['environment'] for r in runs):
        raise ValueError('recorded launch settings differ beyond depth/scope')
    payload = runs[0]['requests'][0]['payload']
    if any(q['payload'] != payload for r in runs for q in r['requests']):
        raise ValueError('workloads differ beyond depth')
    if any(q['metadata'] != runs[0]['requests'][0]['metadata'] for r in runs for q in r['requests']):
        raise ValueError('response model/context/prompt metadata differ')
    outputs = [(f'{order}-nmax{r["n_max"]}-rep{i}', q['text'])
               for order, group in [('forward', runs[:3]), ('reverse', runs[3:])]
               for r in group for i, q in enumerate(r['requests'], 1)]
    pairs = [{'left': a, 'right': b, **compare(x, y)}
             for (a, x), (b, y) in itertools.combinations(outputs, 2)]
    ranks = {}
    for order, group in [('forward', runs[:3]), ('reverse', runs[3:])]:
        rates = {r['n_max']: 512000 / sum(q['timings']['predicted_ms'] for q in r['requests']) for r in group}
        tie_groups = [sorted(n for n in rates if rates[n] == rate) for rate in sorted(set(rates.values()), reverse=True)]
        ranks[order] = {'pooled_tok_s': rates, 'tie_groups': tie_groups,
                        'descending_n_max': [n for group in tie_groups for n in group]}
    return {'runs': runs, 'comparisons': pairs, 'rankings': ranks,
            'same_observed_ranking': ranks['forward']['tie_groups'] == ranks['reverse']['tie_groups'],
            'causal_ranking_proven': False, 'default_changed': False,
            'limitations': ['Single prompt, two repetitions per depth/order, unflushed host cache and swap.',
                            'Recorded launch environment is compared, not a full inherited environment snapshot; binary/model hashes require separate provenance.',
                            'Different outputs imply different realized workloads; no quality or causal explanation inferred.',
                            'VRAM is device-wide sampled usage, not a continuous peak. Swap reported separately.']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forward', nargs=3, type=Path, required=True)
    p.add_argument('--reverse', nargs=3, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.forward, args.reverse)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs', 'comparisons')}, indent=2))


if __name__ == '__main__':
    main()
