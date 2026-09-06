#!/usr/bin/env python3
"""Rank supplied cache candidates under 40-GiB RAM / 28-GiB VRAM budgets.

Analytical serial cost only, not a throughput prediction or runtime allocator.
See docs/expert-bandwidth-advisor.md for the measurement contract.
"""
import argparse
import json
import math
from pathlib import Path

GIB = 2**30
RATES = ('ram_bytes_per_s', 'nvme_bytes_per_s', 'pcie_bytes_per_s')
MEMORY = ('host_cache_bytes', 'gpu_cache_bytes', 'host_other_bytes', 'gpu_other_bytes')
TRAFFIC = ('ram_bytes_per_token', 'nvme_bytes_per_token', 'pcie_bytes_per_token')
COMPUTE = ('cpu_compute_seconds_per_token', 'gpu_compute_seconds_per_token')


def parse_profile(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError('invalid JSON constant: ' + value)

    try:
        result = json.loads(text, object_pairs_hook=pairs, parse_constant=reject)
    except RecursionError as exc:
        raise ValueError('JSON nesting too deep') from exc
    # Python versions have different decoder recursion limits; enforce our own
    # schema-independent bound without recursively walking attacker input.
    pending = [(result, 0)]
    while pending:
        value, depth = pending.pop()
        if depth > 64:
            raise ValueError('JSON nesting too deep')
        if isinstance(value, dict):
            pending.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            pending.extend((item, depth + 1) for item in value)
    return result


def number(value, label, positive=False, integer=False):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid or value < 0 or (positive and value == 0) or (integer and type(value) is not int):
        raise ValueError('invalid ' + label)
    return value


def shape(obj, keys, label):
    if not isinstance(obj, dict) or set(obj) != set(keys):
        raise ValueError('invalid fields in ' + label)


def advise(profile):
    shape(profile, ('schema_version', 'workload', 'candidates') + RATES, 'profile')
    if type(profile['schema_version']) is not int or profile['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    if not isinstance(profile['workload'], str) or not profile['workload'].strip():
        raise ValueError('workload must identify the common measurement workload')
    candidates = profile['candidates']
    if not isinstance(candidates, list):
        raise ValueError('candidates must be an array')
    missing = []
    for key in RATES:
        if profile[key] is None:
            missing.append(key)
        else:
            number(profile[key], key, positive=True)
    ids = set()
    for c in candidates:
        shape(c, ('id',) + MEMORY + TRAFFIC + COMPUTE, 'candidate')
        if not isinstance(c['id'], str) or not c['id'].strip() or c['id'] in ids:
            raise ValueError('candidate ids must be unique nonempty strings')
        ids.add(c['id'])
        for key in MEMORY + TRAFFIC + COMPUTE:
            if c[key] is None:
                missing.append(c['id'] + '.' + key)
            else:
                number(c[key], key, integer=key in MEMORY)
    result = dict(schema_version=1, workload=profile['workload'], status='blocked',
                  model='serial-no-overlap; RAM traffic is staging only, excluding traffic inside GEMM or PCIe measurements',
                  recommendation=None, missing=sorted(missing), ranked=[], rejected=[])
    if missing:
        return result  # Never silently ignore incomplete, possibly faster candidates.
    ranked = []
    for c in candidates:
        host = c['host_cache_bytes'] + c['host_other_bytes']
        gpu = c['gpu_cache_bytes'] + c['gpu_other_bytes']
        if host > 40 * GIB or gpu > 28 * GIB:
            result['rejected'].append(c['id'])
            continue
        seconds = sum(c[t] / profile[r] for t, r in zip(TRAFFIC, RATES)) + sum(c[k] for k in COMPUTE)
        number(seconds, 'derived serial cost')
        ranked.append(dict(id=c['id'], host_cache_bytes=c['host_cache_bytes'],
                           gpu_cache_bytes=c['gpu_cache_bytes'], host_total_bytes=host,
                           gpu_total_bytes=gpu, serial_seconds_per_token=seconds))
    ranked.sort(key=lambda c: (c['serial_seconds_per_token'], c['host_total_bytes'],
                               c['gpu_total_bytes'], c['id']))
    result['rejected'].sort()
    result['ranked'] = ranked
    if ranked:
        result['status'] = 'advisory'
        result['recommendation'] = ranked[0]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    try:
        result = advise(parse_profile(args.profile.read_text()))
    except (ValueError, OSError, OverflowError) as exc:
        parser.error(str(exc))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.json:
        args.json.write_text(text)
    print(text, end='')
    return 0 if result['status'] == 'advisory' else 2


if __name__ == '__main__':
    raise SystemExit(main())
