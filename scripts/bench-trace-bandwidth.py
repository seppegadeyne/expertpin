#!/usr/bin/env python3
"""Bounded, quant-stratified buffered reads of joined expert offsets; NOT trace replay.

Cold-requested then matched warm-repeat per quant. DONTNEED is best effort;
mincore snapshots and process storage accounting do not measure hardware rates.
--plan-only validates CSV and samples without opening/stat-ing any model assets.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import stat
import sys
import time
from typing import Any

import expert_bw_calib as bw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOINED = ROOT / 'evidence/expert-offset-trace/run-20260906T131516/joined.csv'
BASE = 'seq,weight_entry,epoch,ids_rows,tensor,ggml_type,expert,stride,relative_offset,bytes,shadow_hit,shadow_bypass'.split(',')
TAGS = 'context,request_id,seq_id,phase,pos_min,pos_max'.split(',')
COLUMNS = BASE + TAGS + ['shard', 'absolute_offset']
QUANTS = {20: 'IQ4_NL', 21: 'IQ3_S', 22: 'IQ2_S'}
MAX_TRACE_BYTES, MAX_ROWS, MAX_LINE = 64 * bw.MIB, 100000, 65536
MAX_SAMPLE_BYTES, MAX_SLICES, MAX_SHARDS, BUFFER_BYTES = bw.GIB, 8192, 64, bw.MIB
LIMITATIONS = (
    'Bounded quant-stratified unique-offset sample, not full-trace replay, temporal routing, '
    'GPU demand or trace-frequency-weighted coverage. Quant labels/layout come from the joined CSV; '
    'GGUF magic/version and bounds are checked, not tensor metadata or payload identity. '
    'DONTNEED is best effort, including covering boundary pages; dirty/pinned pages may remain. '
    'mincore PROT_NONE snapshots do not fault payload and are non-atomic, not continuous cache guarantees. '
    'read_bytes is process-accounted storage including readahead, not device-isolated traffic. '
    'Effective buffered rates include preadv/copies/Python, not physical hardware or RAM/GEMM rates. '
    'Quant pairs run serially; summed timings exclude setup/observations and are not wall time. '
    'Total residency sums per-quant page occurrences (cross-quant boundary pages may repeat). '
    'No concurrent model edits/ownership changes supported; stat checks cannot prove content identity.')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(text, name, minimum=0):
    require(re.fullmatch(r'(?:0|[1-9][0-9]*)', text, re.ASCII) is not None and len(text) <= 19,
            f'invalid unsigned integer {name}: {text!r}')
    value = int(text)
    require(minimum <= value <= (1 << 63) - 1, f'{name} out of range')
    return value


def signature(info):
    return {key: getattr(info, 'st_' + key) for key in ('dev', 'ino', 'size', 'mtime_ns', 'ctime_ns', 'uid')}


def read_trace(path, phase='decode') -> dict[str, Any]:
    require(phase in ('decode', 'prefill', 'all'), 'unsupported phase filter')
    digest, used, rows = hashlib.sha256(), 0, 0
    unique, all_keys, bounds = {}, {}, {}
    phases, contexts, quant_rows = Counter(), Counter(), Counter()
    with Path(path).open('rb') as source:
        before = signature(os.fstat(source.fileno()))
        require(stat.S_ISREG(os.fstat(source.fileno()).st_mode) and before['size'] <= MAX_TRACE_BYTES,
                'joined CSV must be regular and at most 64 MiB')

        def line():
            nonlocal used
            raw = source.readline(MAX_LINE + 1)
            used += len(raw)
            require(len(raw) <= MAX_LINE and used <= MAX_TRACE_BYTES, 'CSV input limit exceeded')
            digest.update(raw)
            return raw.decode('utf-8')

        require(next(csv.reader([line()], strict=True)) == COLUMNS, 'expected complete tagged joined CSV header')
        while True:
            text = line()
            require(bool(text), 'missing completion footer')
            if text.startswith('# end'):
                match = re.fullmatch(r'# end written=(0|[1-9][0-9]*) dropped=0 error=0\r?\n', text)
                require(match is not None and number(match[1], 'written') == rows,
                        'invalid/incomplete footer or row count mismatch')
                require(source.read(1) == b'', 'data after completion footer')
                break
            fields = next(csv.reader([text], strict=True))
            require(len(fields) == len(COLUMNS), 'CSV field count mismatch')
            row = dict(zip(COLUMNS, fields))
            nums = {key: number(row[key], key, 1 if key in ('ids_rows', 'stride', 'bytes') else 0)
                    for key in BASE if key != 'tensor'}
            require(nums['seq'] == rows and rows < MAX_ROWS, 'noncontiguous seq or row cap exceeded')
            require(nums['ggml_type'] in QUANTS, 'unsupported quant; expected joined IQ4_NL/IQ3_S/IQ2_S')
            require(nums['bytes'] == nums['stride'] and nums['relative_offset'] == nums['expert'] * nums['stride'],
                    'inconsistent expert stride/relative offset/bytes')
            require(nums['shadow_hit'] in (0, 1) and nums['shadow_bypass'] in (0, 1)
                    and nums['shadow_hit'] + nums['shadow_bypass'] <= 1, 'invalid shadow flags')
            require(row['tensor'].startswith('blk.') and '_exps.' in row['tensor'], 'not an expert tensor')
            require(re.fullmatch(r'[a-z_]+', row['phase']) is not None and bool(row['context']), 'invalid scope tags')
            number(row['request_id'], 'request_id')
            positions = [number(row[key], key) if row[key] != '-1' else -1
                         for key in ('seq_id', 'pos_min', 'pos_max')]
            require(positions[1] <= positions[2], 'reversed positions')
            shard, offset, size, quant = row['shard'], number(row['absolute_offset'], 'absolute_offset'), nums['bytes'], nums['ggml_type']
            require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.gguf', shard) is not None,
                    'shard must be a plain GGUF basename within model root (no subdirectories/symlinks)')
            require(offset + size <= (1 << 63) - 1, 'slice end exceeds signed file-offset range')
            bounds[shard] = max(bounds.get(shard, 0), offset + size)
            require(len(bounds) <= MAX_SHARDS, 'too many GGUF shards')
            key = (shard, offset, size)
            require(key not in all_keys or all_keys[key] == quant, 'conflicting quant for exact slice')
            all_keys[key] = quant
            phases[row['phase']] += 1
            rows += 1
            if phase != 'all' and row['phase'] != phase:
                continue
            contexts[row['context']] += 1
            quant_rows[quant] += 1
            if key not in unique:
                unique[key] = dict(shard=shard, absolute_offset=offset, bytes=size, ggml_type=quant,
                                   first_csv_line=rows + 1, trace_occurrences=0)
            unique[key]['trace_occurrences'] += 1
        require(signature(os.fstat(source.fileno())) == before, 'joined CSV changed while parsing')
    require(bool(unique), f'no slices for phase {phase}')
    return dict(slices=list(unique.values()), bounds=bounds, quant_rows=quant_rows,
                provenance=dict(joined_csv=str(Path(path).resolve()), joined_sha256=digest.hexdigest(),
                                joined_stat=before, csv_rows=rows, phase_rows=dict(phases),
                                filtered_rows=sum(quant_rows.values()), filtered_context_rows=dict(contexts)))


def sample_plan(slices, budget, seed):
    require(type(budget) is int and 0 < budget <= MAX_SAMPLE_BYTES, 'sample budget must be in (0, 1 GiB]')
    rng, groups = random.Random(seed), defaultdict(list)
    for item in sorted(slices, key=lambda x: (x['shard'], x['absolute_offset'], x['bytes'])):
        groups[item['ggml_type']].append(item)
    for quant in sorted(groups):
        rng.shuffle(groups[quant])
    order = sorted(groups)
    rng.shuffle(order)
    # Reserve the first seeded candidate per quant. Never silently omit a quant.
    require(sum(groups[q][0]['bytes'] for q in order) <= budget, 'budget cannot cover one seeded slice per quant')
    selected, used, count = {q: [] for q in order}, 0, 0
    for index in range(max(map(len, groups.values()))):
        for quant in order:
            if index >= len(groups[quant]) or count >= MAX_SLICES:
                continue
            item = groups[quant][index]
            if used + item['bytes'] <= budget:
                selected[quant].append(item)
                used += item['bytes']
                count += 1
    return selected


def page_ranges(slices):
    """Merge covering page intervals per shard: no holes mapped/dropped, no duplicate pages."""
    ranges = []
    for item in slices:
        start, length = bw.page_span(item['absolute_offset'], item['bytes'])
        ranges.append((item['shard'], start, start + length))
    merged = []
    for shard, start, end in sorted(ranges):
        if merged and merged[-1][0] == shard and start <= merged[-1][2]:
            merged[-1] = (shard, merged[-1][1], max(end, merged[-1][2]))
        else:
            merged.append((shard, start, end))
    return [dict(shard=s, offset=a, length=b - a) for s, a, b in merged]


def nullable_sum(values):
    values = list(values)
    return sum(values) if all(value is not None for value in values) else None


def residency_summary(samples):
    result: dict[str, Any] = {key: nullable_sum(s[key] for s in samples) for key in ('pages', 'resident_pages')}
    result['state'] = bw.cache_snapshot_state(result)
    return result


def observe_ranges(fds, ranges):
    observations = []
    for span in ranges:
        fd = fds[span['shard']]
        # The covering page may extend beyond EOF; map only through the last valid byte.
        size = min(span['length'], os.fstat(fd).st_size - span['offset'])
        observations.append(dict(shard=span['shard'], **bw.cache_snapshot(fd, size, offset=span['offset'])))
    return dict(ranges=observations, **residency_summary(observations))


def measured_pass(fds, slices, ranges, buffer, cold):
    advice: dict[str, Any] = dict(requested=cold, covering_ranges=len(ranges) if cold else 0, errors=[])
    if cold:
        for span in ranges:
            try:
                bw.drop_cache(fds[span['shard']], span['offset'], span['length'])
            except (OSError, AttributeError) as error:
                advice['errors'].append(dict(**span, error=str(error)))
    cache_before = observe_ranges(fds, ranges)
    storage_before = bw.storage_snapshot()
    started = time.perf_counter()
    total, calls = 0, 0
    for item in slices:
        offset, remaining = item['absolute_offset'], item['bytes']
        while remaining:
            size = min(remaining, len(buffer))
            count = os.preadv(fds[item['shard']], [memoryview(buffer)[:size]], offset)
            if count != size:
                raise OSError(f'short read in {item["shard"]} at {offset}: {count} != {size}')
            total += count
            calls += 1
            offset += count
            remaining -= count
    seconds = time.perf_counter() - started
    storage_after = bw.storage_snapshot()
    cache_after = observe_ranges(fds, ranges)
    a, b = storage_before['read_bytes'], storage_after['read_bytes']
    delta = b - a if a is not None and b is not None and b >= a else None
    return dict(bytes=total, seconds=seconds, read_syscalls=calls, dontneed=advice,
                effective_buffered_gib_per_s=total / bw.GIB / seconds if seconds > 0 else None,
                cache_before=cache_before, cache_after=cache_after, cache_state='unverified',
                storage_before=storage_before, storage_after=storage_after, storage_read_bytes_delta=delta,
                nvme_bytes_per_s=None, ram_bytes_per_s=None)


def aggregate(passes):
    total, seconds = sum(p['bytes'] for p in passes), sum(p['seconds'] for p in passes)
    return dict(bytes=total, seconds=seconds, read_syscalls=sum(p['read_syscalls'] for p in passes),
                effective_buffered_gib_per_s=total / bw.GIB / seconds if seconds > 0 else None,
                storage_read_bytes_delta=nullable_sum(p['storage_read_bytes_delta'] for p in passes),
                cache_before=residency_summary([p['cache_before'] for p in passes]),
                cache_after=residency_summary([p['cache_after'] for p in passes]),
                cache_state='unverified', nvme_bytes_per_s=None, ram_bytes_per_s=None)


def coverage(candidates, selected, rows):
    available_bytes, selected_bytes = sum(x['bytes'] for x in candidates), sum(x['bytes'] for x in selected)
    return dict(filtered_rows=rows, unique_slices=len(candidates), duplicate_rows=rows - len(candidates),
                unique_slice_bytes=available_bytes, sampled_slices=len(selected), sampled_bytes=selected_bytes,
                sampled_slice_fraction=len(selected) / len(candidates), sampled_byte_fraction=selected_bytes / available_bytes,
                sampled_trace_occurrences=sum(x['trace_occurrences'] for x in selected))


def probe(joined, model_root, budget=512 * bw.MIB, seed=17, phase='decode', plan_only=False):
    trace = read_trace(joined, phase)
    selected = sample_plan(trace['slices'], budget, seed)
    groups: list[dict[str, Any]] = [dict(ggml_type=q, quant=QUANTS[q], offsets=items, page_ranges=page_ranges(items),
                   coverage=coverage([x for x in trace['slices'] if x['ggml_type'] == q], items, trace['quant_rows'][q]))
              for q, items in selected.items()]
    sample = [x for items in selected.values() for x in items]
    plan = [dict(ggml_type=g['ggml_type'], offsets=g['offsets']) for g in groups]
    result: dict[str, Any] = dict(schema_version=1, mode='plan_only' if plan_only else 'measured', phase=phase, seed=seed,
                  algorithm='canonical-sort-seeded-quant-round-robin-v1', quant_execution_order=list(selected),
                  provenance=dict(**trace['provenance'], python=sys.version, platform=sys.platform,
                                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  helper_sha256=hashlib.sha256(Path(bw.__file__).read_bytes()).hexdigest(),
                                  plan_sha256=hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()),
                  limits=dict(sample_bytes=budget, max_sample_slices=MAX_SLICES, max_shard_fds=MAX_SHARDS,
                              read_buffer_bytes=min(BUFFER_BYTES, max(x['bytes'] for x in sample)),
                              max_csv_bytes=MAX_TRACE_BYTES, max_csv_rows=MAX_ROWS),
                  coverage=coverage(trace['slices'], sample, trace['provenance']['filtered_rows']),
                  full_trace_replay=False, io_api='read-only buffered preadv', quant_results=groups,
                  total=None, model_files_verified=False, limitations=LIMITATIONS)
    if plan_only:
        return result
    require(model_root is not None, '--model-root is required for actual reads')
    root = Path(model_root).resolve(strict=True)
    uid, fds, stats = bw.filesystem_uid(), {}, {}
    with ExitStack() as stack:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        stack.callback(os.close, root_fd)
        for shard, end in sorted(trace['bounds'].items()):
            fd = os.open(shard, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=root_fd)
            stack.callback(os.close, fd)
            info = os.fstat(fd)
            require(stat.S_ISREG(info.st_mode) and info.st_uid == uid, f'not an owned regular GGUF: {shard}')
            require(end <= info.st_size, f'slice out of file bounds: {shard}')
            fds[shard], stats[shard] = fd, signature(info)
        # Metadata only; never load model/tensor data. Check all ranges before any reads.
        for shard, fd in fds.items():
            header = os.pread(fd, 8, 0)
            require(header[:4] == b'GGUF' and int.from_bytes(header[4:], 'little') in (2, 3) and len(header) == 8,
                    f'not a GGUF v2/v3 file: {shard}')
        buffer = bytearray(result['limits']['read_buffer_bytes'])
        for group in groups:
            for label, cold in (('cold_requested', True), ('warm_repeat', False)):
                group[label] = measured_pass(fds, group['offsets'], group['page_ranges'], buffer, cold)
        for shard, fd in fds.items():
            require(signature(os.fstat(fd)) == stats[shard]
                    and signature(os.stat(shard, dir_fd=root_fd, follow_symlinks=False)) == stats[shard],
                    f'GGUF changed during probe: {shard}')
    result.update(model_root=str(root), model_files_verified=True,
                  model_files=stats, header_validation_bytes=8 * len(fds),
                  total={label: aggregate([g[label] for g in groups]) for label in ('cold_requested', 'warm_repeat')})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--joined', type=Path, default=DEFAULT_JOINED)
    parser.add_argument('--model-root', type=Path, help='directory containing joined shard basenames')
    parser.add_argument('--phase', choices=('decode', 'prefill', 'all'), default='decode')
    parser.add_argument('--sample-mib', type=int, default=512, help='per-pass payload cap, 1..1024 MiB; two passes')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--plan-only', action='store_true', help='no asset access; file bounds/magic/ownership unverified')
    parser.add_argument('--json', type=Path, help='new output file; never overwrite an existing file/symlink')
    args = parser.parse_args(argv)
    try:
        require(1 <= args.sample_mib <= 1024, '--sample-mib must be 1..1024')
        if args.json:
            require(not args.json.exists() and not args.json.is_symlink(), '--json must not already exist')
            if args.model_root:
                require(not args.json.resolve().is_relative_to(args.model_root.resolve()), 'output must be outside model root')
        result = probe(args.joined, args.model_root, args.sample_mib * bw.MIB, args.seed, args.phase, args.plan_only)
        text = json.dumps(result, indent=2, allow_nan=False) + '\n'
        if args.json:
            with args.json.open('x') as output:
                output.write(text)
        print(text, end='')
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
