#!/usr/bin/env python3
"""expertpin: expert-bandwidth calibration probe (issue #2).

Measures buffered sequential or seeded random-permutation reads over a file
prefix. This is NOT a routing trace or proof of physical NVMe bandwidth.
Optional POSIX_FADV_DONTNEED is best effort; cache state remains unverified.

Outputs JSON with bytes, seconds, gib_per_s, dontneed_requested, cache_state,
pattern, seed and chunk_bytes. No warm/cold or physical-device assertion.

Usage:
  scripts/expert_bw_calib.py --file <shard.gguf> [--length MiB] [--chunk MiB]
                             [--cold] [--json OUT] [--self-test]

--cold requests DONTNEED first (read-only fd, no root). It does not prove cold I/O.
--self-test validates the chunk-plan math on a temp file and exits.
"""
import argparse
import json
import os
import posix
import random
from pathlib import Path
import tempfile
import time

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def chunk_plan(length_bytes: int, chunk_bytes: int):
    """Return list of (offset, size) covering [0, length) in chunks.

    The final chunk may be short; chunk_bytes must be > 0. Do not round
    offsets: buffered preadv supports unaligned expert-sized chunks.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if length_bytes < 0:
        raise ValueError("length_bytes must be non-negative")
    chunks = []
    off = 0
    while off < length_bytes:
        size = min(chunk_bytes, length_bytes - off)
        chunks.append((off, size))
        off += size
    return chunks


def read_plan(length, chunk, pattern, seed):
    chunks = chunk_plan(length, chunk)
    if pattern == 'random':
        random.Random(seed).shuffle(chunks)
    elif pattern != 'sequential':
        raise ValueError('unknown read pattern')
    return chunks


def drop_cache(fd: int, offset: int, length: int):
    """Best-effort page-cache drop for [offset, offset+length)."""
    posix.posix_fadvise(fd, offset, length, posix.POSIX_FADV_DONTNEED)


def measure(fd: int, length: int, chunk: int, cold: bool,
            pattern='sequential', seed=0):
    chunks = read_plan(length, chunk, pattern, seed)
    if cold:
        drop_cache(fd, 0, length)
    buf = bytearray(chunk)
    t0 = time.perf_counter()
    total = 0
    for off, size in chunks:
        view = memoryview(buf)[:size]
        n = os.preadv(fd, [view], off)
        total += n
        if n != size:
            raise IOError(f"short read at {off}: {n} != {size}")
    dt = time.perf_counter() - t0
    return total, dt


def self_test():
    # chunk-plan math on a temp file
    with tempfile.NamedTemporaryFile(dir=Path(__file__).resolve().parents[1]) as f:
        f.write(b"x" * (10 * MIB))
        f.flush()
        plan = chunk_plan(10 * MIB, 4 * MIB)
        assert [s for _, s in plan] == [4 * MIB, 4 * MIB, 2 * MIB], plan
        assert sum(s for _, s in plan) == 10 * MIB
        # unaligned tail: 10 MiB + 123 -> chunks 4M, 4M, (2M + 123)
        plan2 = chunk_plan(10 * MIB + 123, 4 * MIB)
        assert sum(s for _, s in plan2) == 10 * MIB + 123
        assert plan2[-1][1] == 2 * MIB + 123, plan2[-1]
        assert len(plan2) == 3, plan2
        # zero/negative guards
        try:
            chunk_plan(100, 0)
            raise AssertionError("chunk_bytes=0 must raise")
        except ValueError:
            pass
        # warm measure roundtrip on the temp file (cache-hot, fast)
        fd = os.open(f.name, os.O_RDONLY)
        try:
            total, dt = measure(fd, 4 * MIB, MIB, cold=False)
            assert total == 4 * MIB and dt >= 0
        finally:
            os.close(fd)
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="GGUF shard to read")
    ap.add_argument("--length", type=int, default=2048,
                    help="region length in MiB (default 2048)")
    ap.add_argument("--chunk", type=int, default=16,
                    help="chunk size in MiB (default 16)")
    ap.add_argument("--cold", action="store_true",
                    help="request best-effort DONTNEED; cold state NOT verified")
    ap.add_argument("--chunk-bytes", type=int, help="override --chunk, e.g. 704000")
    ap.add_argument("--pattern", choices=('sequential', 'random'), default='sequential')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write result JSON to this path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.file:
        ap.error("--file is required (or use --self-test)")

    chunk = args.chunk_bytes if args.chunk_bytes is not None else args.chunk * MIB
    if args.length <= 0 or chunk <= 0:
        ap.error('length and chunk must be positive')

    size = os.path.getsize(args.file)
    length = min(args.length * MIB, size)
    if not length:
        ap.error('file must not be empty')
    fd = os.open(args.file, os.O_RDONLY)
    try:
        total, dt = measure(fd, length, chunk, args.cold, args.pattern, args.seed)
    finally:
        os.close(fd)

    gib_per_s = (total / GIB) / dt if dt > 0 else 0.0
    result = {
        "file": args.file,
        "bytes": total,
        "mib": round(total / MIB, 1),
        "seconds": dt,
        "gib_per_s": gib_per_s,
        "dontneed_requested": args.cold,
        "cache_state": "unverified",
        "io_api": "buffered preadv",
        "pattern": args.pattern,
        "seed": args.seed,
        "chunk_bytes": chunk,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
