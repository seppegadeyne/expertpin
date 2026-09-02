#!/usr/bin/env python3
"""expertpin: expert-bandwidth calibration probe (issue #2).

Measures sequential read bandwidth over a region of the model file, the
way the MoE prefetch engine consumes expert weights: large sequential
chunks, optionally forced cold (page cache dropped via posix_fadvise
DONTNEED) to measure the true NVMe path.

Outputs JSON: {"bytes", "seconds", "gib_per_s", "cold", "chunk_mib", ...}

Usage:
  scripts/expert_bw_calib.py --file <shard.gguf> [--length MiB] [--chunk MiB]
                             [--cold] [--json OUT] [--self-test]

--cold drops the region from the page cache first (needs the file to be
writable-readable by the same user; uses fadvise, no root).
--self-test validates the chunk-plan math on a temp file and exits.
"""
import argparse
import json
import os
import posix
import tempfile
import time

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024


def chunk_plan(length_bytes: int, chunk_bytes: int):
    """Return list of (offset, size) covering [0, length) in chunks.

    The final chunk may be short; chunk_bytes must be > 0. Aligned to
    4 KiB (page granularity of the page cache) except for a short tail.
    """
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if length_bytes < 0:
        raise ValueError("length_bytes must be non-negative")
    page = 4096
    chunks = []
    off = 0
    while off < length_bytes:
        size = min(chunk_bytes, length_bytes - off)
        chunks.append((off, size))
        off += size
    # align all but the last chunk start to page boundaries when possible
    aligned = []
    for i, (o, s) in enumerate(chunks):
        if i < len(chunks) - 1:
            o = (o // page) * page
        aligned.append((o, s))
    return aligned


def drop_cache(fd: int, offset: int, length: int):
    """Best-effort page-cache drop for [offset, offset+length)."""
    posix.posix_fadvise(fd, offset, length, posix.POSIX_FADV_DONTNEED)


def measure(fd: int, length: int, chunk: int, cold: bool):
    if cold:
        drop_cache(fd, 0, length)
    chunks = chunk_plan(length, chunk)
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
    with tempfile.NamedTemporaryFile() as f:
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
                    help="drop page cache for the region first")
    ap.add_argument("--json", help="write result JSON to this path")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.file:
        ap.error("--file is required (or use --self-test)")

    size = os.path.getsize(args.file)
    length = min(args.length * MIB, size)
    fd = os.open(args.file, os.O_RDONLY)
    try:
        total, dt = measure(fd, length, args.chunk * MIB, args.cold)
    finally:
        os.close(fd)

    gib_per_s = (total / GIB) / dt if dt > 0 else 0.0
    result = {
        "file": args.file,
        "bytes": total,
        "mib": round(total / MIB, 1),
        "seconds": round(dt, 4),
        "gib_per_s": round(gib_per_s, 3),
        "cold": args.cold,
        "chunk_mib": args.chunk,
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
