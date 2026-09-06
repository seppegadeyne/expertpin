#!/usr/bin/env python3
"""expertpin: expert-bandwidth calibration probe (issue #2).

Measures buffered sequential or seeded random-permutation reads over a file
prefix. This is NOT a routing trace or proof of physical NVMe bandwidth.
Optional POSIX_FADV_DONTNEED is best effort; cache state remains unverified.

Outputs JSON with bytes, seconds, gib_per_s, dontneed_requested, cache_state,
pattern, seed and chunk_bytes. No warm/cold or physical-device assertion.
--observe adds non-faulting Linux mincore snapshots and /proc/self/io read_bytes
around the reads. Unknown/unsupported observations are null, never fake zeros.

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
from typing import Any

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


def parse_read_bytes(text):
    values = [line.partition(':')[2].strip() for line in text.splitlines()
              if line.partition(':')[0] == 'read_bytes']
    if len(values) != 1 or not values[0].isascii() or not values[0].isdecimal():
        raise ValueError('missing, duplicate or invalid read_bytes counter')
    return int(values[0])


def storage_snapshot() -> dict[str, Any]:
    """Linux process-accounted storage reads, NOT read syscall payload bytes."""
    try:
        return dict(read_bytes=parse_read_bytes(Path('/proc/self/io').read_text()), error=None)
    except (OSError, ValueError) as error:
        return dict(read_bytes=None, error=str(error))


def filesystem_uid():
    """Read this thread's fsuid; never call setfsuid or modify credentials."""
    rows = [line.split()[1:] for line in Path('/proc/thread-self/status').read_text().splitlines()
            if line.partition(':')[0] == 'Uid']
    if len(rows) != 1 or len(rows[0]) != 4 or not all(
            value.isascii() and value.isdecimal() for value in rows[0]):
        raise ValueError('missing, duplicate or invalid thread Uid fields')
    # A nonstandard proc namespace view cannot safely authenticate ownership.
    if int(rows[0][1]) != os.geteuid():
        raise ValueError('proc effective UID differs from the caller')
    return int(rows[0][3])


def page_span(offset, length):
    """Page-aligned covering span, including partial first/last payload pages."""
    if offset < 0 or length <= 0:
        raise ValueError('page range requires nonnegative offset and positive length')
    page = os.sysconf('SC_PAGESIZE')
    start = offset // page * page
    return start, ((offset + length + page - 1) // page) * page - start


def cache_snapshot(fd, length, offset=0):
    """Observe exact range's covering pages without faulting payload into the process.

    The optional byte offset defaults to zero for prefix-probe compatibility.

    PROT_NONE mapping, bounded mincore vectors. Snapshot only: not an atomic
    whole-range observation, residency can change during/after the syscall(s).
    Linux may mask mincore results for files the caller does not own/cannot write;
    refuse those rather than turn a security mask into an all-resident claim.
    """
    result: dict[str, Any] = dict(method='mincore PROT_NONE MAP_SHARED', pages=None,
                  resident_pages=None, error=None, offset=offset, length=length,
                  page_offset=None, page_length=None)
    try:
        import ctypes
        import stat
        import sys
        if not sys.platform.startswith('linux') or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise ValueError('residency observation requires 64-bit Linux')
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != filesystem_uid():
            raise ValueError('mincore requires an owned regular file to avoid masked residency')
        if offset < 0 or length <= 0 or length > 4 * GIB or offset + length > info.st_size:
            raise ValueError('snapshot range must be within the file with length (0, 4 GiB]')
        page_offset, page_length = page_span(offset, length)
        mapped_length = offset + length - page_offset
        result.update(page_offset=page_offset, page_length=page_length)
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_long]
        libc.mmap.restype = ctypes.c_void_p
        libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        libc.mincore.restype = ctypes.c_int
        libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        libc.munmap.restype = ctypes.c_int
        page = os.sysconf('SC_PAGESIZE')
        pages = page_length // page
        addr = libc.mmap(None, mapped_length, 0, 1, fd, page_offset)  # PROT_NONE, MAP_SHARED
        if addr == ctypes.c_void_p(-1).value:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))
        resident = 0
        try:
            for start in range(0, pages, 65536):
                count = min(65536, pages - start)
                vec = (ctypes.c_ubyte * count)()
                if libc.mincore(addr + start * page, min(count * page, mapped_length - start * page), vec):
                    errno = ctypes.get_errno()
                    raise OSError(errno, os.strerror(errno))
                resident += sum(value & 1 for value in vec)
        finally:
            if libc.munmap(addr, mapped_length):
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno))
        result.update(pages=pages, resident_pages=resident)
    except (OSError, ValueError, AttributeError, ImportError) as error:
        result['error'] = str(error)
    return result


def cache_snapshot_state(sample):
    if sample['pages'] is None or sample['resident_pages'] is None:
        return 'unavailable'
    if sample['resident_pages'] == 0:
        return 'all_nonresident'
    if sample['resident_pages'] == sample['pages']:
        return 'all_resident'
    return 'mixed'


def measure(fd: int, length: int, chunk: int, cold: bool,
            pattern='sequential', seed=0, observation=None):
    chunks = read_plan(length, chunk, pattern, seed)
    buf = bytearray(min(chunk, length))
    before: dict[str, Any] = {}
    if cold:
        drop_cache(fd, 0, length)
    if observation is not None:
        observation['cache_before'] = cache_snapshot(fd, length)
        before = storage_snapshot()
    t0 = time.perf_counter()
    total = 0
    for off, size in chunks:
        view = memoryview(buf)[:size]
        n = os.preadv(fd, [view], off)
        total += n
        if n != size:
            raise IOError(f"short read at {off}: {n} != {size}")
    dt = time.perf_counter() - t0
    if observation is not None:
        after = storage_snapshot()
        observation['cache_after'] = cache_snapshot(fd, length)
        a, b = before['read_bytes'], after['read_bytes']
        delta = b - a if a is not None and b is not None and b >= a else None
        observation.update(
            storage_before=before, storage_after=after,
            storage_read_bytes_delta=delta,
            storage_accounted_gib_per_s=delta / GIB / dt if delta is not None and dt > 0 else None,
            cache_before_state=cache_snapshot_state(observation['cache_before']),
            cache_after_state=cache_snapshot_state(observation['cache_after']),
            nvme_bytes_per_s=None, ram_bytes_per_s=None,
            limitations='Non-atomic residency snapshots; process read_bytes includes readahead, '
                        'not device-isolated NVMe traffic. Buffered preadv includes copy and Python '
                        'loop costs; no continuous cache-state guarantee or physical RAM rate.')
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
    ap.add_argument("--observe", action="store_true",
                    help="observe Linux mincore snapshots and process storage read_bytes; not hardware rates")
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
    observation = {} if args.observe else None
    try:
        total, dt = measure(fd, length, chunk, args.cold, args.pattern, args.seed, observation)
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
    if observation is not None:
        result["observation"] = observation
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
