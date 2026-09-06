# Cache-budget advisor: calibration-input slice (issue #2)

`python3 scripts/advise-expert-cache.py PROFILE.json [--json OUT.json]`

This is an **analytical candidate ranker**, not an automatic calibrator, allocator,
throughput benchmark or proof of optimal placement. It recommends host/GPU cache
bytes **among the supplied candidates only**. Full issue #2 remains open: no
local PCIe or quant-specific CPU/GPU GEMM calibration is supplied in this slice.
Do not convert shadow counters into measured physical disk traffic.

## Input contract (schema_version 1)

Exact root keys:
- `schema_version`: integer 1.
- `workload`: nonempty provenance string identifying hardware, quant, context,
  batch size, token phase, routing/replay source and measurement conditions.
- `ram_bytes_per_s`, `nvme_bytes_per_s`, `pcie_bytes_per_s`: finite positive rates,
  or null when not measured. Bytes are bytes, NOT GiB. PCIe is the applicable
  pageable-host transfer path; pinned allocations remain disallowed in our runs.
- `candidates`: array of objects, each with exactly these keys:
  - `id`: unique nonempty name.
  - `host_cache_bytes`, `gpu_cache_bytes`: integer proposed cache budgets.
  - `host_other_bytes`, `gpu_other_bytes`: integer conservative non-cache reserves
    for backbone, KV, staging, allocator overhead, etc. GPU reserve must also
    include other GPU users because the guard measures absolute device usage.
  - `ram_bytes_per_token`, `nvme_bytes_per_token`, `pcie_bytes_per_token`: finite
    nonnegative mean traffic for this candidate on the **same** workload.
  - `cpu_compute_seconds_per_token`, `gpu_compute_seconds_per_token`: finite
    nonnegative GEMM elapsed time for that candidate, including internal weight
    reads/dequantization. They are not FLOPs divided by advertised peak throughput.

All candidate numeric fields may be null if unknown. Any null blocks the entire
recommendation (even when that candidate might be over budget); no silent skips.
Missing/unknown keys, nonfinite values, negative values, duplicate IDs/JSON keys,
bools masquerading as numbers and malformed shapes fail with a nonzero exit.
Provenance is **caller asserted**, not automatically authenticated. An arbitrary
number cannot be distinguished from a genuine measurement by this script.

Traffic accounting must avoid double counting: RAM is **additional staging**,
NOT the weight traffic already inside GEMM timings or pageable PCIe measurements.
NVMe/PCIe rates must match the access sizes, patterns and software path being
modeled. Separate fixed per-request latency is not modeled; it must already be
amortized into the matched effective rate. Include zero for a genuinely unused
path, null for an unknown quantity; do not fill missing measurements with guesses.
The three calibration rates remain mandatory even if a traffic term is zero.

## Calculation and limits

```
serial_seconds_per_token = ram_bytes_per_token / ram_bytes_per_s
                        + nvme_bytes_per_token / nvme_bytes_per_s
                        + pcie_bytes_per_token / pcie_bytes_per_s
                        + cpu_compute_seconds_per_token
                        + gpu_compute_seconds_per_token
```

Reject candidates whose cache + non-cache reserve exceeds **40 GiB host** or
**28 GiB GPU**. Rank by cost, then host total, GPU total, ID (order-independent).
Status `advisory` / exit 0 means a modeled candidate exists. Status `blocked` /
exit 2 means incomplete calibration or no feasible candidates. Invalid input
also exits 2 but produces an argparse error rather than a result object.

Serial, no-overlap cost excludes attention/PLE, synchronization, routing and other
unmodeled work. Zero cost is valid when all modeled terms are zero; it does not
mean inference is free. It is neither a measured tokens/s result nor a rigorous upper or
lower runtime bound. No env line is emitted: GPU cache allocation does not yet
exist in the runtime, and `EXPERT_CACHE_SIM_MIB` is shadow-only, not a hard cache.
Budgets must still be verified with cgroup MemoryPeak and nvidia-smi in a guarded
run; reserve inputs alone cannot prove them. Generating valid candidates from
phase-tagged traces, an actual bounded allocator and launcher integration remain
separate work.

## Probe correction and reproducible local execution

`expert_bw_calib.py` now preserves exact byte coverage for unaligned chunks (old
start-offset rounding produced overlaps and holes). `--pattern random --seed 17`
shuffles a disjoint prefix partition; `--chunk-bytes 704000` exercises an IQ3_S
slice-sized read. This is **not** actual expert-offset alignment or routed demand.
`--cold` requests DONTNEED only; JSON now states `cache_state: unverified` and
`dontneed_requested`, replacing the misleading `cold` field. Timing retains full
precision. Buffers and plans are outside the measured I/O interval.

Executed on 2026-09-06, shard03 of AD-4.27bpw, prefix 1024 MiB, chunks 704000 bytes:
- sequential buffered: 0.3449448320316151 s, 2.8990142977655844 GiB/s;
- random permutation seed17 + DONTNEED: 0.6237257489701733 s,
  1.6032687469630507 GiB/s.

Raw JSON: `evidence/bandwidth-advisor/{sequential,random-dontneed}.json`.
No build/test ran concurrently with these reads, but miner and other host tasks
remained active. Single samples, sequential-first ordering; NOT a warm/cold A/B,
RAM-bandwidth calibration, physical NVMe proof, or decode-speed prediction.
Accordingly these numbers are NOT injected as physical NVMe rates into the
advisor. `local-incomplete.json` exercises its fail-closed CLI on actual missing
calibration. Unit-test numbers are explicitly synthetic, not Aorus measurements.

Telemetry correction: runtime JSON `defer_wait_ratio` was actually seconds and
incorrectly zeroed without resident bytes. It is now `defer_wait_seconds`, always
`defer_wait_ns / 1e9`. The separate store advisory ratio with a decode-time
 denominator is unchanged. Consumers must migrate to the new runtime JSON key.

Arithmetic correction to earlier issue #2 commentary: 100 * 1.97 MiB /
5.60 GiB/s = 34.35407366071429 ms, NOT 222.8 ms. Moreover aggregate misses are
not proof of physical I/O; no NVMe-bound diagnosis follows from this arithmetic.

Run tests with `python3 tests/test-expert-bandwidth.py`; CTest also registers them.

## Observed cache state and storage accounting (CPU-only follow-up)

Add `--observe` to `expert_bw_calib.py` to collect:
- `observation.cache_before/cache_after`: page counts from Linux `mincore` on a
  `PROT_NONE MAP_SHARED` prefix mapping. No payload page is faulted by this probe.
  Mapping is unmapped before reads start; vectors cover at most 65536 pages each.
  Supports owned regular files, 64-bit Linux, prefixes at most 4 GiB. Unsupported
  observation/syscall failures report null counts and an error, NOT zero residency.
  Ownership is checked against the calling thread's filesystem UID (fourth Uid
  field in `/proc/thread-self/status`, with matching effective UID namespace).
  Missing/malformed credentials fail closed; credentials are never changed.
  This avoids Linux's security masking for unowned files under stable credentials
  and file ownership; changing either concurrently is unsupported.
- `cache_before_state/cache_after_state`: `all_nonresident`, `all_resident`,
  `mixed` or `unavailable`. These describe **non-atomic snapshots only**, not a
  guarantee that pages remain cold/warm during the entire timed read loop.
- `storage_before/storage_after`: `/proc/self/io` `read_bytes`, not `rchar`.
  `storage_read_bytes_delta` is the nonnegative process-accounted storage delta,
  including readahead. Missing/malformed/regressing counters produce null delta
  and rate. Zero is valid when the kernel reports no storage reads.
- `storage_accounted_gib_per_s`: that accounting delta divided by read-loop
  elapsed time; the counter interval slightly brackets the timed interval.
  It is **not** an isolated device bandwidth. Other tasks can affect runtime,
  process readahead may extend past the requested range, and storage may not be NVMe.

Planning, buffer allocation, DONTNEED and observations are outside read-loop
timing. `gib_per_s` still measures requested buffered payload over read-loop time
(syscalls, copies and Python included). `cache_state` stays `unverified` because
continuous residency is unproven. `nvme_bytes_per_s` and `ram_bytes_per_s` remain
explicitly null; do not feed either buffered or accounted rate into the advisor
as hardware calibration. Warm preadv is not RAM staging/GEMM bandwidth.

`--cold --observe` can show whether DONTNEED actually left pages nonresident;
it does not force eviction, globally drop caches, or prove all physical traffic.
`--pattern random --chunk-bytes 704000 --seed 17` remains a shuffled prefix
partition, **not an expert-offset/routing trace**. Phase-tagged model traces,
trace-matched NVMe/RAM/PCIe and quant-specific CPU/GPU GEMM remain open.

Tests: `python3 tests/test-expert-io-observation.py` (also Linux 64-bit CTest).
