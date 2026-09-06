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
  Supports owned regular files, 64-bit Linux, ranges at most 4 GiB. The helper's
  optional `offset=0` preserves prefix compatibility; nonzero offsets map only
  their covering pages, not the preceding file prefix. Unsupported
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

## Bounded trace-matched offset probe (CPU-only, no model load)

`scripts/bench-trace-bandwidth.py` consumes the **complete tagged joined CSV**
from `join-expert-trace.py`, including its zero-drop/error completion footer.
The default source is
`evidence/expert-offset-trace/run-20260906T131516/joined.csv`; default phase is
`decode`, default seed is 17. First inspect the plan without accessing assets:

```sh
python3 scripts/bench-trace-bandwidth.py --plan-only --sample-mib 512 \
  --json evidence/trace-matched-bandwidth/plan.json
```

An actual bounded read (not executed as part of implementation/testing) is:

```sh
python3 scripts/bench-trace-bandwidth.py \
  --model-root /home/seppe/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64 \
  --sample-mib 512 --seed 17 \
  --json evidence/trace-matched-bandwidth/measured.json
```

Output files must be new and outside the model root; existing files/symlinks
are never overwritten. Run real measurements in an independently guarded
cgroup, without concurrent tests/builds; capture the host conditions and memory
peak separately. This script does not manage services, GPU state, or a cgroup.

### Sampling and validation

- `--phase decode|prefill|all` filters before exact `(shard, absolute_offset,
  bytes)` deduplication. Repeated demand is counted, not replayed. Conflicting
  quant labels for the same exact range fail. Only the joiner's supported
  `20/IQ4_NL`, `21/IQ3_S`, `22/IQ2_S` are accepted.
- Canonical shard/offset/size sort, seeded per-quant shuffle and seeded quant
  execution order feed a round-robin selection under a **global per-pass**
  payload cap: default 512 MiB, configurable 1..1024 MiB, plus 8192 slices max.
  One seeded candidate per present quant must fit or the probe refuses the
  budget. Subsequent candidates that do not fit are skipped, never shortened.
  This is quant-stratified slice-count sampling, **not** proportional routing
  demand or a claim that all quant bytes/experts are covered.
- Global/per-quant coverage reports filtered rows, exact unique slices/bytes,
  duplicate rows, selected slices/bytes and fractions, and occurrences in the
  trace of selected slices. Byte denominators sum exact slice sizes, not an
  interval union if an input contains partially overlapping ranges. JSON always
  states `full_trace_replay: false`, even if a small fixture fits completely.
- CSV input is capped at 64 MiB, 100000 rows and 64 KiB per physical line;
  multiline records are unsupported. Complete header, numeric fields, sequence,
  stride/relative-offset consistency, scope tags and footer counts are checked.
  Bounds are validated for **all** CSV rows, including excluded phases, before
  any payload/header reads. At most 64 GGUF fds plus one root-directory fd are
  held; allocations use one reusable buffer at most 1 MiB. Larger slices use
  consecutive bounded reads at their exact offsets, without alignment rounding.
- Shards must be plain `.gguf` basenames under the supplied root. `openat` with
  `O_RDONLY|O_NOFOLLOW|O_NONBLOCK` prevents following shard symlinks or blocking
  on FIFOs; `fstat` requires regular files owned by the filesystem UID. No model
  is loaded. Eight bytes per shard validate GGUF v2/v3 magic/version outside
  timed intervals. Stats are checked after both passes; concurrent edits or
  ownership changes are unsupported. Tensor metadata and payload identity are
  still trusted to the joined provenance, not re-authenticated by this probe.

### Matched observations and interpretation

Each quant runs `cold_requested` then `warm_repeat` over **identical ordered
slices**. Quant pairs run serially, not one global cold pass followed by one
global warm pass. The first pass requests best-effort DONTNEED on the union of
covering page spans, aligned outward to include both boundary pages; holes are
not dropped. Partial pages can contain neighboring tensor bytes. There is no
global cache drop or forced flush; advice errors are reported, not disguised as
cold success. Dirty/pinned pages or other readers can prevent eviction.

`mincore PROT_NONE MAP_SHARED` snapshots cover the exact selected page union
per quant before and after each read loop, including the final partial EOF page.
Mappings never fault payload. `/proc/self/io` snapshots immediately bracket the
read-loop timer, before the post-read residency observation. Setup, header checks,
advice, allocation and residency scans are excluded from timed intervals.

JSON includes the exact read order/offsets, merged page spans and residency
details, bytes, syscalls, seconds, process storage counter brackets/deltas and
`effective_buffered_gib_per_s`. Unknown residency/counters or regressing counters
produce nulls, not zero. Total results sum per-quant byte/time/storage intervals;
rates are total bytes divided by summed times, never an average of rates.
Total residency sums **per-quant page occurrences**, so shared cross-quant
boundary pages can count again; it is not an atomic whole-sample snapshot.

The warm label denotes an immediate matched reread, **not verified RAM speed**;
`cache_state` remains `unverified`. Both `nvme_bytes_per_s` and `ram_bytes_per_s`
are explicitly null. Readahead may exceed requested byte spans and the process
storage delta is not physical device bandwidth. Copy/syscall/Python overheads
remain in effective buffered rates. These results must not populate the
advisor's hardware-calibration fields or imply decode throughput/GPU demand.

Reproducibility fields include exact joined CSV SHA-256, script/helper SHA-256,
sample-plan SHA-256, seed, quant execution order, Python/platform, input counts,
and model path/stat identities. No full model payload hash is computed. Timing
and cache state are observations, not reproducible constants. `--plan-only`
reports `model_files_verified: false`, with no asset open/stat/read or fabricated
measurements. Two measured passes consume up to twice the sample payload cap,
plus the separately reported header-validation bytes; readahead is not capped
by the requested payload budget.

Tests: `python3 tests/test-trace-bandwidth.py`, also registered in Linux 64-bit
CTest. All GGUF/CSV fixtures are synthetic, created temporarily inside this
repository, and no real model asset reads are part of the test suite.
