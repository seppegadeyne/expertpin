# CPU expert-GEMM slice — 2026-09-06

## Result and limits

A working `bench-expert-cpu THREADS REPETITIONS` is built and exercised, not a
placeholder. Source: `tests/bench-expert-cpu.cpp`; usage and claim boundary:
`docs/expert-cpu-calibration.md`. This advances the CPU part of issue #2 only.

Measured sequentially after all builds, tests and the review finished, with the
miner and normal host tasks active, no CPU affinity or isolation. Order: 1,8,8,1
threads; 101 synchronous graph calls per quant per run, three untimed warmups.
Hardware/compiler/build flags, timestamps and hashes are in `execution.json`;
raw timings in `cpu-*.json`. No synthetic result values: only the **input weights**
are synthetic. Production quantization and CPU MUL_MAT_ID kernels executed.

| Case | First 1-thread median us | First 8-thread | Second 8-thread | Last 1-thread |
|---|---:|---:|---:|---:|
| IQ2_S 524800-byte slice | 175.609 | 23.291 | 23.811 | 177.639 |
| IQ3_S 704000-byte slice | 268.753 | 34.352 | 35.772 | 272.583 |
| IQ4_NL 921600-byte slice | 79.924 | 13.530 | 13.311 | 81.194 |

Both expert IDs and final timed output pass a scalar dequantized-weight reference
check. These cases have different layouts; no quant-only speed comparison follows.
Tiny repeatedly reused banks are not the real 512-expert working set. CPU graph
latency includes dispatch, activation conversion and internal per-call costs.
It does not establish CPU cache residency, RAM bandwidth, physical I/O, GPU GEMM,
whole routed block cost, model quality, model throughput or model budget peaks.

The local header inventory was checked programmatically for every tensor of types
20/21/22: all shapes and storage spans equal the corresponding benchmark shapes
and slice_bytes*512. No model payload was loaded. `ldd` plus `readelf -d` on both
executable and libggml.so show CPU runtime dependencies only, no CUDA libraries.

Advisor re-run:

```sh
python3 scripts/advise-expert-cache.py evidence/expert-gemm/local-incomplete.json --json evidence/expert-gemm/local-advice.json
```

Actual exit 2, status `blocked`, recommendation `null`. No fake physical bandwidth
or candidate whole-model CPU compute was inferred from these microseconds. The
analytical boundary from 15335f1e remains unchanged. Phase-tagged chronological
model expert/file-offset trace, matched cold/warm I/O and RAM/PCIe calibration,
GPU quant timings and candidate traffic/reserves remain open.

## Verification and independent review

- Full CPU Release build green (GGML_CUDA/MUSA/HIPBLAS/VULKAN/METAL/SYCL off).
- Three Python test methods pass on 3.11.15, and on 3.14.7 through CTest. Tests
  run actual kernels with 1/2 threads, odd/even sample counts, numerical validation,
  exact layouts and JSON contract, and reject malformed/empty/overflow arguments.
- Targeted CTest 10/10 green, including test-expert-manifest. Broad main 28/30:
  known `test-tokenizer-0-bert-bge` and `test-chat-template` failures persist.
  Logs remain locally in this directory (not all tracked).
- Test-first RED was a missing executable, not a behavior-regression proof.
  Build initially found the local quantize_chunk signature needed its eighth
  argument and the nlohmann include prefix; corrected before final testing.
- Independent gpt-6-astra review `deleg_f4c4e1a5` read the full staged diff:
  GO, no blockers, four low-priority issues. Parent verified core claims against
  `ggml/src/ggml.c:18307-18313` (real ID routing),
  `ggml/src/ggml-backend.cpp:877-898` (workspace and synchronous compute),
  `ggml/src/ggml.c:29235-29259` (per-call non-OpenMP worker creation), and
  `ggml/src/ggml.c:328-340` (fatal allocation abort).
- All four low-priority issues processed: docs now qualify abort/exit and
  allocation timing; multithread tests reject empty case arrays and include
  empty/long integer input; CMake excludes every listed GPU backend rather than
  just CUDA. Parent verified final sources and re-ran full build/tests afterward.
  No second review pass. Review also confirmed local inventory/quant layouts;
  parent independently read the inventory and ran the equality checks above.
- `/usr/bin/time` is absent: first measurement orchestration failed before any
  benchmark sample. Re-run used direct CPU executable; no peak-RSS claim made.

## GPU and follow-up

No GPU work/model load, qli never stopped and no host-prep was used. The fixed
agent developer rule permits GPU work only if the already-recorded Step-1 baseline
has not run; state-file steering cannot remove that active instruction. The
requested physical GPU/model trace phase is therefore **not complete**.

No new model tok/s, RAM run-footprint or VRAM peak. Historical 8K target-only
54–56 tok/s at 33.83 GiB cgroup MemoryPeak and 26.0 GiB VRAM remains evidence only
for that workload, not for >=20 tok/s at depth. Final qli verification and mail
closeout are recorded in `.hermes-cron-state.md`.
