# CPU expert-kernel calibration

Build in a CPU-only tree (tests enabled):

```sh
cmake -S . -B build-cpu-shadow -DGGML_CUDA=OFF -DGGML_MUSA=OFF -DGGML_HIPBLAS=OFF -DGGML_VULKAN=OFF -DGGML_METAL=OFF -DGGML_SYCL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu-shadow --target bench-expert-cpu -j 4
build-cpu-shadow/bin/bench-expert-cpu 8 101
ctest --test-dir build-cpu-shadow -R test-expert-cpu-bench --output-on-failure
```

The executable prints one JSON document after all cases pass validation. Arguments
are mandatory bounded decimal integers: CPU threads 1..256, repetitions 1..10000.
Exit 2 is a CLI error; exit 1 is a caught calibration/validation failure (no partial JSON).
Fatal ggml assertions/allocation failures may abort instead of returning exit 1.
It never opens a model, calls GPU initialization or changes the miner/host services.
The target is deliberately absent from GPU-enabled build trees.

## Measured operation

Real synchronous CPU `GGML_OP_MUL_MAT_ID` on one token, selecting ID 1 from a
**two-expert synthetic bank**. The weights are deterministically generated F32
(seed 17), quantized by the production quantizer with uniform importance weights,
and checked with `ggml_validate_row_data`. The three fixed layouts match
`evidence/bounded-cache-design/local-header-inventory.json` and the quant-block
layout, checked against actual `nb[2]` and `ggml_nbytes` at runtime:

| Quant | Input columns | Output rows | Bytes / expert slice |
|---|---:|---:|---:|
| IQ2_S | 2560 | 640 | 524800 |
| IQ3_S | 2560 | 640 | 704000 |
| IQ4_NL | 640 | 2560 | 921600 |

Before timing, both distinct expert IDs are validated against scalar double
accumulation over dequantized weights and original F32 activations. Relative L2
error must be <=3% (allows internal activation quantization); all output values
must be finite. This tests kernel execution/ID selection, **not model quality**.
The final timed result is validated again. There are three untimed warmups.

Each raw sample times one synchronous `ggml_backend_graph_compute`, including
CPU scheduling/dispatch and activation conversion, including any internal per-call
allocation or worker creation. Benchmark setup, initial workspace allocation,
weight quantization, reference calculations, JSON generation and warmup are excluded. Samples and
min/median/max are in **microseconds**, not seconds or tokens/s. The bank is tiny
and repeatedly reused (cache residency is not measured): this is a **hot-cache microbenchmark**, not a representative
512-expert routing trace, cold miss, fused gate/up block, model decode, RAM bandwidth,
NVMe throughput or GPU kernel timing. No thread affinity or host isolation is imposed.
Record compiler/build flags and host contention alongside measurements.

## Advisor boundary

Do not divide these timings into model tokens/s or insert `slice_bytes / time` as
physical RAM bandwidth. Do not assume linear scaling to routed expert count or
use these hot two-expert results as a measured whole-model `cpu_compute_s`.
The phase-tagged real routing/offset trace, matched cold/warm expert I/O, physical
RAM/PCIe calibration, GPU expert timings and candidate traffic/reserves remain
required before the advisor can make a deployment recommendation. Its existing
blocked/null recommendation remains correct when these inputs are absent.
