# expertpin

**Saliency-pinned MoE expert residency for CUDA inference.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`expertpin` is an experimental, CUDA-focused fork of
[`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp). It consumes an
ordered REAP saliency manifest and pre-populates the top-K experts of each layer
before the existing mmap expert streamer begins access-order prefetching. The
remaining experts stay mmap-backed and can continue to stream from disk.

The change is intentionally narrow: it integrates with
`--defer-experts`, `--prefetch-experts`, and `--prefetch-experts-threads`; it
does not rewrite CUDA kernels, prune a GGUF, or remap router IDs.

> [!WARNING]
> This is an early experiment. It has no quality, latency, throughput, or memory
> reduction claims. Validate it on your own model and workload. The residency
> path is currently Linux-specific because it builds on the existing mmap and
> `madvise` prefetch implementation.

## Why

REAP ranks MoE experts by measured router saliency. Those rankings and kept
expert manifests are runtime-independent, but the initial Qwen3.8-Flash-Next
work targeted MLX. `expertpin` provides a small bridge for CUDA users: keep the
original stock expert IDs, load an ordered manifest at runtime, and give its
salient prefix priority in the existing `ik_llama.cpp` expert paging path.

For each manifest layer, `--resident-experts K`:

1. selects exactly the first `K` expert IDs in manifest order;
2. queues their mmap ranges before normal access-order streaming;
3. waits for that initial population to complete; and
4. excludes those ranges from full-tensor lookahead reclamation, leaving the
   non-resident tail available for mmap-backed streaming.

`K=0` is the default and leaves the upstream execution path unchanged. A
positive `K` requires both `--expert-manifest` and `--prefetch-experts`.

## Manifest format

Expert IDs are the **stock model IDs**. Ordering is significant: highest
saliency comes first.

A bare layer map is accepted:

```json
{
  "0": [447, 194, 390, 446],
  "1": [254, 340, 282, 202]
}
```

The same map may be wrapped by any one of these keys:

```json
{
  "layers": {
    "0": [447, 194, 390, 446]
  }
}
```

```json
{
  "pinned_experts": {
    "0": [447, 194, 390, 446]
  }
}
```

```json
{
  "kept_experts": {
    "0": [447, 194, 390, 446]
  }
}
```

A wrapped document may contain sibling metadata. Multiple wrapper keys in one
document are rejected as ambiguous.

Validation is fail-closed:

- the root and selected wrapper must contain a non-empty layer map;
- layer keys must be non-negative integers;
- every layer must have a non-empty expert-ID list;
- expert IDs must be non-negative integers and unique within their layer;
- layer and expert IDs are checked against the loaded model; and
- `K` must fit every provided layer list.

No fixed `48 x 512` shape is assumed. Malformed JSON, unreadable files, invalid
coordinates, or incompatible `K` values abort model loading with an error.

## Build

A CUDA build for NVIDIA Blackwell (`sm_120`) can be configured with:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_CURL=OFF

cmake --build build --target llama-server llama-cli test-expert-manifest -j
```

Use the CUDA architecture appropriate for your GPU when it is not `sm_120`.
See the upstream [`docs/build.md`](docs/build.md) for other platforms and build
options.

## Usage

### `llama-server`

```bash
./build/bin/llama-server \
  --model /path/to/model.gguf \
  --gpu-layers 999 \
  --cpu-moe \
  --defer-experts \
  --prefetch-experts \
  --prefetch-experts-threads 8 \
  --expert-manifest /path/to/qwen38-flash-next-512e_saliency_pinned.json \
  --resident-experts 288
```

### `llama-cli`

```bash
./build/bin/llama-cli \
  --model /path/to/model.gguf \
  --gpu-layers 999 \
  --cpu-moe \
  --defer-experts \
  --prefetch-experts \
  --expert-manifest /path/to/qwen38-flash-next-512e_saliency_pinned.json \
  --resident-experts 288 \
  --prompt "Explain mixture-of-experts routing."
```

`--cpu-moe` is shown to keep MoE tensors mmap-backed on the CPU while other
eligible tensors can use CUDA. Adjust offload and prefetch settings for your
hardware. If no selected expert tensor is mmap-backed, `expertpin` emits a
warning rather than claiming residency was established.

## Tests

The unit test covers bare and wrapped manifests, malformed input, duplicate and
empty IDs, model-bound checks, CLI constraints, and exact prefix selection:

```bash
ctest --test-dir build -R '^test-expert-manifest$' --output-on-failure
```

The repository contains compact fixtures only. To verify a full REAP export
locally, the test binary also supports:

```bash
./build/bin/test-expert-manifest --verify-prefix \
  /path/to/512e_saliency_pinned.json \
  /path/to/reap-288_kept_saliency_ordered.json \
  288
```

## Example launch script

[`scripts/run-qwen38-flash-next.sh`](scripts/run-qwen38-flash-next.sh) wraps a
guarded `llama-server` run for large MoE models:

- Pre-flight RAM/VRAM/busy-GPU guards (`FORCE=1` overrides, `DRY=1` prints the plan).
- Runs the server inside a `systemd-run --user --scope` cgroup with
  `MemoryHigh`/`MemoryMax`, so an over-budget run dies alone instead of
  triggering a system-wide `systemd-oomd` kill.
- Sets `GGML_CUDA_NO_PINNED=1` by default to avoid the pinned-host-allocation
  failure mode, and bounds `--cache-ram`.
- `MANIFEST=… RESIDENT=288` adds `--expert-manifest`/`--resident-experts 288
  --prefetch-experts`; `DRAFT=1` adds an MTP draft model.

See the script header for a worked 262K-context example configuration.

## Credits

This project exists because of work by several upstream communities:

- **Eyal Toledano / Hamster Research** (`sh0wie`) developed and published the
  Qwen3.8-Flash-Next REAP saliency/pruning work and expert disk-streaming ideas.
  See his [Hugging Face profile](https://huggingface.co/sh0wie), the
  [REAP saliency thread](https://x.com/EyalToledano/status/2092771625523413316),
  the
  [REAP-288 model and reproducible manifest](https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-288-MLX-4bit),
  and the
  [expert disk-streaming post](https://x.com/EyalToledano/status/2093429897188299113).
  `expertpin` is an independent CUDA runtime integration and is not presented as
  an official Hamster Research implementation.
- **ikawrakow and the `ik_llama.cpp` contributors** provide the CUDA/CPU runtime,
  MoE mmap streaming path, quantization work, and the upstream history this fork
  retains: <https://github.com/ikawrakow/ik_llama.cpp>.
- **The `llama.cpp` and ggml contributors** built the original inference stack:
  <https://github.com/ggml-org/llama.cpp> and
  <https://github.com/ggml-org/ggml>.

## License

MIT. See [LICENSE](LICENSE). Existing upstream copyright notices and repository
history are retained.
