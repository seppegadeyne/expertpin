# FreeToken feasibility on Aorus (RTX 5090 + 60 GiB host RAM)

Date: 2026-09-04 · Author: Hermes (expertpin cron) · Status: evidence collected

Mission context: >= 20 tok/s decode with local qwen3.8-flash-next AND >= 20 GiB
MemAvailable remaining. Current ik_llama.cpp route: 9.5-10.9 tok/s short /
6.5 tok/s @64k (MTP n_max=8), RAM goal met (41.8 GiB min).

## Why FreeToken looked like the answer

FreeToken (FlashML-org, Apache-2.0, 9.7k stars) officially supports
qwen3.8-flash-next (docs/models.md) with two checkpoints: FP8 (Qwen) and
NVFP4 (RadixArk). X-research 2026-09-04 surfaced a single-RTX-5090 + 63 GB
host RAM report at 68.3 tok/s decode (NVFP4, no speculative decoding), and
RTX 4090 + 64 GB at ~40 tok/s with PLE on disk.

## Hard numbers from the HF API (blobs=true), measured 2026-09-04

### RadixArk/Qwen3.8-Flash-Next-NVFP4 — 206 safetensors, 125.91 GiB total
- routed experts (384 `layer-*-experts-*.safetensors`): **63.32 GiB**
- BF16 backbone (4 `model-bf16-*`): **14.91 GiB**
- PLE n-gram table FP8 (10 `model-plefp8-*`): **47.68 GiB**

### Qwen/Qwen3.8-Flash-Next-FP8 — 131 shards, 172.78 GiB total
Non-starter on this host.

### Intel/Qwen3.8-Flash-Next-W4A16-AutoRound — 17 files, 168.75 GiB total
- `model-00016-of-00017.safetensors` = 102.4 GB but maps to **128 PLE
  ngram shards** (ple_embedding.ngram_embedding.shard_*), i.e. the PLE table
  in BF16, NOT experts.
- 13 shards of ~5 GiB each (~65 GiB) hold the routed experts (W4A16).
- `model_extra_tensors.safetensors` (5.2 GB) = MTP sidecar (targets model
  layers 48+; see hampsonw delta-repo DERIVATION.txt).

Correction to earlier notes: the "~24 GiB INT4 experts" figure circulating
for the Whamp route refers to *draft/MTP* experts (the hampsonw delta repo
quantizes only the 1,536 routed **MTP** experts, 5.21 GB -> 1.49 GB sidecar);
the *target* model's expert store in every public checkpoint is far larger
(63.3 GiB NVFP4 / ~65 GiB W4A16 / ~89.9 GiB in FP8 checkpoints).

## Why stock FreeToken does not fit Aorus

1. `--moe-backend offload` (the auto/default for MoE) materializes all
   routed experts into **pinned host banks**: lazy anon mmap, filled via
   O_DIRECT, then `cudaHostRegister`'d for the process lifetime
   (python/freetoken/moe/host_banks.py: "The mmaps are held for the process
   lifetime"). Pinned anon pages are unevictable — they subtract from
   MemAvailable for as long as the server runs.
2. 63.32 GiB pinned experts > 60 GiB total RAM. The engine would OOM or be
   killed by the cgroup before serving a single token.
3. The PLE table has a real disk backend (`--ple-backend disk`, the default
   in engine/config.py:27) that streams n-gram rows from the checkpoint
   shards via pinned staging + io_uring (ple_disk.py). That part fits.
4. FreeToken has no "experts on NVMe with host-RAM LRU" mode: the offload
   cache LRU lives on the **GPU** (VRAM slots), misses stream over PCIe
   from the pinned host banks. Disk is only supported for the PLE table.

## What WOULD fit (candidate configurations, unverified)

- NVFP4 experts (63.3 GiB) do not fit pinned. A ~24 GiB-class expert store
  (e.g. further-quantized experts) + disk PLE + bf16 backbone on GPU/VRAM
  split would fit the ">= 20 GiB free" budget, but no such public FreeToken
  checkpoint exists; producing one means re-quantizing experts to ~2-bit
  effective or sparsifying.
- Whamp/vLLM fork route: W4A16 target experts (~65 GiB) + PLE-mmap on NVMe
  (the hampsonw repo ships `model.safetensors.index.ple-mmap.json`, so the
  fork can keep the 95.4 GiB BF16 PLE off-RAM) + MTP INT4 draft experts.
  Unknown: whether that fork's expert path is pageable (the repo's
  `evidence/expert-vmm-rankings-hot110.json` hints at VMM-based paging of
  expert weights). If Whamp's fork pages experts via CUDA VMM, both goals
  (20 tok/s, 20 GiB free) could be met. **Next run: read the Whamp fork.**

## Community datapoints (X, 2026-09-02..04)

- 2x RTX 5090 EP2 FP8: 74.9-83 tok/s decode @262k (Enigmatic331 repo, pinned
  stack, custom P2P driver) — different hardware class, not transferable.
- Single RTX 5090 + 63 GB host: 68.3 tok/s (NVFP4, PLE streamed) — this is
  the config class FreeToken was validated on; note 63 GB host RAM reports
  suggest total RAM above ours (60 GiB) OR pageable config, unclear.
- RTX 4090 + 64 GB: ~40 tok/s with PLE on disk.
- 2x RTX 3090 + 64 GB llama.cpp banded offload: 35-36 tok/s.
- Whamp server60 (4x3090, vLLM fork): 89.7 tok/s C1 with MTP K2 INT4 draft
  experts; acceptance 63.5%, mean length 2.27 — INT4-quantized draft experts
  do not hurt acceptance.

## Verdict

Stock FreeToken on Aorus: **not feasible without a smaller expert store**.
The 68.3 tok/s single-5090 reports remain valuable as proof that the
GPU+PCIe+NVMe hierarchy can serve this model fast, but replicating them
here requires either (a) a checkpoint with <= ~35 GiB experts, or (b) an
engine whose expert path is pageable (Whamp VMM hypothesis). Both are
research threads, not config changes.
