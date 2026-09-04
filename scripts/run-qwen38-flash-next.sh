#!/usr/bin/env bash
# expertpin example launcher — safe, RAM/VRAM-guarded serving of a large MoE GGUF.
#
# Demonstrates the expertpin residency flags together with hard memory guards,
# so a misconfigured run can never take down the host:
#   1. Pre-flight guards: refuses to start when MemAvailable/VRAM are insufficient
#      or the GPU is busy with other work (FORCE=1 overrides).
#   2. systemd-run --user --scope with MemoryHigh/MemoryMax: exceeding the RAM
#      budget kills only this server, never the desktop session.
#   3. GGML_CUDA_NO_PINNED=1 by default: avoids pinned host allocations that
#      previously caused systemd-oomd kills on this class of hardware.
#   4. --cache-ram bounds the server-side cache.
#
# Reference config (Qwen3.8-Flash-Next AD-4.27bpw-Q4_K_M-M64, RTX 5090 + 64 GB
# host RAM): 40.4 tok/s decode / 1067.7 tok/s prefill at 131K ctx on a single
# RTX 3090 with q8_0 KV (community report; ik_llama.cpp #2373 lands after that).
#
# Usage:
#   scripts/run-qwen38-flash-next.sh                     (defaults: 32K ctx)
#   CTX=262144 NCMOE=44 RAM_BUDGET_GIB=38 scripts/run-qwen38-flash-next.sh
#   DRAFT=1 DRAFT_NMAX=3 ...                              (add MTP draft, n_max tokens)
#   MANIFEST=~/*saliency_pinned.json RESIDENT=288 ...     (expertpin residency)
#   EXPERT_CACHE_SIM_MIB=32768 EXPERT_STATS_FILE=/path/cache-shadow.json DRY=1 ...
#                                                         (shadow bounded host-LRU; no eviction)
#   DRY=1 ...                                             (print plan, start nothing)
#   FORCE=1 ...                                           (skip guards)
#
# Example 262K recipe (trained context of Qwen3.8-Flash-Next; 24 of 48 layers
# are linear-recurrent with constant state, so KV scales on 24 layers only):
#   CTX=262144 KVT=q8_0 RAM_BUDGET_GIB=38 NCMOE=44 scripts/run-qwen38-flash-next.sh
#   VRAM ~30: ~18 weights + 9.6 KV(q8_0) + ~3 compute | RAM: experts within 38
#   If VRAM is tight: KVT=q4_0 saves ~4.2 GiB (test q8_0 first: 2 KV-heads are
#   more sensitive to KV quant noise than wide models).
set -euo pipefail

# --- defaults (all overridable via environment) ---
MODEL_DIR="${MODEL_DIR:-$HOME/Models/qwen3.8-flash-next/AD-4.27bpw-Q4_K_M-M64}"
MODEL="$MODEL_DIR/Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64-00001-of-00033.gguf"
DRAFT_MODEL="${DRAFT_MODEL:-$HOME/Models/qwen3.8-flash-next/mtp-drafter/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf}"
DRAFT_NMAX="${DRAFT_NMAX:-4}"
MANIFEST="${MANIFEST:-}"
RESIDENT="${RESIDENT:-0}"
EXPERT_CACHE_SIM_MIB="${EXPERT_CACHE_SIM_MIB:-0}"
EXPERT_STATS_FILE="${EXPERT_STATS_FILE:-}"

CTX="${CTX:-32768}"
NCMOE="${NCMOE:-36}"
NGL="${NGL:-99}"
THREADS="${THREADS:-16}"
PORT="${PORT:-8101}"
KVT="${KVT:-q8_0}"
RAM_BUDGET_GIB="${RAM_BUDGET_GIB:-40}"
GPU_NEED_GIB="${GPU_NEED_GIB:-24}"
CACHE_RAM_MIB="${CACHE_RAM_MIB:-4096}"
DRY="${DRY:-0}"
FORCE="${FORCE:-0}"

BIN_DIR="${BIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/build-sm120/bin}"
[ -x "$BIN_DIR/llama-server" ] || { echo "No llama-server build in $BIN_DIR (see README for build instructions)"; exit 1; }

[ -f "$MODEL" ] || { echo "Model not found: $MODEL (set MODEL_DIR)"; exit 1; }

# --- pre-flight guards ---
mem_avail_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
mem_avail_gib=$(( mem_avail_kib / 1024 / 1024 ))
gpu_busy=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "${gpu_util:-0}" -ge 50 ] && gpu_busy=1
  gpu_free_gib=$(( gpu_free_mib / 1024 ))
else
  gpu_free_gib=0
fi

guard_fail=0
[ "$mem_avail_gib" -lt $(( RAM_BUDGET_GIB + 4 )) ] && {
  echo "GUARD: MemAvailable ${mem_avail_gib} GiB < budget ${RAM_BUDGET_GIB} + 4 GiB headroom"; guard_fail=1; }
[ "$gpu_busy" = "1" ] && {
  echo "GUARD: GPU busy (util ${gpu_util}%) — stop other GPU workloads first"; guard_fail=1; }
[ "${gpu_free_gib:-0}" -lt "$GPU_NEED_GIB" ] && {
  echo "GUARD: free VRAM ${gpu_free_gib} GiB < required ${GPU_NEED_GIB} GiB"; guard_fail=1; }

if [ "$guard_fail" = "1" ] && [ "$FORCE" != "1" ] && [ "$DRY" != "1" ]; then
  echo "Refusing to start (FORCE=1 to override)."
  exit 2
fi

# --- optional MTP draft model ---
DRAFT_ARGS=()
if [ "${DRAFT:-0}" = "1" ] && [ -f "$DRAFT_MODEL" ]; then
  DRAFT_ARGS=(-md "$DRAFT_MODEL" -ngld 99 --spec-type "mtp:n_max=${DRAFT_NMAX}")
fi

# --- optional expertpin residency and bounded-cache shadow ---
EXPERT_ARGS=()
if [ -n "$MANIFEST" ]; then
  [ -f "$MANIFEST" ] || { echo "Manifest not found: $MANIFEST"; exit 1; }
  EXPERT_ARGS+=(--expert-manifest "$MANIFEST")
  [ "$RESIDENT" != "0" ] && EXPERT_ARGS+=(--resident-experts "$RESIDENT" --prefetch-experts)
fi
[ "$EXPERT_CACHE_SIM_MIB" != "0" ] && EXPERT_ARGS+=(--expert-cache-sim-mib "$EXPERT_CACHE_SIM_MIB")
[ -n "$EXPERT_STATS_FILE" ] && EXPERT_ARGS+=(--expert-stats-file "$EXPERT_STATS_FILE")

# --- disable pinned host allocations by default (OOM hardening) ---
if [ "${PINNED:-0}" != "1" ]; then
  export GGML_CUDA_NO_PINNED=1
fi

cd "$(dirname "$0")/.."
RAM_BUDGET_MIB=$(( RAM_BUDGET_GIB * 1024 ))
MEM_HIGH=$(( RAM_BUDGET_MIB * 90 / 100 ))

if [ "$DRY" = "1" ]; then
  echo "DRY-RUN plan:"
  echo "  binary      : $BIN_DIR/llama-server (expertpin)"
  echo "  model       : $MODEL"
  echo "  ctx=$CTX ncmoe=$NCMOE ngl=$NGL kv=$KVT threads=$THREADS port=$PORT"
  echo "  draft       : ${DRAFT:-0}"
  echo "  manifest    : ${MANIFEST:-none} (resident=$RESIDENT)"
  echo "  cache-shadow: ${EXPERT_CACHE_SIM_MIB} MiB (telemetry only; no eviction)"
  echo "  stats-file  : ${EXPERT_STATS_FILE:-none}"
  echo "  pinned      : $([ "${PINNED:-0}" = "1" ] && echo on || echo off)"
  echo "  cache-ram   : ${CACHE_RAM_MIB} MiB"
  echo "  RAM-budget  : ${RAM_BUDGET_GIB} GiB (cgroup MemoryMax; High ${MEM_HIGH} MiB)"
  echo "  system      : MemAvailable ${mem_avail_gib} GiB, VRAM free ${gpu_free_gib} GiB, gpu-busy=${gpu_busy}"
  echo "  guards      : $([ "$guard_fail" = "1" ] && echo WOULD BLOCK || echo OK)"
  exit 0
fi

exec systemd-run --user --scope --unit="expertpin-test-$(date +%s)" \
  -p MemoryHigh="${MEM_HIGH}M" \
  -p MemoryMax="${RAM_BUDGET_MIB}M" \
  "$BIN_DIR/llama-server" \
  -m "$MODEL" \
  --jinja \
  -ngl "$NGL" \
  --n-cpu-moe "$NCMOE" \
  -c "$CTX" \
  --cache-type-k "$KVT" --cache-type-v "$KVT" \
  -fa on \
  -t "$THREADS" \
  --cache-ram "$CACHE_RAM_MIB" \
  --host 127.0.0.1 --port "$PORT" \
  "${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"}" \
  "${EXPERT_ARGS[@]+"${EXPERT_ARGS[@]}"}" \
  "$@"
