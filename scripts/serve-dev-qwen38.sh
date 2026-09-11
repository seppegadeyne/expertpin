#!/usr/bin/env bash
# expertpin dev-serving launcher — resident, bounded, for daily Hermes dev.
#
# Differences from scripts/run-qwen38-flash-next.sh (the benchmark launcher):
#   - Runs as a LONG-LIVED systemd user service (expertpin-dev.service),
#     not a one-shot scope: no qli interruption per run.
#   - Same hard guards: refuses to start unless the host is idle enough;
#     cgroup MemoryHigh/MemoryMax bound the server.
#   - CTX=65536 default (Hermes >=64K requirement; proven within budget).
#   - Skips the MTP drafter by default (dev sessions are tool-loop bound,
#     not decode-throughput bound; set DRAFT=1 to enable).
#   - Endpoint 127.0.0.1:8102/v1; model id = the GGUF path.
#
# The qubic miner (qli.service) is NOT stopped by this launcher: both fit
# the host simultaneously in dev use (server cgroup 36G; miner ~1G). If
# MemAvailable is too low, this refuses to start — check qli manually.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/Models/qwen3.8-flash-next-ps-iq2xxs}"
MODEL="${MODEL:-$MODEL_DIR/Qwen3.8-Flash-Next-IQ2_XXS.gguf}"
CTX="${CTX:-65536}"
NCMOE="${NCMOE:-36}"
NGL="${NGL:-99}"
THREADS="${THREADS:-16}"
PORT="${PORT:-8102}"
KVT="${KVT:-q8_0}"
RAM_BUDGET_GIB="${RAM_BUDGET_GIB:-36}"
GPU_NEED_GIB="${GPU_NEED_GIB:-24}"
CACHE_RAM_MIB="${CACHE_RAM_MIB:-512}"
DRAFT="${DRAFT:-0}"
DRAFT_NMAX="${DRAFT_NMAX:-4}"
DRAFT_MODEL="${DRAFT_MODEL:-$HOME/Models/qwen3.8-flash-next/mtp-drafter/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf}"
DRY="${DRY:-0}"

BIN_DIR="${BIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)/build-sm120/bin}"
[ -x "$BIN_DIR/llama-server" ] || { echo "No llama-server build in $BIN_DIR"; exit 1; }
[ -f "$MODEL" ] || { echo "Model not found: $MODEL"; exit 1; }

# --- guards (identical philosophy to the benchmark launcher) ---
mem_avail_kib=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
mem_avail_gib=$(( mem_avail_kib / 1024 / 1024 ))
gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
gpu_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
gpu_free_gib=$(( gpu_free_mib / 1024 ))

guard_fail=0
[ "$mem_avail_gib" -lt $(( RAM_BUDGET_GIB + 4 )) ] && {
  echo "GUARD: MemAvailable ${mem_avail_gib} GiB < budget ${RAM_BUDGET_GIB} + 4 headroom"; guard_fail=1; }
[ "${gpu_util:-0}" -ge 50 ] && {
  echo "GUARD: GPU busy (${gpu_util}%)"; guard_fail=1; }
[ "${gpu_free_gib:-0}" -lt "$GPU_NEED_GIB" ] && {
  echo "GUARD: free VRAM ${gpu_free_gib} GiB < ${GPU_NEED_GIB} GiB"; guard_fail=1; }

if [ "$guard_fail" = "1" ] && [ "$DRY" != "1" ]; then
  echo "Refusing to start (dev unit should restart when the host frees up)."
  exit 2
fi

DRAFT_ARGS=()
if [ "$DRAFT" = "1" ] && [ -f "$DRAFT_MODEL" ]; then
  DRAFT_ARGS=(-md "$DRAFT_MODEL" -ngld 99 --spec-type "mtp:n_max=${DRAFT_NMAX}")
fi

export GGML_CUDA_NO_PINNED=1
cd "$(dirname "$0")/.."
RAM_BUDGET_MIB=$(( RAM_BUDGET_GIB * 1024 ))

if [ "$DRY" = "1" ]; then
  echo "DRY-RUN dev plan: ctx=$CTX port=$PORT draft=$DRAFT budget=${RAM_BUDGET_GIB}G"
  echo "  guards: $([ "$guard_fail" = "1" ] && echo WOULD BLOCK || echo OK)"
  exit 0
fi

exec "$BIN_DIR/llama-server" \
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
  "$@"
