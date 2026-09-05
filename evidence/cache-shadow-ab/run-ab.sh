#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/seppe/Projects/expertpin
PREP=/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh
LAUNCHER="$ROOT/scripts/run-qwen38-flash-next.sh"
BASE="$ROOT/evidence/cache-shadow-ab"
RUN_ID="$(date +%Y%m%dT%H%M%S%z)"
RUN_DIR="$BASE/$RUN_ID"
PORT=8102
PREDICT_TOKENS=128
WARM_TOKENS=32
SAMPLES=3
CTX=8192
NCMOE=30
RAM_BUDGET_GIB=35
CACHE_RAM_MIB=512
VRAM_CAP_MIB=$((28 * 1024))
HIT_RATE_GATE=0.90
OVERHEAD_GATE_PCT=5
# Tier A host guard (Seppe protocol rev 2026-09-04): >=34 GiB. Hard bounds: cgroup MemoryMax=35 GiB + VRAM cap 28672 MiB.
MEM_GUARD_KIB=$((34 * 1024 * 1024))
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$BASE/latest-dir.txt"
exec > >(tee -a "$RUN_DIR/session.log") 2>&1

qli_restart_needed=0
prep_ran=0
cleanup_done=0
server_launcher_pid=""
server_scope=""
monitor_pid=""
watchdog_pid=""
headless_initial=""

now() { date --iso-8601=seconds; }
mem_avail_kib() {
    local key value unit
    while read -r key value unit; do
        if [[ "$key" == "MemAvailable:" ]]; then
            printf '%s\n' "$value"
            return 0
        fi
    done < /proc/meminfo
    return 1
}
record_mem() {
    local label="$1" value
    value="$(mem_avail_kib)"
    printf '%s=%s KiB at %s\n' "$label" "$value" "$(now)" | tee -a "$RUN_DIR/memory-checkpoints.log"
    free -g >> "$RUN_DIR/memory-checkpoints.log"
}
stop_server() {
    if [[ -n "$server_scope" ]]; then
        systemctl --user stop "$server_scope" 2>/dev/null || true
    fi
    if [[ -n "$server_launcher_pid" ]] && kill -0 "$server_launcher_pid" 2>/dev/null; then
        kill -TERM "$server_launcher_pid" 2>/dev/null || true
    fi
    if [[ -n "$server_launcher_pid" ]]; then
        wait "$server_launcher_pid" 2>/dev/null || true
    fi
    if [[ -n "$monitor_pid" ]] && kill -0 "$monitor_pid" 2>/dev/null; then
        kill -TERM "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    server_launcher_pid=""
    server_scope=""
    monitor_pid=""
}
restore_all() {
    local rc=0 active journal_start
    if [[ "$cleanup_done" == 1 ]]; then
        return 0
    fi
    stop_server
    if [[ "$qli_restart_needed" == 1 ]]; then
        journal_start="$(now)"
        printf 'QLI_START_REQUEST=%s\n' "$journal_start" | tee -a "$RUN_DIR/qli.log"
        if ! systemctl --user start qli.service; then
            printf 'QLI_START_FAILED=%s\n' "$(now)" | tee -a "$RUN_DIR/qli.log"
            rc=1
        fi
        active=""
        for _ in {1..30}; do
            active="$(systemctl --user is-active qli.service 2>&1 || true)"
            [[ "$active" == "active" ]] && break
            sleep 1
        done
        printf 'QLI_ACTIVE=%s at %s\n' "$active" "$(now)" | tee -a "$RUN_DIR/qli.log"
        journalctl --user -u qli.service --since "$journal_start" --no-pager > "$RUN_DIR/qli-journal-since-start.log" 2>&1 || true
        [[ "$active" == "active" ]] || rc=1
    fi
    if [[ "$prep_ran" == 1 ]]; then
        printf 'HOST_RESTORE_START=%s initial=%s\n' "$(now)" "$headless_initial" | tee -a "$RUN_DIR/restore.log"
        if [[ "$headless_initial" == "active" ]]; then
            if ! bash "$PREP" --restore > "$RUN_DIR/host-restore.log" 2>&1; then
                rc=1
            fi
        else
            printf '[gpu-host-prep] restore skipped: headless-chromium.service was initially %s\n' "$headless_initial" > "$RUN_DIR/host-restore.log"
        fi
        printf 'HOST_RESTORE_END=%s\n' "$(now)" | tee -a "$RUN_DIR/restore.log"
    fi
    record_mem mem_available_after_restore_kib
    cleanup_done=1
    return "$rc"
}
on_exit() {
    local original_rc=$?
    if [[ -n "$watchdog_pid" ]]; then
        kill "$watchdog_pid" 2>/dev/null || true
    fi
    if ! restore_all; then
        original_rc=90
    fi
    printf 'FINAL_RC=%s at %s\n' "$original_rc" "$(now)" | tee -a "$RUN_DIR/result.log"
    exit "$original_rc"
}
on_timeout() {
    printf 'GLOBAL_TIMEOUT_30M at %s\n' "$(now)" | tee -a "$RUN_DIR/result.log"
    exit 124
}
trap on_timeout TERM INT
trap on_exit EXIT
(
    sleep 1800
    kill -TERM $$
) &
watchdog_pid=$!

printf 'RUN_ID=%s\nRUN_DIR=%s\nRUN_START=%s\nHEAD=%s\nPORT=%s\nPREDICT_TOKENS=%s\nWARM_TOKENS=%s\nSAMPLES=%s\nCTX=%s\nNCMOE=%s\nRAM_BUDGET_GIB=%s\nCACHE_RAM_MIB=%s\nVRAM_CAP_MIB=%s\nHIT_RATE_GATE=%s\nOVERHEAD_GATE_PCT=%s\n' \
    "$RUN_ID" "$RUN_DIR" "$(now)" "$(git -C "$ROOT" rev-parse HEAD)" "$PORT" "$PREDICT_TOKENS" "$WARM_TOKENS" "$SAMPLES" "$CTX" "$NCMOE" "$RAM_BUDGET_GIB" "$CACHE_RAM_MIB" "$VRAM_CAP_MIB" "$HIT_RATE_GATE" "$OVERHEAD_GATE_PCT" \
    | tee "$RUN_DIR/run-meta.log"
printf 'QLI_INITIAL=%s at %s\n' "$(systemctl --user is-active qli.service 2>&1 || true)" "$(now)" | tee "$RUN_DIR/qli.log"
headless_initial="$(systemctl --user is-active headless-chromium.service 2>&1 || true)"
printf 'HEADLESS_INITIAL=%s at %s\n' "$headless_initial" "$(now)" | tee -a "$RUN_DIR/run-meta.log"
record_mem mem_available_before_cleanup_kib
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits > "$RUN_DIR/gpu-before-cleanup.csv"

if pgrep -x llama-server > /dev/null; then
    printf 'ABORT: pre-existing llama-server process\n' | tee -a "$RUN_DIR/result.log"
    exit 10
fi
if [[ "$(systemctl --user is-active qli.service 2>&1 || true)" != "active" ]]; then
    printf 'ABORT: qli was not active at run start\n' | tee -a "$RUN_DIR/result.log"
    exit 11
fi

printf 'HOST_PREP_START=%s\n' "$(now)" | tee "$RUN_DIR/host-prep-times.log"
prep_ran=1
stale_scope=""
while read -r unit _; do
    [[ -n "$unit" ]] && stale_scope="$unit"
done < <(systemctl --user list-units --type=scope --state=running --no-legend --plain "expertpin-test-*.scope" 2>/dev/null || true)
if [[ -n "$stale_scope" ]]; then
    printf "ABORT: pre-existing expertpin-test scope: %s\n" "$stale_scope" | tee -a "$RUN_DIR/result.log"
    exit 13
fi
bash "$PREP" > "$RUN_DIR/host-prep.log" 2>&1
printf 'HOST_PREP_END=%s\n' "$(now)" | tee -a "$RUN_DIR/host-prep-times.log"
record_mem mem_available_after_cleanup_before_qli_kib

qli_restart_needed=1
printf 'QLI_STOP_REQUEST=%s\n' "$(now)" | tee -a "$RUN_DIR/qli.log"
systemctl --user stop qli.service
qli_active=""
for _ in {1..30}; do
    qli_active="$(systemctl --user is-active qli.service 2>&1 || true)"
    [[ "$qli_active" == "inactive" ]] && break
    sleep 1
done
printf 'QLI_AFTER_STOP=%s at %s\n' "$qli_active" "$(now)" | tee -a "$RUN_DIR/qli.log"
[[ "$qli_active" == "inactive" ]] || { printf 'ABORT: qli did not stop\n' | tee -a "$RUN_DIR/result.log"; exit 12; }
sleep 4

check_arm_guards() {
    local label="$1" gpu_util gpu_used gpu_free mem
    IFS=, read -r gpu_util gpu_used gpu_free < <(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits)
    gpu_util="${gpu_util// /}"
    gpu_used="${gpu_used// /}"
    gpu_free="${gpu_free// /}"
    mem="$(mem_avail_kib)"
    printf 'ARM_GUARD label=%s gpu_util_pct=%s vram_used_mib=%s vram_free_mib=%s mem_available_kib=%s at %s\n' \
        "$label" "$gpu_util" "$gpu_used" "$gpu_free" "$mem" "$(now)" | tee -a "$RUN_DIR/guards.log"
    (( gpu_util < 5 )) || { printf 'ABORT: %s GPU util is not below 5%%\n' "$label" | tee -a "$RUN_DIR/result.log"; exit 20; }
    (( mem > MEM_GUARD_KIB )) || { printf 'ABORT: %s MemAvailable is not above 34 GiB (Tier A)\n' "$label" | tee -a "$RUN_DIR/result.log"; exit 21; }
}

make_request() {
    local tokens="$1" output="$2"
    printf '{"prompt":"Explain in detail how a bounded multi-tier expert cache can accelerate mixture-of-experts inference while preserving exact model output. Cover host RAM, NVMe misses, GPU residency, and bandwidth-aware routing.","n_predict":%s,"temperature":0.0,"seed":42,"stream":false,"cache_prompt":false}\n' "$tokens" > "$output.request.json"
    curl --silent --show-error --fail-with-body --max-time 600 \
        -H 'Content-Type: application/json' \
        --data-binary @"$output.request.json" \
        "http://127.0.0.1:$PORT/completion" > "$output"
}

make_arm_request() {
    local tokens="$1" output="$2" arm_dir="$3"
    if ! make_request "$tokens" "$output"; then
        if [[ -f "$arm_dir/vram-cap-violation.log" ]]; then
            printf 'ABORT: absolute 28-GiB VRAM cap exceeded in %s\n' "$(basename "$arm_dir")" | tee -a "$RUN_DIR/result.log"
            exit 32
        fi
        printf 'ABORT: request failed in %s\n' "$(basename "$arm_dir")" | tee -a "$RUN_DIR/result.log"
        exit 33
    fi
    if [[ -f "$arm_dir/vram-cap-violation.log" ]]; then
        printf 'ABORT: absolute 28-GiB VRAM cap exceeded in %s\n' "$(basename "$arm_dir")" | tee -a "$RUN_DIR/result.log"
        exit 32
    fi
}

run_arm() {
    local label="$1" sim_mib="$2" measured="$3" arm_dir dry_rc health_ready i
    arm_dir="$RUN_DIR/$label"
    mkdir -p "$arm_dir"
    check_arm_guards "$label"
    record_mem "mem_available_before_${label}_kib"
    local run_env=(
        DRAFT=0
        CTX="$CTX"
        NCMOE="$NCMOE"
        NGL=99
        KVT=q8_0
        RAM_BUDGET_GIB="$RAM_BUDGET_GIB"
        GPU_NEED_GIB=28
        CACHE_RAM_MIB="$CACHE_RAM_MIB"
        PORT="$PORT"
        BIN_DIR="$ROOT/build-sm120/bin"
        EXPERT_CACHE_SIM_MIB="$sim_mib"
        EXPERT_STATS_FILE="$arm_dir/expert-stats.json"
    )
    printf 'ARM_START label=%s sim_mib=%s measured=%s at %s\n' "$label" "$sim_mib" "$measured" "$(now)" | tee "$arm_dir/meta.log"
    env "${run_env[@]}" DRY=1 "$LAUNCHER" > "$arm_dir/dry-run.log" 2>&1
    printf 'timestamp,mem_available_kib,gpu_util_pct,vram_used_mib,vram_free_mib,scope_memory_current_bytes\n' > "$arm_dir/runtime-metrics.csv"
    (
        while true; do
            local_ts="$(now)"
            local_mem="$(mem_avail_kib)"
            IFS=, read -r local_gpu_util local_gpu_used local_gpu_free < <(
                nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader,nounits 2>/dev/null || printf ',,'
            )
            local_gpu_util="${local_gpu_util// /}"
            local_gpu_used="${local_gpu_used// /}"
            local_gpu_free="${local_gpu_free// /}"
            local_scope=""
            while read -r unit _; do
                [[ -n "$unit" ]] && local_scope="$unit"
            done < <(systemctl --user list-units --type=scope --state=running --no-legend --plain 'expertpin-test-*.scope' 2>/dev/null || true)
            local_scope_memory=""
            if [[ -n "$local_scope" ]]; then
                local_scope_memory="$(systemctl --user show "$local_scope" --value -p MemoryCurrent 2>/dev/null || true)"
            fi
            printf '%s,%s,%s,%s,%s,%s\n' "$local_ts" "$local_mem" "$local_gpu_util" "$local_gpu_used" "$local_gpu_free" "$local_scope_memory" >> "$arm_dir/runtime-metrics.csv"
            if [[ "$local_gpu_used" =~ ^[0-9]+$ ]] && (( local_gpu_used > VRAM_CAP_MIB )); then
                printf 'VRAM_CAP_EXCEEDED used_mib=%s cap_mib=%s scope=%s at %s\n' "$local_gpu_used" "$VRAM_CAP_MIB" "${local_scope:-unknown}" "$local_ts" > "$arm_dir/vram-cap-violation.log"
                if [[ -n "$local_scope" ]]; then
                    systemctl --user stop "$local_scope" 2>/dev/null || true
                fi
                break
            fi
            sleep 0.5
        done
    ) &
    monitor_pid=$!
    env "${run_env[@]}" "$LAUNCHER" > "$arm_dir/server.log" 2>&1 &
    server_launcher_pid=$!
    health_ready=0
    for _ in {1..180}; do
        if ! kill -0 "$server_launcher_pid" 2>/dev/null; then
            break
        fi
        server_scope=""
        while read -r unit _; do
            [[ -n "$unit" ]] && server_scope="$unit"
        done < <(systemctl --user list-units --type=scope --state=running --no-legend --plain 'expertpin-test-*.scope' 2>/dev/null || true)
        if [[ -f "$arm_dir/vram-cap-violation.log" ]]; then
        break
        fi
        if curl --silent --show-error --fail --max-time 2 "http://127.0.0.1:$PORT/health" > "$arm_dir/health.json" 2>/dev/null; then
            health_ready=1
            break
        fi
        sleep 1
    done
    printf 'HEALTH_READY=%s scope=%s at %s\n' "$health_ready" "${server_scope:-unknown}" "$(now)" | tee -a "$arm_dir/meta.log"
    if [[ -f "$arm_dir/vram-cap-violation.log" ]]; then
        printf 'ABORT: absolute 28-GiB VRAM cap exceeded while loading %s\n' "$label" | tee -a "$RUN_DIR/result.log"
        exit 32
    fi
    [[ "$health_ready" == 1 ]] || { printf 'ABORT: %s server not healthy\n' "$label" | tee -a "$RUN_DIR/result.log"; exit 30; }

    printf 'WARM_START=%s\n' "$(now)" | tee -a "$arm_dir/meta.log"
    make_arm_request "$WARM_TOKENS" "$arm_dir/warm-response.json" "$arm_dir"
    printf 'WARM_END=%s\n' "$(now)" | tee -a "$arm_dir/meta.log"
    if [[ "$measured" == 1 ]]; then
        for ((i=1; i<=SAMPLES; ++i)); do
            printf 'SAMPLE_%s_START=%s\n' "$i" "$(now)" | tee -a "$arm_dir/meta.log"
            make_arm_request "$PREDICT_TOKENS" "$arm_dir/sample-$i.json" "$arm_dir"
            printf 'SAMPLE_%s_END=%s\n' "$i" "$(now)" | tee -a "$arm_dir/meta.log"
        done
    fi
    sleep 2
    if [[ -n "$server_scope" ]]; then
        systemctl --user show "$server_scope" -p MemoryCurrent -p MemoryPeak -p MemoryHigh -p MemoryMax -p OOMPolicy -p Result > "$arm_dir/cgroup-final.log" 2>&1 || true
    fi
    stop_server
    for _ in {1..30}; do
        [[ -f "$arm_dir/expert-stats.json" ]] && break
        sleep 1
    done
    [[ -f "$arm_dir/expert-stats.json" ]] || { printf 'ABORT: %s missing expert-stats.json\n' "$label" | tee -a "$RUN_DIR/result.log"; exit 31; }
    record_mem "mem_available_after_${label}_kib"
    printf 'ARM_END label=%s at %s\n' "$label" "$(now)" | tee -a "$arm_dir/meta.log"
}

check_arm_guards post_qli_stop
run_arm prime 0 0
run_arm shadow-off 0 1
run_arm shadow-32g 32768 1
python3 "$BASE/verify-ab-gates.py" "$RUN_DIR" "$HIT_RATE_GATE" "$OVERHEAD_GATE_PCT" > "$RUN_DIR/gates.log" 2>&1 || { cat "$RUN_DIR/gates.log"; printf "ABORT: A/B gates failed at %s\n" "$(now)" | tee -a "$RUN_DIR/result.log"; exit 40; }
cat "$RUN_DIR/gates.log"
printf 'RUN_RESULT=ab_complete at %s\n' "$(now)" | tee "$RUN_DIR/result.log"
restore_all
printf 'RUN_END=%s\n' "$(now)" | tee -a "$RUN_DIR/run-meta.log"
