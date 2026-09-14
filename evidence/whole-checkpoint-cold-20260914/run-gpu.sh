#!/usr/bin/env bash
# One bounded physical probe; outer shell remains outside both memory scopes.
set -euo pipefail
cd /home/seppe/Projects/expertpin
OUT=.hermes-work/daily-20260914
PREP=/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh
[ ! -e "$OUT/server-unit.txt" ] || { printf 'Refusing reuse of run artifacts\n'; exit 2; }
systemctl --user is-active --quiet jetski.service
if systemctl --user is-active --quiet expertpin-dev.service; then
    printf 'Refusing active dev server\n'; exit 2
fi
cleanup() {
    set +e
    systemctl --user stop expertpin-daily-client-20260914.scope
    if [ -f "$OUT/server-unit.txt" ]; then
        read -r scope < "$OUT/server-unit.txt"
        case "$scope" in
            expertpin-test-*.scope) systemctl --user stop "$scope" ;;
            *) printf 'Unexpected server scope identifier\n' ;;
        esac
    fi
    systemctl --user start jetski.service
    systemctl --user is-active jetski.service
    bash "$PREP" --restore
    systemctl --user show qli.service jetski.service headless-chromium.service -p Id -p ActiveState -p LoadState
}
trap cleanup EXIT
# Scope includes harness + curl + diagnostic subprocesses; launcher creates a
# separate server36G scope. Sum of caps40G. 900s + bounded cleanup <1800s.
timeout --signal=TERM --kill-after=90 900 \
    systemd-run --user --scope --unit=expertpin-daily-client-20260914.scope \
    -p MemoryMax=4G -p MemorySwapMax=0 \
    python3 .hermes-work/daily-20260914/run-ud.py
