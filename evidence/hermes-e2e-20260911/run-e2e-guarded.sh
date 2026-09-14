#!/usr/bin/env bash
# Physical Hermes E2E run (UD code/multi-turn rerun): the runner executes
# INSIDE a dedicated 4G client scope so harness + hermes client + curl RAM is
# kernel-accounted; the launcher creates the separate 36G server scope.
# Sum of caps = 40G. Work deadline is enforced by run-e2e.py itself (<=1800 s,
# hard-capped); the outer timeout below is only a cleanup backstop.
# The outer shell stays OUTSIDE both memory scopes.
set -euo pipefail
cd /home/seppe/Projects/expertpin
PREP=/home/seppe/.hermes/profiles/expertpin/scripts/gpu-host-prep.sh
UNIT="expertpin-e2e-client-$(date +%Y%m%dT%H%M%S).scope"
export E2E_CLIENT_UNIT="$UNIT"
if systemctl --user is-active --quiet expertpin-dev.service; then
    printf 'Refusing active dev server\n'; exit 2
fi
cleanup() {
    set +e
    systemctl --user stop "$UNIT"
    bash "$PREP" --restore
    systemctl --user show qli.service jetski.service headless-chromium.service \
        -p Id -p ActiveState -p LoadState
}
trap cleanup EXIT
# 1800 s work cap + bounded cleanup reserve; SIGTERM lands in the runner's
# deferred-signal path, so cleanup (incl. miner restore) still runs.
timeout --signal=TERM --kill-after=90 1980 \
    systemd-run --user --scope --unit="$UNIT" \
    -p MemoryMax=4G -p MemorySwapMax=0 \
    python3 evidence/hermes-e2e-20260911/run-e2e.py
