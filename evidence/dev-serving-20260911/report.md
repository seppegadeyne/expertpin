# Resident dev-serving unit — 2026-09-11 (16:55–17:05 CEST)

Question: does a resident systemd user unit around the guarded launcher
serve Hermes dev sessions with fast warm reuse, within budget caps,
without touching the miner?

## Verdict: PASS

- Cold first agent query (shell tool): 84.4 s (session 20260911_165914_e79502).
- Warm second query on the SAME server instance (no restart): **15.9 s**
  with the exact expected answer — the resident unit's whole point.
- Unit state during: active, 0 restarts, listening on 127.0.0.1:8102;
  cgroup MemoryCurrent 32.1 GiB / peak 33.0 GiB under MemoryHigh 33G /
  MemoryMax 36G / MemorySwapMax 16G.
- qli.service stayed ACTIVE throughout — dev serving coexists with the
  miner (GPU util 0% between queries; the launcher's VRAM guard would
  refuse to start if the miner held the GPU, by design).
- After stop: VRAM back to 2.3 GiB, MemAvailable 45 GiB — the 03:00
  benchmark cycle stays unblocked. The unit is NOT auto-enabled.

## Files

- scripts/serve-dev-qwen38.sh — dev launcher (CTX 65536 default, no
  drafter by default, same guard philosophy, no qli interruption).
- scripts/expertpin-dev.service — systemd user unit (copy to
  ~/.config/systemd/user/). Opt-in: `systemctl --user enable --now
  expertpin-dev.service`.
- One iteration: first start failed on `--keep-alive 0` (not a flag in
  this build); removed — slot persistence across requests is what the
  warm reuse proves anyway.

## Limitations

Two-query warm proof, not a multi-hour soak; miner coexistence measured
only at idle GPU util.
