## 2026-09-14 — UD code-gate + multi-turn rerun: PASS with client-inclusive RAM accounting

Repeat of the UD-Q4_K_XL Hermes code gate (the 2026-09-12 formal PASS), now closing the three open harness items from the 2026-09-13 steering and the 2026-09-14 night run.

Harness changes (commits 9f3c8152, 71bc47f5; TDD: 20 new CPU tests, all suites green):

- Total work deadline hard-capped at 1800 s — the retired 2700 s code default fails closed at import; no env override can exceed the cap.
- Client-inclusive cgroup accounting: the runner + hermes client + probes now execute inside a dedicated 4 GiB client scope; every guard sample records server+client kernel peaks and refuses a combined footprint above 40 GiB. Raw launches without the client scope are refused before any host change.
- Summary metadata now records the actual checkpoint/task/budget instead of hardcoded defaults, and a completed verdict propagates into the harness summary status. Caveat: the status fix landed after the physical run, so in that run's `summary.json` the top-level status still shows the old default; the authoritative verdict is `e2e.status: completed` / `e2e-summary.json`.

Physical run `evidence/hermes-e2e-20260911/run-20260914T104344-e2e` (UD-Q4_K_XL, NCMOE 40, CTX 65536, KV q8_0, MTP n_max=4, reasoning off = the dev default):

- Turn 1 code task: rc 0, wall 799.5 s; independent artifact probe `python3 /tmp/e2e-codegate.py` rc 0 printing exactly `code-gate-ok-7391` (required for-loop over range(3) present). Stale-artifact guard active.
- Turn 2 (`--resume latest`): rc 0, 10.6 s; reused the existing script without rewriting it, answered exactly the marker. Session continuity proven (8 messages / 4 tool calls across both turns).
- Total work ~888 s, inside the enforced 1800 s deadline.
- Budgets: server scope peak 32.41 GiB /36G cap, client scope peak 0.37 GiB /4G cap, combined kernel-peak upper bound 32.78 GiB /40 GiB; zero max/oom events in both scopes; client swap 0. VRAM sampled peak 24.05/28 GiB over 844 samples, max gap 2.04 s (the 2026-09-12 sampling-gap class is closed; sampling is still not a continuous-peak guarantee).
- Request timings are agent-workload numbers (reasoning off, 64K ctx): one dominant 14213-token cold prefill at 18.84 tok/s, decodes 5.56–18.29 tok/s on short outputs. Not the throughput benchmark; no new >=20 tok/s evidence claimed.
- Host lifecycle: Jetski (found `failed` from a deliberate 08:37 stop) was stopped/restored by the harness — verified active afterwards per the daytime mandate; headless-chromium restored active; qli skipped per the pause mandate (never started, now also code-guarded); cleanup_errors []; scope teardown verified; no llama-server/listener left.

Report: `evidence/hermes-e2e-20260911/report-20260914.md`. No delegate review this run (profile-level gpt-6-astra pause until ~2026-09-15); self-verified against transcripts and cgroup counters, pin unchanged.

Remaining on this issue: multi-hour dev soak on UD and longer multi-step agent trajectories (nice-to-haves); the first cold decode below 20 tok/s stands as recorded.
