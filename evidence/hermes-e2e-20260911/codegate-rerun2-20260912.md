# Runner-incident 2026-09-12 03:40 — criteria bug; agent+artifact PASS again (post-hoc)

Run-20260912T034046-e2e (UD-Q4_K_XL code gate, work-budget fix active):

- RUNNER: completed its full lifecycle for the first time in this gate's
  history (hermes_wall_seconds 756.1, well inside the 2700 s budget — the
  work_budget() fix from 8b068271 held). Cleanup again pause-conform and
  clean (cleanup_errors [], Jetski restored+active, headless active).
- AGENT: PASS on the transcript evidence — session 20260912_034154_5fc3fc,
  12m35s, 4 messages / 2 tool calls: heredoc-writes /tmp/e2e-codegate.py,
  runs python3, final answer exactly `code-gate-ok-7391`.
- HARNESS VERDICT: client_failed — WRONG. ok_turn1 applied the MARKER-task
  criteria ('hermes-e2e-ok' + printf) to a code task, so both flags were
  false and the codegate artifact probe (behind ok_turn1) never executed.
  Fourth trap in this lineage. Fixed by task-kind specific turn1_checks()
  (commit follows); regression test replays the REAL captured stdout of
  this run (tests/test-e2e-turn1-criteria-20260912.py).
- ARTIFACT (post-hoc, committer-run): /tmp/e2e-codegate.py = required
  for-loop over range(3) summing iterations, prints exactly
  `code-gate-ok-7391`, python3 rc=0. Stale-artifact guard was active
  (both /tmp files removed at run start, recorded in e2e-summary.json).

Budgets: cgroup memory.peak 34,796,847,104 B = 32.40 GiB / 40; VRAM sample
peak 24,855 MiB = 24.27 GiB / 28; swap sample max ~0.73 GiB (separate);
memory.events max/oom/oom_kill 0; MemAvailable at guard ~50.4 GiB Tier A.

Open: one clean re-run for the formal in-runner PASS (all four known traps
fixed: construction-start, GPU-idle clock burn, module-constant deadline,
marker-criteria-on-code-task).
