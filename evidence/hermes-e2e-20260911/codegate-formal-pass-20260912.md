# UD-Q4_K_XL code-precision gate — FORMAL PASS (run-20260912T035703-e2e)

Third attempt of the night, after two runner-side harness bugs were found
and fixed (8b068271: module-constant work deadline; 1a7bdf09: marker-task
criteria applied to the code task). All four known lineage traps closed:
construction-time start stamp, GPU-idle clock burn (7d57edcc), module
WORK_SECONDS override, task-kind criteria.

## Result

- status: completed; runner exit 0; hermes_returncode 0; wall 853.5 s.
- hermes_stdout_has_marker true (expected output after query echo);
  hermes_ran_shell_tool true (script written + python3 execution evidence).
- INDEPENDENT ARTIFACT PROBE (in-runner this time): codegate_script_rc 0,
  codegate_script_output exactly `code-gate-ok-7391`, codegate_pass true.
- Stale-artifact guard active: both /tmp gate files removed before the
  client started (recorded in e2e-summary.json).

This closes open item #1 of evidence/dev-serving-20260911/
ud-evaluation-outcome.json ("formal clean re-run of the UD code gate with
the fixed runner").

## Budgets (samples.jsonl)

- cgroup memory.peak 34,794,840,064 B = 32.40 GiB / 40 GiB (MemoryMax 36G).
- VRAM device sample peak 24,841 MiB = 24.26 GiB / 28 GiB.
- Swap sample max 5,284,798,464 B ≈ 4.92 GiB (reported separately, outside
  the 40 GiB RAM budget; MemoryHigh reclaim pressure during the long agent
  decode; higher than the 0.73 GiB of run-034046).
- memory.events max/oom/oom_kill 0 across all samples.

## Pause-conform lifecycle (second production proof)

- jetski.service stop requested 03:57:05 before GPU work; restored +
  is-active verified after; qli masked → skipped, recorded
  ("masked/not-found — not stopped (pause mandate)"), never started.
- cleanup_errors []; headless-chromium restored active; no llama-server
  left running (pgrep rc=1).

## Night totals (03:00 cycle)

- 2 harness bugs found+fixed with RED-first regression tests (7 new tests
  total across two files); 4 commits pushed (5f9f78ff, 4af0f879, 8b068271,
  1a7bdf09); code gate: 2 post-hoc PASSes + 1 formal in-runner PASS.
