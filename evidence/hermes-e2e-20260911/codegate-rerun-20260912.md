# Code-gate re-run 2026-09-12 03:18 — runner incident, agent+artifact PASS (post-hoc verified)

## What was run

Formal clean re-run of the UD-Q4_K_XL code-precision gate with the runner
that includes the 7d57edcc start-clock re-arm fix (open item #1 from
ud-evaluation-outcome.json). Command:
`E2E_CHECKPOINT=ud-q4kxl E2E_TASK=code E2E_NCMOE=40 E2E_REASONING=off
E2E_WORK_SECONDS=2700 python3 evidence/hermes-e2e-20260911/run-e2e.py`

## Result

- RUNNER: failed — TimeoutError('work deadline; cleanup reserve begins') at
  ~660 s after the re-armed start, mid-agent-run. Third timing trap in this
  lineage: sample() checks the MODULE constant WORK_SECONDS=660, while the
  runner wires run.work_seconds=2700 (used only by its own health deadline).
  Fixed after this run (work_budget() override; regression test
  tests/test-e2e-work-budget-20260912.py, RED-verified against the pre-fix
  code).
- AGENT: completed the task fully (Hermes transcript session
  20260912_031930_2ec64d, 6 messages): API call #1 (prefill 14,224 tok in,
  73 out — 832 s wall, cold cache), tool write_file (script, 186 chars),
  API call #2, tool terminal (python3 run, 62 chars output), API call #3,
  final answer finish_reason=stop. Turn timeline 03:19:31 → 03:33:36.
- ARTIFACT PROBE (independent, run post-hoc by the committer):
  /tmp/e2e-codegate.py contains the required for-loop over range(3) summing
  iteration numbers and prints exactly `code-gate-ok-7391`; python3 exit 0.
  Stale-artifact protection was ACTIVE in this run: both /tmp artifacts were
  removed before the client started (recorded in e2e-summary.json), so the
  script can only originate from this run's agent.
- VERDICT: code-precision PASS for UD-Q4_K_XL on the artifact evidence; the
  runner-side summary JSON did not record it (runner died first). The formal
  in-runner PASS requires one more run with the work-budget fix — queued.

## Budgets (samples.jsonl, run-20260912T031812-e2e)

- cgroup memory.peak 34,792,988,672 B = 32.40 GiB / 40 GiB cap (MemoryMax
  36 GiB); swap.current sample max 787,013,632 B ≈ 0.73 GiB (reported
  separately, outside the 40 GiB RAM budget).
- VRAM device sample peak 24,446 MiB ≈ 23.87 GiB / 28 GiB cap.
- memory.events max/oom/oom_kill 0 across all samples.
- MemAvailable at guard: ~50.4 GiB (Tier A green), GPU util 0-1% before load.

## Pause-conform cleanup — first production proof

- jetski.service stop requested 03:18:14 (before GPU work), restored +
  verified active after teardown; qli masked → skipped with recorded reason,
  NEVER started; cleanup_errors []; headless-chromium restored active.

## Iterations preserved

- 03:18:12 runner: module-constant work-deadline killed the runner at ~660 s
  (this incident; fix follows in the same night).
