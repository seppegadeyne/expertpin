# Review 2026-09-11 — sustained-decode harness slice

Review attempt deleg_251bbce0 (11:20): FAILED — HTTP 429 usage limit after
3 retries, 10.0s. Fourth 429-blocked review attempt today (after deleg_1901d667,
deleg_e1f13ec2, deleg_a025517a); same mode as the 03:00 and prior-day runs.
Not treated as a blocker; the review pin stays unchanged.

Parent self-verification instead of an independent second opinion:

1. The harness is a byte-level copy of evidence/clean-mtp-ab/run-guarded.py
   except exactly one line (verified with diff against the committed base):
   TOKENS = int(os.environ.get('SUSTAINED_TOKENS', '256')). The base lineage
   is twice-reviewed code that has physically completed multiple guarded
   runs (checkpoint A/B 03:00, tool-call gate, needle gate today).
2. `os` is imported above the constant (line 14 vs 28), so the env read is
   safe at module load; a non-integer SUSTAINED_TOKENS value raises
   ValueError during import — before any host change, subprocess spawn, or
   service stop (fail-closed, verified by the CPU-only import test).
3. validate_completion compares completion_tokens == TOKENS (module
   constant), so the fail-closed length/finish_reason/usage checks scale
   with the env override; budget validation (cgroup MemoryMax, RAM/VRAM
   caps, memory.events) is untouched.
4. Timing budgets: measured decode at 31.5-50 tok/s gives 10-33 s per
   request at 512/1024 tokens; REQUEST_SECONDS 180 and WORK_SECONDS 660
   held with margin (actual run walls: 65-90 s total including model load).

No second independent opinion this run.
