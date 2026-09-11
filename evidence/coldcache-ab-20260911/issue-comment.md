Progress update 2026-09-11 (fourth morning slice) — cold-cache A/B: worst observed case still 35.5 tok/s within 40/28; cold state advisory, not verified.

State-file next step 2 done. Harness (7d14cc14): a --cold-cache flag on the sustained-decode lineage that requests whole-file POSIX_FADV_DONTNEED on the 75.2 GB GGUF (read-only fd) after the Tier A guard and before the dry check/scope launch. 5 CPU-only tests (drop semantics, single whole-file call, wiring, ordering, fail-closed); targeted ctest 2/2, clean-mtp 16/16.

Also in 7d14cc14 — a fix: the evidence-dir checkout in 3b9c8928 had silently reverted evidence/quality-gate-20260911/run-guarded.py to the 815448ae state, dropping the committed needle-gate code. Caught it because the 23-test suite started failing 10 needle imports; restored from d3fe9fc2, 23/23 green again.

Results (ps-iq2xxs, identical config both arms, only the fadvise differs; binary SHA 3235529f... unchanged):
- cold arm: fadvise completed on 75,216,526,912 bytes; load->ready 36 s; requests 35.54 / 49.45 tok/s; RAM 32.41/40 GiB, VRAM 18.89/28 GiB.
- warm arm: load->ready 32 s; requests 43.56 / 50.40 tok/s; RAM 32.41/40, VRAM 18.81/28.
- memory.events max/oom/oom_kill zero in both.

Honest reading (in outcome.json): DONTNEED is advisory and no mincore residency snapshot was taken, so a fully cold load is NOT proven — the loader mmaps and may fault only a subset, and the same-day page cache was rich. The deltas (4 s load, ~19% slower first request) are consistent with a partially cold load but also with n=1 variance. What stands practically: even the slower cold-arm request is 35.5 tok/s, far above the 20 tok/s floor, within all budgets. A future slice can wire the proven mincore check (scripts/expert_bw_calib.py lineage) before the launcher to convert advisory-drop into verified-cold and re-run.

Evidence: evidence/coldcache-ab-20260911/ (outcome.json, report.md, runs 115540 cold / 120004 warm incl. raw logs). Commits 7d14cc14, 4a844ccc, af4a62c0 pushed and read back exactly.

Remaining next steps: 64K context within 40/28, Hermes end-to-end slice.
