# Cold-cache A/B — 2026-09-11 (11:55–12:05 CEST)

Question: worst-case first tokens after a controlled page-cache drop
(POSIX_FADV_DONTNEED) on ps-iq2xxs, vs warm, within 40/28 budgets.

## Result: both arms green; cold arm slower but far above target

| arm | load→ready | req 1 decode | req 2 decode | RAM peak | VRAM peak |
|---|---|---|---|---|---|
| cold (--cold-cache) | 36 s | 35.54 tok/s | 49.45 tok/s | 32.41/40 GiB | 18.89/28 GiB |
| warm | 32 s | 43.56 tok/s | 50.40 tok/s | 32.41/40 GiB | 18.81/28 GiB |

The fadvise completed on the exact model file (75,216,526,912 bytes,
read-only fd). Both arms: identical config (nmax4/DRAFT1, seed 42, temp 0,
2x256 decode); only the fadvise differs. memory.events max/oom/oom_kill
zero in both.

## Honest reading: AMBIGUOUS — advisory drop, cold state NOT verified

- DONTNEED is advisory and no mincore residency snapshot was taken before
  the launcher, so it is NOT proven that the load was fully cold: the
  loader mmaps and may fault only the subset it needs, and other same-day
  runs left the host page cache rich (buff/cache ~42 GiB around the run).
- The observed deltas (load 36 vs 32 s; first request 35.5 vs 43.6 tok/s,
  ~19% slower; second request essentially equal) are consistent with a
  partially cold load, but also with ordinary run-to-run variance at n=1.
- What DOES stand practically: even the slower cold-arm request decodes at
  35.5 tok/s — the 20 tok/s floor holds in the worst observed condition of
  this slice.

Next refinement (future slice): mincore residency check on the model file
between the drop and the launcher (pattern already proven in
scripts/expert_bw_calib.py, 2026-09-06) to turn "advisory drop" into
"verified cold" — then re-run the A/B for a defensible worst-case number.

## Artifacts

- outcome.json (machine-readable incl. honest_reading), runs
  run-20260911T115540 (cold) and run-20260911T120004 (warm).
- Harness: 7d14cc14 (5 CPU-only tests; also restores the needle-gate
  harness code that 3b9c8928's dir checkout had silently reverted — caught
  by the 10 failing needle-test imports, fixed from d3fe9fc2, 23/23 green).
