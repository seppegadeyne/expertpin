# Verified-cold residency — 2026-09-11 (13:38 CEST)

Question: is the cold-cache arm measurably cold (mincore residency), and
what is the decode speed from a verified-cold state?

## Verdict: COLD VERIFIED — 0/18,363,410 pages resident after DONTNEED

- Whole-file batched mincore (PROT_NONE MAP_SHARED, d09d914c) counted
  exactly 0 resident pages of the 71 GiB GGUF after the fadvise drop:
  resident_ratio 0.0, verdict COLD against the 10% threshold.
- From this PROVEN cold state: first decode 33.27 tok/s, second 50.75
  tok/s (ps-iq2xxs, nmax4). The 20 tok/s floor holds verified-cold.
- RAM peak 32.41/40 GiB, VRAM 18.68/28 GiB; clean run, no OOM events.

## What this settles

- The earlier advisory-drop arm (run 115540, 35.54 tok/s) is
  retroactively validated: DONTNEED on this file/kernel evicts fully, so
  that measurement was genuinely cold too.
- Verified-cold worst case for the default-switch decision: 33.3 tok/s.

## Limitations

- Snapshot not atomic (nothing else touched the file during the window).
- n=1 verified-cold run, single prompt.
- mincore observes page-cache residency only (not GPU caches).

## Artifacts

- verified-cold-outcome.json, run-20260911T133852. Harness: d09d914c
  (5 CPU-only tests incl. drop-must-not-increase-residency).
