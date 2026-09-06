# X intake — 2026-09-06, before effective tests

Read-only x_search completed before tests; no browser or model process opened.
Search: Qwen3.8-Flash-Next/Qwen3.8 Flash/flash-next, AD-4.27bpw, n-cpu-moe,
expert-cache, EXL3, MTP draft, INT4/W4A16, phase-tagged routing and cold/warm I/O.

Leads (search summaries only, not independently verified original posts):
- https://x.com/SamSchieds/status/2096508476499562784 — expert-cache implementation
  on AMD; hardware/speed claim is NOT evidence for our 40/28 GiB budgets.
- https://x.com/Oluwaphilemon1/status/2096411031664787833 — sustained long-context
  run on two 3090s; supports measuring workload depth, not importing throughput.
- https://x.com/TeksEdge/status/2096297314595864698 — hybrid expert offload lead;
  old 63GB-expert-bank vs whole-host confusion remains rejected.

No requested phase-tagged or random expert warm/cold I/O trace surfaced in this
search. The search provider calls some posts 'verified benchmarks'; we do NOT
adopt that label without independent inspection/reproduction.

Slice: CPU-only cache-residency snapshots + process storage-read accounting for
existing bounded buffered-read calibration. Not a routing trace, isolated physical
NVMe bandwidth, RAM bandwidth or GPU GEMM measurement. The fixed developer gate
permits GPU work only if the Step-1 baseline has not yet run; state proves it has.
No GPU run will be attempted. Phase-2 remains incomplete.
