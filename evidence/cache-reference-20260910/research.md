# Read-only research intake — 2026-09-10

Completed before tests, delegation or any service stop/model load. `x_search`
queried 2026-09-07 through 2026-09-10 for Qwen3.8 Flash Next, AD-4.27bpw,
expert-cache/LRU, n-cpu-moe, EXL3, INT4/W4A16, MTP divergence and kimi-k3-in-c.
No browser was launched, no authenticated X interaction or external post made.

Leads returned by the search tool (NOT independently verified measurements):
- https://x.com/ViC305/status/2097723351993385463 — reported numerical kernel
  fixes and tool-parser-sensitive evaluation; motivates correctness and actual
  tool-call contracts rather than throughput-only acceptance.
- https://x.com/jvr0x/status/2097738328246198311 — reported MTP vocabulary trimming;
  possible later optimization, not appropriate before our divergence diagnosis.
- https://x.com/UrbanAstroFella/status/2097407843783921773 — reported PLE mmap versus
  resident accounting. PLE lookup traffic is not routed-expert weight traffic.
- https://x.com/grok/status/2097826983996428578 — second-hand Kimi streaming claim;
  not evidence of local performance or cache participation.

The search answer called several claims 'verified' and conflated n-gram rows
with experts; neither characterization is adopted. No external speed, accuracy,
context, RAM or VRAM number is used as an expertpin result.

Chosen slice: Seppe's requested source-grounded comparison of the local read-only
`third-party/kimi-k3-in-c` LRU/trace-sizing design against expertpin's advisor.
A CPU-only documentation/validation slice, no Kimi execution or weight downloads.
Concrete implementation prerequisites and budget/correctness limitations will be
recorded in `docs/research/kimi-cache-comparison.md`.
