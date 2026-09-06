# X research — 2026-09-06, completed before tests/GPU

Read-only x_search for Qwen3.8 Flash Next, AD4.27, CPU MoE, expert cache, MTP, EXL3 and runtime tracing. Leads:
- https://x.com/lefu777/status/2096059297046437921 : pure LRU, CPU routed experts without repack, MTP (search summary only).
- https://x.com/SamSchieds/status/2096508476499562784 : CPU/DDR5/cache implementation lead (search summary only).
- https://x.com/JonathanLeaders/status/2095715911072207016 : AD3.84 variant, not our AD4.27 measurement.

Search answer mixes speculative architecture/variant claims; these are NOT accepted as facts. No external performance numbers imported. Useful idea: trace actual workload phases before choosing transfer/cache policy. Implement request-scoped logical CPU tracing; GPU accesses and physical I/O remain separate evidence requirements. No browser tabs/processes launched by this research. No external writes.
