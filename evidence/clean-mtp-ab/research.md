# X research — 2026-09-06, completed before testing or GPU lifecycle

Read-only x_search used (from_date 2026-09-01), query included Qwen3.8-flash-next, Flash, AD-4.27bpw, MTP n_max, expert cache, EXL3 and INT4/W4A16.

Useful lead: https://x.com/ItsmeAjayKV/status/2094895868621345162 — search reports an n_max sweep and a trade-off between speed and acceptance. Transfer only the experimental idea: hold workload/placement/sampling fixed and measure multiple draft depths, not external throughput claims.
Other leads: https://x.com/inovelloE/status/2096577582741238212 (expert cache + MTP); https://x.com/stepbystepnomad/status/2094644187895595442 (sampling/acceptance). Search synthesis contains unverified model/hardware details; none are accepted as local evidence. No exact AD-4.27bpw result surfaced. No browser or local research service was opened.

Selected slice: clean production-graphs, no trace/shadow 256-token decode at n_max 4 / 8 / 16, repeated fixed prompt/seed. Existing server timings.draft_by_depth supplies acceptance counts without diagnostic tracing. First measure baseline at 4; then 8; then 16 if budgets/time allow. No placement or quantization change. Lower depth acceptance is not a standalone throughput prediction. Repeated/order-controlled results needed; OS cache is not flushed and remains a confound.
