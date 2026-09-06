# X research — 2026-09-06
Completed before tests, review or GPU lifecycle. x_search read-only; no browser or local tabs started.

Leads: https://x.com/inovelloE/status/2096577582741238212 (expert cache / CPU MoE / MTP); https://x.com/xueyu1125/status/2096574113943089359 (mmap/on-demand host data). Search output contains unverified architecture/quant and performance assertions; none are imported as evidence or local metrics. No raw pageable CUDA transfer attribution was returned. Useful idea: separate actual CPU/GPU transfer calls from logical expert routing; GPU-resident experts need not cross PCIe on each access. Instrument or profile the actual request, not infer transfers from routing counts or device dmon.
