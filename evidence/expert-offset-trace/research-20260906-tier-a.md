# X research — 2026-09-06, Tier A trace capture

Completed before tests, host cleanup, miner stop or model load. x_search read-only;
no browser/CDP session opened, so no browser cleanup required by this search.

- https://x.com/lefu777/status/2096059297046437921 — search summary describes
  pure LRU, routed experts on CPU without repack, MTP changes. Useful lead for
  future policy comparisons; hardware and budgets not established here.
- https://x.com/JonathanLeaders/status/2095715911072207016 — summary describes
  AD-3.84bpw/MTP on 5090, NOT our AD-4.27bpw. No adopted throughput claim.
- https://x.com/marcozerbato/status/2096492426219106759 — EXL3/model-variant lead,
  no direct evidence for our Flash-Next quant or 40/28 GiB budgets.

Only search-provider summaries, not independently authenticated post contents.
Mixed PLE/NVFP4/27B reports rejected as evidence for our model. No new verified
performance result. Chosen slice: align the existing guarded capture harness with
now-active >=34 GiB Tier A mandate and retain the launcher's budget+4 headroom;
then attempt the first real bounded CPU expert observation trace with DRAFT=1.
This does NOT add phase tags, absolute GGUF offsets or physical bandwidth data.
