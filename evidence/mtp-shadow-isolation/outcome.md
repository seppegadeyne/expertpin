# Outcome — MTP shadow isolation, 2026-09-06

Code: 7742c11f29f2661bf2e0c626ab44e09b1665747f (ownership/stats lifetime), 7602fdc48d782c84d2f10bfc2ba89bff137f8fea (templated request and32token validation). Both pushed/read back on origin/main. Reviews: review.md and request-review.md. Targeted CTest13/13; main31/33 (existing BERT-tokenizer/chat-template failures); full CPU build and CUDA-server build exit0. Repeated final CUDA build is incremental, so its final log only shows build-info generation. Initial compile/link was independently read by reviewer. Server's embedded build-info says0dbbd35e because binary was built before commit; not a claim that unmodified0dbbd35e passed. Post-run server SHA256:9bf5a2f46a8ceee466aae30bd0a06f3cfe5635e7ca275dec7b852b3882762b13; both runs used this unchanged binary with reviewed ownership fix.

| Run | Result | cgroup memory.peak, highest read GiB | device VRAM highest sample GiB | cgroup cap GiB |
|---|---|---:|---:|---:|
| run-20260906T124502 | server+MTP ready; raw prompt EOS after1token; no usable speed |32.40407180786133|22.638671875|36|
| run-20260906T125002 | templated68token input,32generated reasoning tokens |32.404659271240234|22.65234375|36|

RAM budget40GiB, VRAM28GiB. Last sampled oom/oom_kill/max counters zero. VRAM is sampled, NOT a continuously captured maximum. Final systemd journals confirm rounded32.4G RAM peaks; swap peaks1G/1.7G. Highest sampled RAM+swap.current33.19472885131836/34.09754180908203GiB. TierA poststop43.180206298828125/44.2716064453125GiB; GPU2%, launcher DRY and headroom pass, no override. DRAFT1/n_max4,NCMOE36,ctx8192,8GiB logical shadow,MemoryMax36G,GGML_CUDA_NO_PINNED1. 8K is allocated context, NOT an8K-token prompt.

Real short decode12.228905234009655tok/s:32tokens/2616.751ms, independently recalculated from response. Goal20 not reached (gap7.771094765990345tok/s). MTP21/22 accepted=95.4545%; tiny sample, not general acceptance. Generated reasoning is nonempty but truncated: not a completed answer or quality benchmark. No long-context/speedup claim. First response's1000000tok/s artifact explicitly rejected.

Trace1:66471 rows,dropped0,error0. Trace2:100000 rows,dropped3227,error0; total103227 equals aggregate requests and hits+misses. Sequence,relative_offset=expert*stride,footer/counters validated; first trace exact hits validated, second prefix-hitcount bounded by total. Second aggregate8GiB shadow:21037hits,82190misses,69680 simulated evictions. This is initialization+request mixed, NOT decode-only or physical I/O. No true phase/context tags, GGUF absolute offsets or complete request prefix coverage. Do not feed it as full trace to advisor. Correct next slice: request-scoped/phase+context tagging before initial warmup fills100K bound, then runtime→GGUF join. Physical matched bandwidth/GPU-GEMM and full model-pressure both-arm gate remain OPEN.

Qli stop/start requests12:45:04.841667→12:46:03.068697 and12:50:04.802591→12:51:13.258834CEST. Both exact scopes stopped verified, cleanup_errors empty, qliactive and Chromiumactive, journal[qli-tune]OK. Parent rechecked bothactive and no llama-server afterward. All research/reviews/CPUbuild-tests/codepush done before each respective GPU attempt. No guard-forcing/OOM. X leads in research.md; no external repo activity.

Raw files under evidence/expert-offset-trace/run-20260906T124502/ and run-20260906T125002/; computed results outcomes.json. State and final mail updated separately.
