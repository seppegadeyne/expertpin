# Draft shadow ownership isolation

Scope: fix external-draft and embedded-MTP context parameter conversion. Both suppress the process-wide shadow budget and target stats filename *after* draft CLI overrides. Target settings and llama's single-owner guard stay unchanged. Independent draft shadow budgets are unsupported. The zero budget disables normal draft shadow observations (src/llama.cpp:6431,6454); the separate prefetch hook may still update global residency telemetry with include_cache_sim=false (ggml/src/ggml-moe-prefetch.cpp:832-841). No general per-context measurement isolation is claimed.

TDD: new target initially required CMake reconfigure (missing target, not RED). With a behavior-preserving extraction of the original conversion, test-speculative-shadow built and returned six failures, including two actual singleton double-claim rejections (same conditional acquire as context initialization, no weights). Fix then clears both fields. No full model initialization claimed by unit test. GPU integration uses the existing guarded 8K context/32-output-token harness; 8K is capacity, NOT an 8K-token input.

Stats filename clearing also removes a pointer into the local temporary gpt_params string returned by the old draft conversion. Neither actual per-context/per-phase trace tags nor runtime-to-GGUF join is introduced in this minimal bugfix. Existing trace remains kernel-distinct CPU observations, including initialization; no physical I/O or decode-only interpretation. Those follow after successful request readiness.

Research: research.md. No upstream/external repo activity.
