# Request capture correction

First real run run-20260906T124502 proves MTP+shadow server initialization is fixed, but the raw untemplated completion prompt stops at EOS after one token and no generated text. Its reported 1000000 tok/s is NOT a usable decode speed. No MTP drafts generated. Kept as negative evidence, not promoted as successful32token workload.

Harness now sends messages to /v1/chat/completions (server applies model template), max_tokens32, without ignoring EOS. Validates exact integer completion_tokens32, one choice finish_reason length, nonempty content/reasoning_content. Reasoning-only counts as decoding, not a completed answer. No guards changed. Seven CPU harness tests and targeted13/13 pass. Harness correction is not test-first; original C++ ownership fix was RED/GREEN.

Second independent review deleg_9c9d0178 gpt-6-astra: PASS, no blockers. Parent independently read examples/server/server-task.cpp:363-398 (EOS stop vs length, chat schema) and server-common.cpp:862-878 (real template application). GPU validation still pending at this commit. Both reviews finished before next qli-stop; previous run restored miner/Chromium active. X research remains completed for this run, no browser processes started.
