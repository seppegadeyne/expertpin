Tussentijds verslag 5 2026-09-11 13:05 — 64K-context AF (9/9) + Hermes-API-contract AF (5/5)

Beide laatste open roadmap-items vandaag afgerond onder je verlengde
GPU-duurmandaat (RAM/VRAM 40/28 onverkort, geen enkele OOM-event).

1. 64K-CONTEXT op ps-iq2xxs (ladder 16K→32K→64K, tokenizer-exacte tellingen):
   16K: 15.269 tok — 3/3 needle-recall (RAM 32,42/40, VRAM 16,14/28)
   32K: 30.920 tok — 3/3 (32,41/40, 16,57/28)
   64K: 61.401 tok — 3/3 (32,41/40, 17,14/28)
   Totaal 9/9 recall. Je 64K-productacceptatie-eis voor Hermes-context is
   gehaald, met ~11 GiB VRAM-marge op de 64K-stap. 264K blijft een apart
   doel (observatie: KV ~0,5 GiB VRAM per 16K tokens).

2. HERMES-API-CONTRACT (OpenAI-compat endpoint-tests, 5/5 PASS):
   - models-discovery (let op: model-id = GGUF-pad, moet exact zo in de
     Hermes-config)
   - non-streaming completion (finish stop)
   - SSE-streaming plain (coherente chunks, één terminale finish)
   - SSE-streaming forced toolcall (10 geïndexeerde delta's, finish
     tool_calls)
   - volledige tool-round: call-id-koppeling in de tool-message → clean
     stop-antwoord
   Integratienotities: tool_choice alleen als string ("required"/"auto"/
   "none"); streaming tool-deltas per index aggregeren.

Commits: 468455ca, ff77d267, 5f686ddd (64K) + e84ddcd5, 4f057fc1, 99118e11
(contract) — gepusht en exact herlezen. Issue-comments op #2 en #3 geplaatst
en herlezen.

PS IQ2_XXS-status binnen 40/28: ≥20 tok/s warm+sustained+koud-worst-case,
toolcalling PASS, needle 2K+64K PASS, 64K-capaciteit PASS, API-contract
PASS. Resterend vóór default-switch: error-path-contracttests, dev-serving-
profiel + echte Hermes-agent-run (issue #3), 264K, multi-needle.

Host: qli actief; geen llama-server; tracked tree schoon.
