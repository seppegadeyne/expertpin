Slotverslag 3 2026-09-11 17:05 — laatste drie roadmap-items AF; dag compleet

1. MULTI-TURN AGENT E2E PASS: turn 1 maakt marker-bestand via shell-tool
   (62,5 s); turn 2 hervat de sessie (--resume latest) en cat's het eigen
   artefact, antwoordt exact (4,4 s). Sessie 20260911_164157_a98121:
   8 messages / 4 tool calls over beide beurten — continuïteit bewezen.
   (Valkuil: -c = continue-by-title, niet last.)

2. SSE MID-STREAM ERROR-CONTRACT: bronverifieerd (server.cpp:1215/:1264):
   fout vóór eerste chunk = gewone HTTP-error; fout ná chunk = één
   data:{"error":...}-frame, stream eindigt ZONDER [DONE]. Fysiek niet
   deterministisch te triggeren zonder fault-injectie → client-regels
   gepind met 3 synthetische tests (fail-closed); fysiek NOT TESTED.

3. DEV-SERVING-RECEPT: docs/dev-serving-recipe.md — bewezen integratie
   geconsolideerd (CTX>=65536, klassiek custom-provider-config, 3 valkuilen,
   gemeten timings). Residente systemd-unit = follow-up in issue #3.

Commits 0d306ff4..b601b793 gepuspt/herlezen; issue #3-comment 5636304136
herlezen; targeted ctest 7/7; qli+Chromium actief; geen llama-server; tree
schoon. Reviewverbod gpt-6-astra nageleefd (0 dispatches).

Dagtotaal 11-09: 10 slices, allen PASS binnen 40/28 — kwaliteitsgates
(toolcall+needle+multi-needle), sustained decode, koude-cache (advisory +
mincore-verified), contextladder 16K→64K→264K, OpenAI-contract 5/5,
error-path 3/3, Hermes-agent E2E single-turn én multi-turn, SSE-contract,
dev-serving-recept. Resterend voor issue #3: residente dev-unit en langere
agent-trajecten.
