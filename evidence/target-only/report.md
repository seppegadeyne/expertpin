# expertpin — target-only diagnose, 06-09-2026

**Target-only is reproduceerbaar, maar wijkt af van alle bestaande MTP-uitvoeren. Oorzaak nog open; geen enginefix of diepte-defaultwijziging.**

- Eén vaste prompt, 2×256 tokens, seed42/temp0/cache_prompt=false. DRAFT=0 daadwerkelijk bevestigd in launcherplan, scopejournal (geen -md/--spec-type) en ontbrekende draftcounters. Ongewijzigde enginebinary, SHA256 vóór/na gelijk aan eerdere MTP-runs.
- Beide target-only-uitvoeren exact gelijk: content+reasoning-hash `6b0c803deb1fb4d791efd430f4b1cedaf2c4ec5ca92e7240f554b3ffa5455cb7`. Alle 12 eerdere responses teruggelezen: geen enkele gelijk aan target-only. Exacte hashes/diffs bewaard.

| Vergelijking | Eerste verschil in her-tokeniseerde reasoningtekst, 1-based | Target-ID / MTP-output-ID |
|---|---:|---:|
| target vs n_max=4 | 34 | 8404 / 314 |
| target vs n_max=8 | 43 | 11 / 303 |
| target vs n_max=16 | 54 | 11312 / 59056 |

**Beperking:** dit zijn exact roundtrippende her-tokenisaties van API-tekst, geen oorspronkelijke decode-ID-trace. Een MTP-output-ID is niet automatisch het draftvoorstel. Het oorspronkelijke drafttoken, de targetlogits en acceptance-status op die posities zijn onbekend: bestaande per-diepte-counters zijn geaggregeerd. Stap token-diagnose dus slechts gedeeltelijk afgerond.

**Codeanalyse:** temp0 gebruikt strikte argmax, zonder epsilon; verifier stopt bij eerste draft/target-ID-mismatch en emit targetcorrectie. Parent zelf gecontroleerd: `common/sampling.cpp:546–548,790–821`, `src/llama-sampling.cpp:683–696`, `server-context.cpp:4274–4276,4307–4315`. MTP batcht meerdere verificatietokens, target-only één (`server-context.cpp:3570–3612`). Batchnumeriek is plausibel; KV/recurrent-state of een andere bug niet uitgesloten. De meting lokaliseert configuratie-afhankelijkheid, niet de causale component.

**Gemeten GPU-run:**
- Decode **5,621 / 7,874 tok/s**, beide onder **20**; beste tekort **12,126 tok/s**. Geen snelheidswinstclaim.
- RAM: hoogste gelezen cgroup memory.peak **32,4043 GiB / 40**; MemoryMax=36G, finaal scopejournal 32,4G. Swap apart **1,2601 GiB** sampled.
- VRAM: hoogste device-sample **20,0049 GiB / 28**, 149 samples; geen continue-piekclaim. Geen max/oom/oom_kill-events.
- Tier A **45,3936 GiB**, GPU0%, DRY groen, no-pinned=1. Qli stop/start **16:37:14–16:39:51**; scope leeg/weg geverifieerd, cleanup_errors=[]; qli en Chromium **active**, `[qli-tune] OK`, geen llama-server achtergebleven.

**Codecommit:** `8b74a2f6` gepusht naar origin/main en exact remote-SHA teruggelezen. Nieuwe tests **6/6**, harness **14/14**, vergelijking **9/9**, targeted CTest **16/16**. Main **37/39**: bestaande BERT-tokenizer- en chat-template-failures blijven; niet alles groen.

**Review:** onafhankelijke gpt-6-astra vroeg wijzigingen wegens losse UTF-8-tokenstukjes en ontbrekende capturetests. Parent bevestigde de premissen in `server-common.cpp:241–264` en `server.cpp:1498–1508`; losse-tokenverrijking verwijderd, directe fout-/roundtriptests toegevoegd. Geen tweede onafhankelijke APPROVE. Alles vóór GPU afgerond.

**X-research:** [batchgevoeligheid](https://x.com/defilan/status/2094556589265211507), [numerieke variatie](https://x.com/sheik_gulfaan/status/2094755616019669277). Alleen door zoektool gevonden leads, geen lokaal oorzaakbewijs of overgenomen externe metrics.

**Volgende slice:** begrensde echte verifiertrace rond deze vroege afwijkingen: outputpositie, voorstel-ID, targetargmax, top-logitmarges, batchlengte en accept/reject; daarna identiek-prefix replay om batchnumeriek van recurrent/KV-state te scheiden. Correctness blijft vóór placement/snelheid/requant.

Evidence: `evidence/target-only/{outcome.json,output-differences.md,all-prior-readback.json,review.md}` en `evidence/clean-mtp-ab/run-20260906T163711-45502ba6d21c48ab95be359f73fb7eac/`.
