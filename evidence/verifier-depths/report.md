# Dagverslag expertpin — 09-09-2026

**Originele verifier-matrix n_max=4/8/16 compleet. Nieuwe diagnostische decode-metingen boven 20 tok/s; correctness en volledige RAM-footprint nog niet bewezen.**

## Slice en bewijs

Analyzer uitgebreid met expliciete `--mtp-depth 8/16` (default4 behouden), codecommit `519aedab824d908190782951e7141a7c0784bf9f` gepusht en exact origin/main herlezen vóór GPU. Geen engine, launcher, harness, placement of default gewijzigd. Binary-SHA256 vóór/na gelijk: `3235529f95aeba4b4a71d4e33608fc0a1f4e6a78d5378616e7b286ca1dc8f279`.

Twee nieuwe guarded starts: n_max8→16, ieder 2×256 tokens +68 prompt, temp0/seed42/cache_promptfalse. Target0 en n_max4 zijn historische originele traces van 08-09, geen nieuwe gecontroleerde A/B.

| n_max | Eerste afwijkende positie | Target-ID | Draftvoorstel | MTP-beslissing |
|---|---:|---:|---:|---|
| 4 (historisch) | 34 | 8404 | 314 | accepted →314 |
| 8 (nieuw) | 43 | 11 | 539 | **rejected →303** |
| 16 (nieuw) | 54 | 11312 | 59056 | accepted →59056 |

Beide herhalingen bevestigen dezelfde posities/IDs/beslissingen. Nieuw: n_max8 wijkt af **na afwijzing** van het voorstel; dus niet simpelweg een onterecht geaccepteerde drafttoken op die positie. De raw target-argmax verschilt al vóór sampling. Batch target/MTP=1/2 bij beide nieuwe afwijkingen; raw top2-marges target/MTP: n8=0.096357/0.160439, n16=0.119034/0.661217. Dit bewijst geen numerieke oorzaak: recurrent/KV/checkpointstate blijft mogelijk. Ook n16 is niet uitsluitend een bijna-gelijke top2-marge.

Alle nieuwe first128-traces compleet; herhaalde IDs/proposals/rawtop2 exact, selected==rawargmax overal. Seriële tasklaunches zelf gekoppeld, payloads/responses en batchrijgrenzen gecontroleerd. Alle16 exacte content/reasoningpresence+tekstvergelijkingen met vroegere schone runs groen. Hun oude binary verschilt: regressiecheck, geen geïsoleerd trace-nietinterferentiebewijs. Verifiertraces zijn vóór commit/stopafhandeling, geen delivered-tokenbewijs.

## Metingen en budgetten

| Nieuwe run | Decode tok/s (2 herhalingen) | Cgroup RAM-piek /40 GiB | VRAM-samplepiek /28 GiB |
|---|---|---|---|
| n_max8 | 23.951473 /41.103864 | 32.405876 GiB | 22.122070 GiB |
| n_max16 | 28.104172 /40.999668 | 32.410915 GiB | 22.122070 GiB |

Beste gemeten41.103864 >20, maar uitsluitend diagnostisch op één prompt; geen snelheidswinst door deze analyzerwijziging, geen stabiele diepteranking of afgerond hoofddoel. Host had bij start ~43 GiB buff/cache. Cgroup memory.peak attesteert de aan de scope toegerekende RAM, niet afzonderlijk alle vooraf aanwezige gedeelde modelpagecache buiten die scope. Totale model-gerelateerde fysieke footprint inclusief zulke cache is niet gemeten; warme-cache-invloed niet geïsoleerd. VRAM48/44 samples, geen continuepiekgarantie. Swap-samplepieken1.561775/0.741447 GiB apart.

MemoryMax36G, finale scopejournals32.4G, alle bemonsterde max/oom/kill-events0. TierA46.720741/49.468655 GiB, GPU0%, DRY groen, GGML_CUDA_NO_PINNED=1; geen guard/budgetverlaging. Eerste n16-poging weigerde vóór hostprep/modelload wegens poort8102 TIME-WAIT; geen listener/server, na vrije poort één normale retry geslaagd.

## Tests, review en herstel

- Tests: analyzer8/8, harness14/14, target-only6/6, targetedCTest20/20 inclusief manifest. Main39/41: bestaande BERT-tokenizer/chat-templatefouten blijven; niet alles groen.
- Onafhankelijke gpt-6-astra-review `deleg_df05f517`: **AKKOORD**. Bestaande provenance/budgetbeperkingen benoemd. Parent kernclaims zelf herlezen: analyzer92–127, harness122–133/242–261/364–407, launcher77–82/131–133, server-context4348–4353 vóór4390–4399. Zie review.md.
- Qli onderbroken03:10:43–03:11:49 (66.804647s) en03:12:41–03:13:43 (61.360997s). Beide scopes leeg/weg bevestigd, cleanup_errors[], qli en Chromium **active**, `[qli-tune] OK`; finale extra qli-start/readback, geen llama-server.

X-research vóór tests/review/stop: [MTP-vocab-cut-lead](https://x.com/JASONMCNAB/status/2097082485926740450), [LayerStoRm backing-store-lead](https://x.com/TeksEdge/status/2097120231214764371). Alleen zoektoolclaims/read-only ideeën; geen externe cijfers als lokaal bewijs overgenomen, geen browser gestart.

**Volgende stap:** identiek-prefixreplay met gecontroleerde batch/checkpointstate rond34/43/54; vóór snelheidsacceptatie ook gedeelde pagecache-attributie en schone herhaalmeting isoleren. Geen default/placement/requantwijziging vóór correctnessdiagnose.

Evidence: `evidence/verifier-depths/{outcome.json,matrix.json,comparison-8.json,comparison-16.json,artifact-manifest.json,review.md,research.md}` en raw `evidence/clean-mtp-ab/run-20260909T031040-985ef1c0463044cdb5e4637559f186ca`, `run-20260909T031239-422f72d154c8411ea86bb8a64177dd53`; ook geweigerde poging bewaard.
