# GPU readiness + fysieke synthetische GPU-meting — 6 september 2026

Code: `4014acf59a23470050542b64256352c5f3a91f49` (origin/main geverifieerd).

## Resultaat

- Readiness echt benut: direct na qli-stop **100%** uitlopende utilsample, na
  2,0686s **0%**. Niet afgebroken op één sample; <5%-guard ongewijzigd.
  MemAvailable 43,62825 GiB, Tier A >=34 GiB. Ook pre-GEMM nvidia/free guard groen.
- Bestaande CUDA `bench-expert-gpu 101` daadwerkelijk geslaagd op RTX 5090.
  Drie fixed-shape synthetische hot single-expert MUL_MATs, één token,
  CUDA graphs uit, GGML_CUDA_NO_PINNED=1, 101 gemeten dispatch+sync-samples per type.

| Type | Quant | Mediane dispatch+sync | Max relatieve L2-fout |
|---|---|---:|---:|
| 20 | IQ4_NL | 7,651 µs | 0,0035470 |
| 21 | IQ3_S | 7,420 µs | 0,0038636 |
| 22 | IQ2_S | 7,790 µs | 0,0037871 |

Parent heeft sampleaantallen, medianen/min/max, types, correctness <=0,03 en
budgetten uit raw JSON zelfstandig nagerekend. Geen kernel-only timingclaim,
geen volledig model, geen MUL_MAT_ID/GPU-request-routing of fysieke VRAM-rate.

## Budget/protocol

- Hoogste gelezen cgroup memory.peak **0,3890075684 GiB / 40 GiB**.
  MemoryMax=40G, MemorySwapMax=0 binnen scope teruggelezen; alle swap.current=0.
  Finale scopejournal: **398,3M memory peak**, 4,018s wall clock.
- Device-wide VRAM-samplepiek **2,482421875 GiB / 28 GiB**, 67 samples;
  géén continu bewezen hardwaremaximum. Inclusief niet-benchmarkdevicegebruik.
- Alle gelezen memory.events max/oom/oom_kill nul.
- Qli-stoprequest 14:01:33.729458, startrequest 14:01:42.472532 CEST:
  8,743074s. Exactscope inactive, cleanup_errors=[], qli active + [qli-tune] OK;
  Chromium hersteld active, beide parent herbevestigd; ps geen benchproces.
- Binary SHA256: `93166c47cb03d87fb569d678a2c7e609285ca41e51d059c3b1f6d4b1e184cea6`.

## Doel/advisor

Geen nieuwe model-tok/s. Vorige korte 32-tokenprobe **16,2689 tok/s <20**;
geen nieuwe performancewinst of dieptebewijs. Budgetresultaten hierboven horen
alleen bij de microbench, niet bij een modelrun.

Advisor opnieuw uitgevoerd met provenance van de nieuwe GPU-observaties:
exit2/blocked, geen aanbeveling. Hot matvec medians mogen niet als complete
routed per-token compute worden ingevuld; matched pageable PCIe/staging,
complete CPU/GPU-routing en candidate-accounting ontbreken. Ook eerdere
buffered koud/warm preadv-rates zijn geen geïsoleerde hardware-/stagingrates.

## Kwaliteit en volgende stap

Review deleg_62d0fb75 initieel BLOCK: output-afwijzing kon oude evidence
overschrijven; bestaande cleanup kon stop/kill overslaan bij observatiefouten.
Beide parent met codebewijs bevestigd en hersteld, nieuwe foutinjecties groen;
geen tweede onafhankelijke PASS geclaimd. Details review.md, run-physical.py
120–163. Readiness13/13, targetedCTest15/15, CPU+CUDAbenchbuild groen.
Main33/35 met dezelfde bestaande BERTtokenizer/chattemplatefailures.

Research vóór tests/review/GPU: research.md. X-leads SamSchieds expert-cache en
JonathanLeaders AD/MTP, alleen ideeën; geen externe metrics overgenomen,
geen browserprocessen gestart, pinned-memorysuggestie verworpen.

Volgende kleinste slice: echte GPU-request-routing van dezelfde32tokenworkload,
matched pageable PCIe/staging; daarna candidate-accounting/advisorvergelijking.
Model-shadow-pressure-gate en integrale fase2 blijven OPEN.
