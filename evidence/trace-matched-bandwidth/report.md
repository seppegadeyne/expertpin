# expertpin — 06-09, trace-matched fysieke reads

Codecommit `7a47a91c` gepusht; origin/main exacte hash herlezen. Nieuwe probe leest werkelijk de gejoinde GGUF-expertslices; CUDA-matvecbench gebouwd maar deze run niet uitgevoerd.

## Meting
CPU-only, seed17, decode van request27 uit run131516. 749 unieke slices, 511,816 MiB per pass, 6,65% van de unieke decodebytes; geen volledige trace-replay. Alle geselecteerde offsets door parent tegen joined.csv geverifieerd.

| Quant | Koud aangevraagd (GiB/s) | Warme herlezing (GiB/s) |
|---|---:|---:|
| IQ2_S | 1,188 | 16,534 |
| IQ4_NL | 1,383 | 14,702 |
| IQ3_S | 1,417 | 17,412 |
| Totaal | **1,340** | **15,948** |

Koude snapshots: 0/131758 pagina-occurrences resident; warme snapshots: allemaal resident. `/proc/self/io` storage-delta koud 603906048 bytes (575,930 MiB), warm 0. Readahead/accounting-amplificatie 1,125× payload. Dit zijn effectieve buffered-preadv-rates inclusief Python/copykosten, GEEN geïsoleerde NVMe- of RAM-bandbreedte. Miner/Chromium actief tijdens deze CPUmeting; geen model geladen. Cgroup MemoryMax40G, MemorySwapMax0; journal RAMpiek602,6M, onder40GiB; geen GPUcompute of nieuwe VRAMrunpiek.

## GPU/doel
Hostprep + qli-stop uitgevoerd na alle reviews/builds/tests/codepush. Directe GPUguard zag29% util (vereist<5%), MemAvailable44GiB volgens free. Afgebroken vóór scope/probe/modelload, niet geforceerd of GPUherprobeerd. Util kan een uitlopende meetsample zijn, oorzaak niet bewezen. CUDAbench compile/link groen, runtime/GEMMtimings dus nog onbekend.
Geen nieuwe tok/s: vorige korte modelprobe **16,27 tok/s**, doel≥20 niet gehaald; vorige RAM32,40/40GiB en VRAMsample22,65/28GiB, geen nieuwe budgetmeting van een modelrun.
Advisor opnieuw echt uitgevoerd: **blocked**, geen gewijzigde aanbeveling. Fysieke offsetread-observaties nu aanwezig; pageablePCIe, stagingRAM, GPUrouting/timings en gemeten candidate traffic/reserves ontbreken nog. Hardware-rates niet ingevuld met proxycijfers.

## Kwaliteit/status
- Volledige CPUbuild + CUDAbenchbuild groen; targetedCTest14/14. Main32/34: bestaande BERTtokenizer/chattemplatefailures. Nieuwe traceprobe15tests + I/O-observation14tests groen in onafhankelijke review.
- Review vond cleanupfout; parent zelf controlflow bevestigd en SIGKILL-escalatie + exactscope/cgroupcontrole toegevoegd. Drie mocked foutpaden groen; geen tweede review-PASS geclaimd. Parent GPUallocatie/synchronisatie en backenddispatch zelf herlezen.
- Qli-stoprequest13:44:04.509871, start13:44:07.589147; qli **active**, `[qli-tune] OK`, Chromium **active**, cleanup_errors leeg; nadien opnieuw geverifieerd.
- X-research vooraf: https://x.com/DoingFedTime/status/2096079245437001737 (offload/transferlead) en https://x.com/JonathanLeaders/status/2095715911072207016 (ander quant+MTP). Alleen zoekbevindingen, geen externe metrics overgenomen.
- Evidence: `evidence/trace-matched-bandwidth/{bandwidth-cpu,outcome,execution,local-advice}.json` plus rawlogs.

Volgende stap: GPUidle-readiness begrensd afwachten zonder guards te verlagen; guarded CUDAbench uitvoeren, daarna GPU-requestrouting en matched PCIe/staging/candidate-accounting. Integrale fase2 en model-shadow-pressuregate blijven open.
