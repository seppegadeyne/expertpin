# Dagverslag expertpin — 08-09-2026

## Slice: eerste originele verifierdivergentie fysiek gelokaliseerd

Geen engine-, launcher-, quant- of placementwijziging. Bestaande bounded trace
uitgevoerd: MTP n_max=4 en target-only n_max=0, elk tweemaal 256 tokens,
68 prompttokens, seed42/temp0/cache_prompt=false. Beide runs en cleanup voltooid.

**In beide herhalingen is positie 34 de eerste afwijkende originele geselecteerde
ID (posities 1–33 gelijk):**

| Context op positie34 | Target-only | MTP4 |
|---|---:|---:|
| geselecteerd target-ID / raw argmax | 8404 | 314 |
| draftvoorstel | niet aanwezig | 314 |
| verifierbeslissing | target_only | accepted |
| raw runner-up | 314 | 8404 |
| raw hoogste logit | 23.919544219970703 | 23.80539894104004 |
| raw runner-up-logit | 23.79714584350586 | 23.794544219970703 |
| raw marge | 0.12239837646484375 | 0.010854721069335938 |
| assembled batch tokens | 1 | 5 |

De ranking is dus al verschillend in de **raw targetlogits vóór sampling**.
Op dit punt accepteert MTP het voorstel omdat het gelijk is aan zijn eigen
gekozen target-ID; geen bewijs voor een fout in de ID-matchregel. Numerieke
batchgevoeligheid is een concrete hypothese, maar recurrent/KV/hidden-state
verschillen zijn NIET geïsoleerd. Geen identiek-prefixreplay gedaan, oorzaak open.
Assembled batch is niet kernelmicrobatch; tracebeslissingen zijn vóór commit/
stophandling, niet zelfstandig bewijs van aflevering.

Alle vier first-128-traces volledig gevalideerd; geselecteerd ID was overal raw
argmax. Binnen modus waren IDs, proposalbeslissingen en raw top-two exact gelijk
in de twee herhalingen. Taskvolgorde zelf gekoppeld aan seriële launchregels in
serverlogs (MTP29/168,target25/283). Alle 12 exacte vergelijkingen van nieuwe
content/reasoning_content (inclusief veldpresence) met historische bronresponses
van dezelfde modus gelijk. Historische binary verschilt: dit is een regressiecheck,
GEEN geïsoleerd bewijs dat tracing nooit interfereert of volledige verliesvrijheid.

## Echte metingen — diagnostisch, geen schone snelheids-A/B

| Run | Decode tok/s (2 herhalingen) | RAM-piek / 40 GiB | VRAM-samplepiek / 28 GiB | Swap apart |
|---|---|---|---|---|
| MTP4 + trace | 3.566077 / 5.306082 | 32.404732 GiB | 22.101563 GiB | 2.205364 GiB |
| Target-only + trace | 12.027517 / 17.053902 | 32.404755 GiB | 19.445313 GiB | 1.238167 GiB |

Beste diagnostische meting17.053902<20 tok/s; tekort2.946098 tok/s. Geen causale
ranking of optimalisatiewinstclaim: tracing, volgorde en cache/swap confounden.
Historische schone beste14.259921 tok/s blijft een afzonderlijke oude meting.
RAM = hoogste gelezen cgroup memory.peak; finale scopejournals beide32.4G.
VRAM = device-wide steekproef (189/83 samples), geen continue piekgarantie.
MemoryMax36G, GGML_CUDA_NO_PINNED=1, max/oom/oom_kill-events0; geen guardoverride.
TierA42.441185/44.103722GiB, beideGPU0%, DRYgroen; budgetten niet verlaagd.
BinarySHA256 vóór/na3235529f95aeba4b4a71d4e33608fc0a1f4e6a78d5378616e7b286ca1dc8f279.
Geen rebuild/modelhashherberekening.

## Tests, review en herstel

- TargetedCTest20/20 inclusief expert-manifest; harness14/14, analyzer7/7,
  target-only6/6 groen. Main39/41: bestaande BERT-tokenizer/chat-templatefouten.
  Eerste discovery op niet-bestaande build/ hersteld naar build-cpu-shadow.
- Reviewdeleg_4d105993: inhoudelijk voorwaardelijk GO, maar tool rapporteert
  timeout/error na volledige samenvatting; geen formele APPROVE geclaimd.
  Parent verifieerde guards, cleanup, vóór-sampler/commitplaatsing en afwijkende
  historische binary zelf; file:line-bewijs in review.md. Geen codecommit.
- Research/tests/review volledig vóór hostprep/qli-stop; geen browser gestart.
- Qli stop/start03:10:55–03:14:39 en03:15:23–03:17:10, onderbreking224.310680/
  107.377048s. Beide scopes leeg/weg geverifieerd, cleanup_errors=[];
  qli en Chromium active, [qli-tune]OK bevestigd, geen llama-server over.

## X-leads (read-only; geen lokaal bewijs)

- https://x.com/ViC305/status/2097005177668812843 — zoektool rapporteert ook
  MTP/no-draftdivergentie; paritycheck belangrijker dan 'coherent'.
- https://x.com/inovelloE/status/2096577582741238212 — batched-expertcachebugs als lead.
- https://x.com/Tech2Wild/status/2096754559263617169 — W4A16/NVMe later; nu geen requant.

## Volgende stap

Identiek-prefixreplay rond positie34 met gecontroleerde batchvorm en
checkpoint/recurrent-state; zo numeriek versus state isoleren. Daarna pas
enginefix/default/placementbesluit. nmax8/16 originele verifiercontext nog niet
vastgelegd. Geen extra GPU-runs deze slice.

Artifacts: evidence/verifier-runtime/{comparison.json,outcome.json,validated-*.json,
review.md,research.md,*log}; originele logs/requests/responses in
clean-mtp-ab/run-20260908T031053-0ae82941ea594b31bce4238180c84c2e en
clean-mtp-ab/run-20260908T031521-7774af1cc46549f28b0afa89da520986.
