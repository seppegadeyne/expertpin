# X-research — 2026-09-09, vóór tests en GPU-werk

Read-only x_search afgerond (venster 2026-09-06 t/m 09). Geen browser/CDP gestart, geen externe posts.

- https://x.com/JASONMCNAB/status/2097082485926740450 — zoektool rapporteert een MTP draft-vocab-cut-replicatie op Spark. Idee: draftkosten apart verlagen; hardware/cijfers niet lokaal geverifieerd en geen reden correctnessgate over te slaan.
- https://x.com/TeksEdge/status/2097120231214764371 — LayerStoRm expert backing store / PCIe streaming. Idee: bounded hot working set; gerapporteerde ruime host-RAM-opstelling bewijst niets binnen onze 40 GiB. Lokale read-only referentie blijft onaangeroerd.
- https://x.com/TeksEdge/status/2096780493026902287 — FreeToken expert-cache/overlap-lead; geen lokale prestatieclaim overgenomen.

Zoektool vond geen directe nieuwe oorzaak voor greedy batchdivergentie. Resultaten zijn zoektoolclaims, niet onafhankelijk gelezen primaire benchmarks.

Slice: originele verifiertraces voor de nog open n_max=8/16 vastleggen en vergelijken met de bestaande target-only referentie; analyzer expliciet uitbreiden voor ondersteunde dieptes. Geen engine/default/placementwijziging. Identiek-prefix/checkpointreplay blijft de volgende causale isolatie; deze slice maakt de eerder gevraagde token/proposal/acceptance-matrix eerst volledig. Traced tok/s alleen diagnostisch.
