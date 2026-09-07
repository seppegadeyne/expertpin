# X-research — 2026-09-07, vóór tests/review/GPU

x_search uitgevoerd met from_date=2026-09-05 voor Qwen3.8-flash-next,
Flash, AD-4.27bpw, MTP greedy/n_max/batch-invariance/rollback, expert-cache,
EXL3 en INT4/W4A16. Geen browserprocessen/tabbladen gestart.

Leads (alleen leeswerk):
- https://x.com/inovelloE/status/2096577582741238212 — zoeksamenvatting noemt
  een batched-token cachebug en MTP. Bruikbaar idee: echte verifierbeslissingen
  en batchgrootte vastleggen vóór een enginefix. Geen bewijs voor onze oorzaak.
- https://x.com/geldeki/status/2096738747547013337 — EXL3/CPU-offload lead;
  geen lokaal vergelijkbare 40-GiB/28-GiB meting.
- https://x.com/Tech2Wild/status/2096754559263617169 — W4A16/MTP lead;
  geen directe aanleiding om de open correctness-gate met requant te omzeilen.

De zoektool noemt cijfers en architectuurkenmerken 'verified', maar levert
geen onafhankelijke verificatie. Die claims NIET overgenomen: verwarring tussen
experts/lagen, quantformaten en modelvarianten mogelijk. Geen directe post gevonden
die onze greedy n_max-divergentie verklaart. Numeriek versus recurrent/KV blijft open.

Slice: bounded oorspronkelijke target-logit-samenvattingen en verifier-IDbeslissingen
bij echte serverrequests; diagnostiek, geen acceptanceregel/default/placementwijziging.
Identiek-prefix replay is een aparte vervolgstap, niet bewezen door deze trace.
