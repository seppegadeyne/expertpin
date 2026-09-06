# X-research — 2026-09-06, vóór tests/review/GPU

Read-only x_search afgerond. Geen browser of CDP gestart door deze run.
- https://x.com/defilan/status/2094556589265211507 — zoektool rapporteert temp0-uitvoerdivergentie bij verschillende batchgroottes zonder MTP. Alleen hypothese-inspiratie: geen onafhankelijk geverifieerde externe proof/metrics, geen bewijs voor onze oorzaak.
- https://x.com/sheik_gulfaan/status/2094755616019669277 — zoektool rapporteert floating-point/reductievolgorde als batchgevoeligheid. Lokaal toetsen; geen causale conclusie.
- https://x.com/ItsmeAjayKV/status/2094895868621345162 — MTP-dieptesweep als context, niet als correctnessbewijs.

Slice: target-only (DRAFT=0) twee maal exact dezelfde bestaande 256-token chatrequest, seed42/temp0/cache_promptfalse. Engine en launcher onveranderd. Vergelijk exacte content/reasoning-hashes met bestaande MTP4/8/16. Her-tokeniseer geretourneerde tekst via dezelfde geladen target-vocab. Dit is NIET een opname van oorspronkelijke decode-token-IDs; de bestaande API-responses bewaren die niet. Bestaande acceptance-per-depth is geaggregeerd, dus bewijst geen acceptance-status op één outputpositie. Geen enginefix in deze diagnoserun.
