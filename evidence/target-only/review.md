# Review en beperkte codeanalyse — target-only

Review deleg_a13e6583 (gpt-6-astra): WIJZIGINGEN GEVRAAGD.
Parent zelf herlezen: server-common.cpp:241–264 en server.cpp:1498–1508 bevestigen ruwe tokenbytes/strikte JSON; losse-tokenverrijking verwijderd in plaats van foutgevoelig te houden. Capture bewaart uitsluitend volledige exact roundtrippende velden en IDs. Nieuwe directe mocktests oefenen Unicode-bytefallback, twee eigen outputs + referentie, null, HTTP-fout, ongeldige IDs, roundtripmismatch en deadline. Prefix/length/null-diff hoort bij offline analyse, geen claims op basis van zip-only. Geen tweede onafhankelijke APPROVE gevraagd.

## Parent-verificatie engine (geen fix)
- common/sampling.cpp:546–548: temp0 gebruikt greedy; src/llama-sampling.cpp:683–696 std::max_element met strikte logit<, geen epsilon. Bij exacte tie eerste maximale kandidaat in huidige arrayvolgorde. Geen logits gemeten, dus tie/kleine marges niet vastgesteld.
- common/speculative.cpp:1943–1974: MTP-resultaat heeft geen DFlash proposal-distributies; server-context.cpp:4274–4276 kiest gewone target-sampleracceptance.
- common/sampling.cpp:790–821: sample target, emit target-ID; alleen doorgaan bij draft[i]==target-ID, anders stoppen. Volledig matchend prefix krijgt bonus-targettoken. Draft-quantverschil op zichzelf rechtvaardigt geen ander targetargmax; het beïnvloedt wel voorstellen/batchlengte.
- server-context.cpp:3570–3600 vs 3606–3612: MTP verifieert sampled + meerdere drafttokens in één batch, target-only één token. Numeriek batchpad is plausibele hypothese, NIET gemeten oorzaak.
- server-context.cpp:4307–4315 en 3587–3594: counters sommeren per draftdiepte over alle stappen. Geen koppeling outputpositie->proposal-ID/acceptance. Bestaande metrics kunnen die gevraagde detailstatus niet leveren.
- server-context.cpp:4317–4337: rollback/commit en cache-posities bestaan; correcte recurrent/KV-state nog niet bewezen door deze beperkte lezing. Zowel numerieke als statehypothese blijven open.

Target-only vergelijken kan de configuratie-afhankelijkheid vernauwen, niet op zichzelf bewijzen of de oorzaak in drafter/acceptance/numeriek/KV zit. Geen kwaliteitsoordeel of verliesvrijheidclaim.
