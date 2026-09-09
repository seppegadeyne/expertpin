# Onafhankelijke review — 2026-09-09

Review deleg_df05f517, gpt-6-astra, 255.79 s, schema geldig, eindoordeel **AKKOORD** voor beperkte analyzeruitbreiding en guarded 8→16-runs. Volledige diff gelezen; geen blokkerende regressie. Geen GPU-werk door reviewer.

Beperkingen: fixtures testen parameterafhandeling, geen brede fysieke batches; analyzer alleen geen resource/provenance-attestatie; VRAM bemonsterd, swap apart; target0 historisch, geen gecontroleerde A/B. Daarom bronpayloads/responses, environment, DRY, samples, serial tasklaunches, batchrijgrenzen en cleanup afzonderlijk controleren. Geen numeriek-causaliteit of throughput-ranking claim.

Parent zelf gecontroleerd:
- scripts/analyze-verifier-trace.py:92–127: strict int4/8/16, zelfde diepte in startup/labels/payload/response; overige prefixvalidatie blijft.
- evidence/clean-mtp-ab/run-guarded.py:122–133,198–219,242–261,364–407,439–453: budgetmonsters, TierA+DRY, finally/restore met active-readbacks. Niet gewijzigd.
- scripts/run-qwen38-flash-next.sh:77–82,105–108,131–133: extra RAM+4 guard, no-pinned, echte cgroup MemoryMax.
- examples/server/server-context.cpp:4348–4353 trace vóór checkpointcommit4390–4399. Geen delivered-tokenbewijs.
- common/sampling.cpp:790–819: greedy ID-match prefix + correctietoken, geen epsilonacceptance.
- evidence/verifier-depths/main.log:584–594: 39/41, historische BERT/chatfailures; targeted20/20, analyzer8/8, harness14/14, target6/6.

TDD parent heeft vóór implementatie de gewijzigde tests echt uitgevoerd: 21 TypeErrors op ontbrekend mtp_depth-keyword. Tooloutput beschikbaar in sessie, geen apart opgeslagen RED-log; reviewer kon deze historische stap niet onafhankelijk staven. Geen claim dat reviewer die RED-run reproduceerde.

X-research/tests/review afgerond vóór hostprep/qli-stop. Engine/harness/binary onveranderd, SHA256 in binary-before.log. Lokale analysehelpers zijn ad-hoc evidence tooling, niet onderdeel van de codecommit.
