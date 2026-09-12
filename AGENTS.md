# AGENTS.md — expertpin

## Repo-identiteit (LEES EERST)

- Dit is **seppegadeyne/expertpin**: https://github.com/seppegadeyne/expertpin — een VOLLEDIG ZELFSTANDIG project (Seppe-mandaat 2026-09-04).
- Geen upstream-relatie meer: alle externe remotes (`upstream`, `fork`) zijn verwijderd; alleen `origin` rest. Er is géén ander publiek repo waar we aan bijdragen, PR's opvolgen of discussiëren. Alle gh-activiteit (issues, commits, pushes) uitsluitend op seppegadeyne/expertpin.
- **`gh` default repo is vastgezet met `gh repo set-default seppegadeyne/expertpin`.** Controleer bij twijfel met `gh repo set-default --view`.
- De codebase (oorsprong ik_llama.cpp-fork) is volledig eigendom van dit project: vrij breken, hernoemen, herschrijven. Compatibiliteit met andere forks/upstreams is GEEN criterium.
- Andere repo's (club-3090, Whamp/vLLM, FreeToken, buun, EXL3, ...) alléén READ-ONLY als inspiratie (klonen naar `third-party/`, buiten git, mag); eigen implementaties schrijven; nooit contribueren of posten.

## Werkafspraken

- Taal: ALLE GitHub-content van deze repo is Engels — issues, issue-comments, PR-beschrijvingen, reviews, releases en commit-messages. Nederlands alléén in lokale, ongepubliceerde notities (state-file, lokale logs, mails naar Seppe). Bestaande NL-berichten worden bij ontdekking naar het Engels vertaald (GitHub bewaart de edit-history, dus niets gaat verloren).
- Commits: Engels, conventional style (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
- Push naar `origin main`; nooit force-push op main (op eigen feature-branches mag geschiedenis herschreven worden).
- Tests: bestaande unit tests (o.a. `test-expert-manifest`) moeten altijd groen blijven; nieuw gedrag krijgt eigen tests (TDD waar haalbaar).
- GPU-werk op Aorus: qubic-miner is user-service `qli.service` — PAUZE sinds 2026-09-11 (Seppe): gemaskeerd, nooit (her)starten, ook niet na GPU-werk. Guards uit `scripts/run-qwen38-flash-next.sh` respecteren (DRY=1 eerst; MemAvailable-check; cgroup MemoryMax).
- Budgetten (Seppe, 2026-09-04): run-footprint max 40 GiB systeemRAM én max 28 GiB VRAM — elke GPU-run tegen beide meten en rapporteren.
- Modellen/assets staan buiten de repo in `/home/seppe/Models/qwen3.8-flash-next/` — nooit committen.

## Voorbeeld-launcher

Zie `scripts/run-qwen38-flash-next.sh` (RAM/VRAM-guards, cgroup-cap, expertpin-vlaggen `--expert-manifest`/`--resident-experts`).
