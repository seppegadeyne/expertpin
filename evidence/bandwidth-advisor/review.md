# Independent review and self-verification

Review `deleg_fd28a931`, configured gpt-6-astra, full staged diff (+499/-27,
12 files). Reviewer verdict: NO-GO for one P2 contract mismatch: zero traffic
and GEMM terms permitted by docs but derived zero cost rejected by positive=True.
No other concrete blockers reported. These are reviewer self-reports, not an
independent certification of the physics or measurements.

Parent verified docs/expert-bandwidth-advisor.md:26-30 and
scripts/advise-expert-cache.py:102-103 directly. Added zero-cost candidate next
to positive-cost candidate: test FAILED with ValueError invalid derived serial
cost before fix. Removed positive=True only for derived cost; zero is valid
analytical cost, never an infinite-throughput claim. Docs clarified. No second
review round requested; final tests are parent's runtime verification.

Parent also verified exact contiguous planner at scripts/expert_bw_calib.py:41-47,
DONTNEED/preadv at :59-75, and runtime stats diff src/llama-expert-stats.cpp:180:
seconds are now independent of resident bytes, no invented denominator. Unit test
covers 2.5s at zero and nonzero residency. Earlier RED captured missing JSON key;
coverage RED captured 4096 != 5000 and 0 != 123. New advisor's initial RED was
missing-module scaffold, not behavioral evidence. Own CTest found Python3.14
accepted deep JSON that Python3.11 rejected: explicit iterative depth64 check
added; both interpreters subsequently tested.

No GPU runtime validation of this CPU-only slice. Caller-supplied traffic/reserves
must still be established experimentally. Full issue #2 and launcher env output
remain open. No false 'phase2 completed' claim.
