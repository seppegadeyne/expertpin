# Review (self-verification — gpt-6-astra usage-pause until ~2026-09-15, Seppe mandate)

Slice: pause-conform miner handling in the guarded harness lineage + dev unit.

Claims checked against the code by the committer (file:line):

1. "cleanup never starts qli anymore" — run-guarded.py: the old unconditional
   `systemctl --user start qli.service` + `errors.append('qli not verified active')`
   block is REPLACED by restore_miners(), which iterates
   self.summary['miners_stopped'] only. qli can never enter that list because
   _miner_stop_uses() returns None for is-enabled in ('not-found','masked')
   and stop_miners() then only records qli.service_skip. Verified by
   test_qli_masked_is_never_stopped_or_started (asserts zero 'start' argv
   across stop+restore) and test_cleanup_masked_qli_no_longer_fails_the_run
   (pre-fix behavior: run-20260911T195653-e2e/summary.json ended
   cleanup_failed with qli_status:null on this masked host).

2. "Jetski is stopped before GPU work and restored+verified after" —
   stop_miners() stops jetski.service when its unit exists; restore_miners()
   starts it and requires is-active == 'active', else cleanup error.
   Verified by test_jetski_active_is_stopped_then_restored_and_verified and
   test_jetski_restore_failure_is_reported_not_swallowed.

3. "run-e2e.py uses the same path" — its qli-stop lines are replaced by
   run.stop_miners() (same harness class); cleanup flows through run.cleanup()
   which now calls restore_miners(). tests/test-hermes-e2e-20260911 still 6/6.

4. "dev unit never touches miners" — scripts/expertpin-dev.service no longer
   has ExecStartPre stop / ExecStopPost start of ANY miner. The previous
   draft would have started qli.service on every teardown (pause-mandate
   violation the moment qli is unmasked) and stopped it on start (contradicts
   'Jetski keeps running during the day; guard refusal is correct behavior').

5. No guard/budget weakening: Tier A check, dry guard, cgroup caps, sampling,
   budget validation untouched (diff is confined to miner stop/restore paths).

Test evidence: python3 -m unittest tests.test-coldcache-ab-20260911 → 9/9 OK
(5 pre-existing + 4 new); test-hermes-e2e-20260911 → 6/6; targeted ctest
(coldcache|hermes-e2e|quality-gate|context64k|multineedle) 5/5; full suite
49/52 with the 3 known historic failures (bert-bge tokenizer, chat-template,
eval-callback libcurl) — unchanged by this slice.
