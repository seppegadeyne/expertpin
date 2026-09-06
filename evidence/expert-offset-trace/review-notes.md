# Review disposition — 2026-09-06

Reviewer deleg_104d9078 (gpt-6-astra), initial NO-GO. No second review pass.
Parent independently verified the key findings:
- Original run-guarded.py finally called timeout-capable command() before qli-start;
  check=False did not catch timeouts or log write errors. Corrected to isolated
  attempts, ignored repeated termination signals during cleanup, bounded child
  termination/reaping, separate qli/host-restore attempts and cleanup error status.
  CPU tests execute the actual finally AST against fake services with OSError,
  TimeoutExpired and KeyboardInterrupt at each service step. No real service
  manipulation in those tests; they are not hardware fault-injection evidence.
- Replaced wildcard scope inference by a reserved UUID unit passed through new
  launcher EXPERTPIN_SCOPE_UNIT. Scope readiness/cgroup MemoryMax required and all
  subprocess monitoring calls have timeouts. 900s sampled deadline, 960s signal
  watchdog (not a strict 900s claim); cleanup calls bounded, below 30min run window
  under normal OS timeout semantics. No promise against SIGKILL/host power loss.
- Unconditional qli-start on a preflight conflict is RETAINED because developer
  requires it even on failure; do not run this harness concurrently with another
  model workload. Pre-existing server/scope aborts before this run's model load.
- Draft file required before launch. HTTP success is now explicitly PENDING
  evidence validation, not trace success; parent must validate files after shutdown.
- ggml-moe-trace.h now flushes records before footer construction; footer flush
  and file close failures warn. Consumers must check stderr AND footer; no footer
  can retroactively encode a failure writing itself. File durability not claimed.
- Cap tests cover 100001 attempts at default cap and excessive requested cap,
  exact footer/line count; deterministic EBADF buffered-flush test; exact hook
  seq/entry/epoch/ids_rows metadata checks added. The first cleanup-test execution
  failed from using Mock instead of MagicMock for the path operator; fixed test
  fixture, not evidence of a production cleanup regression.
- Parent read ggml/src/ggml-backend.cpp:2455-2456: epoch advances only inside
  moe_prefetch branch. Docs now explicitly say it may remain constant.
- Research scope corrected: tensor-relative offsets and ids_rows, NO authenticated
  phase or GGUF absolute offset. CPU-only observer, not physical I/O.
- Previous committed evidence/expert-gemm/ctest-main.log:570-572 has the same BERT
  tokenizer/chat-template failures. Not a fresh checkout baseline rerun this run.

Test-first writer compile failed due missing header (API RED, not behavior RED).
An initial command used nonexistent build-cpu; corrected to existing build-cpu-shadow.
Final tests and GPU outcome are recorded separately; no invented runtime numbers.
