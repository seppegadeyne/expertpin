# Bounded verifier diagnostics

Opt in with `EXPERTPIN_VERIFIER_TRACE=1` on llama-server, or
`python3 evidence/clean-mtp-ab/run-guarded.py --verifier-trace --startup-n-max 4`.
The harness keeps all existing host/cgroup/GPU/cleanup guards. Use depth 0 for
target-only. It always scrubs inherited tracing; the flag is explicit.

The server prints `EXPERTPIN_VERIFIER {json}` for positions 1..128 per request,
up to 4096 records per process (explicit LIMIT marker). No task can emit more
than 128 records, even after rewinds; duplicated positions are rejected by the
analyzer and unsuitable for prefix comparison. It snapshots original
raw target logits before the sampler transforms them: strict raw argmax,
runner-up, top-two margin, and raw proposal logit. Nonfinite rows are invalid,
not silently repaired. No full logit vectors, token text, or draft logits saved.

The IDs are real sampler/verifier decisions, not retokenized text. Each record
includes internal task/slot, 1-based generated position, verifier row/count,
logits row and **assembled batch tokens** (not kernel microbatch size). MTP
records are sampled before checkpoint commit and stop handling: they must not
be presented as confirmed delivered output. `target_only` identifies a normal
single-token sampling path, including the initial token of an MTP request.
Only the ID-match verifier is instrumented; distribution-based verification
is deliberately not represented as equivalent greedy acceptance.

Run `scripts/analyze-verifier-trace.py TARGET_RUN MTP4_RUN --output FILE` offline.
For MTP8 or MTP16, pass `--mtp-depth 8` or `--mtp-depth 16` explicitly. The
default remains 4; mismatched startup depths, request labels, payloads and
response counters are rejected. This is still a pairwise prefix comparison,
not a controlled same-state replay or a cross-run throughput ranking.
It requires two complete first-128 traces per successful guarded run, rejects
cap/incomplete/invalid rows, and compares original selected IDs. Source request
and response JSON and server logs remain authoritative. Compare exact response
fields against prior untraced outputs separately before claiming noninterference.

Raw argmax is not necessarily the selected ID when bias/penalties/grammar apply.
Equal output prefixes do not prove equal hidden/recurrent/KV state. A batch
argmax difference with a small margin is a lead, not causal proof of numerical
error. Identical-prefix replay and checkpoint/state isolation remain separate.
Throughput from tracing is diagnostic, not an optimization A/B or new baseline.
