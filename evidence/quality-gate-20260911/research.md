# Research 2026-09-11 — IQ2_XXS tool-calling quality (before the gate run)

Search-tool leads only (x_search, read-only). No browser started, no external
posts, no metrics adopted as facts about our setup. These are signals, not proof;
the gate run is the actual evidence for ps-iq2xxs on Aorus.

## Query: IQ2_XXS tool calling degradation MoE quantization

Community claims (2026) as returned by the search tool:

- Very large MoE models tolerate extreme low-bit expert quantization unusually
  well due to redundancy; Benjamin Marie noted many experts are nearly useless
  or highly redundant (x.com/bnjmn_marie/status/2093586057669582867).
- Asymmetric schemes (experts ~IQ2_XXS/Q2_K, attention/router/norms kept higher)
  preserve structured output far better than uniform 2-bit
  (x.com/antirez/status/2048425610809131406). The PS checkpoint follows exactly
  this split (experts IQ2_XXS, PLE n-gram IQ4_NL, attention Q4_K, norms F32).
- Tool calling on quantized large MoEs is reported reliable in practice, with
  complex multi-hop trajectories still accumulating errors like any low-bit
  quant. Some users report IQ2_XS/IQ3_S tiers behaving better in specific cases
  than the absolute lowest tier (x.com/PaulGugAI/status/2097646607336620115).

## Why a gate on our hardware

The author of the PS checkpoint claims tool-calling JSON correctness, but those
are community claims on V100 hardware. Our decision (default-switch or not)
needs our own forced-choice tool-call round-trip inside the 40/28 budgets on
Aorus: exact function name, schema-exact arguments (both enum-bound fields),
a usable tool_call id, and a clean follow-up answer that reflects the tool
result. Target-only (DRAFT=0) so quantization of the checkpoint is the only
variable; the MTP drafter adds a second quant layer that would confound the
quality attribution.

## Scope note

This gate covers round-1 schema conformance + one tool round. It does NOT
cover: multi-tool selection, parallel calls, streaming tool deltas, long
trajectories, needle-recall (separate slice), or Hermes end-to-end.
