#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Raw target logits BEFORE sampler bias/penalty/grammar transforms. No model load.
struct server_verifier_logits {
    bool valid = false;
    int32_t argmax = -1;
    int32_t runner_up = -1;
    float top = 0;
    float second = 0;
    double margin = 0;
    bool has_proposal = false;
    float proposal_logit = 0;
};

inline server_verifier_logits server_verifier_summarize(const float * logits, int32_t count, int32_t proposal) {
    server_verifier_logits out;
    if (!logits || count < 2 || proposal < -1 || proposal >= count) {
        return out;
    }
    // Strict '>' preserves lowest token ID on ties, matching the raw greedy scan.
    for (int32_t id = 0; id < count; ++id) {
        if (!std::isfinite(logits[id])) {
            return server_verifier_logits{};
        }
        if (out.argmax < 0 || logits[id] > out.top) {
            out.runner_up = out.argmax;
            out.second = out.top;
            out.argmax = id;
            out.top = logits[id];
        } else if (out.runner_up < 0 || logits[id] > out.second) {
            out.runner_up = id;
            out.second = logits[id];
        }
    }
    out.margin = double(out.top) - double(out.second);
    out.has_proposal = proposal >= 0;
    out.proposal_logit = out.has_proposal ? logits[proposal] : 0;
    out.valid = true;
    return out;
}

inline const char * server_verifier_decision(size_t row, size_t draft_size, int32_t proposal, int32_t selected) {
    return row >= draft_size ? "bonus" : proposal == selected ? "accepted" : "rejected";
}

// Trace prefix per request. Decisions may precede stop handling or failed commit;
// this is not a delivered-token trace. A separate process-wide cap bounds logs.
inline bool server_verifier_in_window(int64_t output_position) {
    return output_position >= 1 && output_position <= 128;
}

struct server_verifier_request_limit {
    int64_t task = -1;
    size_t written = 0;
    bool admit(int64_t current_task) {
        if (current_task != task) { task = current_task; written = 0; }
        if (written >= 128) { return false; }
        ++written;
        return true;
    }
};
