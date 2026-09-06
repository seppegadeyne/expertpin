// CPU-only parameter/real-singleton regression; no model or GPU initialization.
#include "speculative.h"
#include "ggml-moe-prefetch.h"
#include "ggml-moe-stats.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

static int failures = 0;
static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}

int main() {
#ifdef __linux__
    // Only this test process is changed; never occupy a caller's capture path.
    if (unsetenv("GGML_MOE_TRACE_FILE") != 0) return 1;
#endif
    gpt_params base;
    base.expert_cache_sim_mib = 8192;
    base.expert_stats_file = "target-stats.json";
    base.n_ctx = 8192;
    base.n_batch = 128;
    const auto target = common_context_params_to_llama(base);
    require(target.expert_cache_sim_bytes == 8192ULL * 1024 * 1024, "target keeps budget");
    require(target.expert_stats_file && std::strcmp(target.expert_stats_file, "target-stats.json") == 0,
            "target keeps stats output");
#ifdef __linux__
    require(ggml_moe_cache_sim_try_acquire(target.expert_cache_sim_bytes), "target acquires real singleton");
#endif
    // Both external drafts (including MTP) and embedded MTP use this conversion.
    for (bool embedded : {false, true}) {
        auto draft_base = base;
        if (embedded) draft_base.pooling_type = LLAMA_POOLING_TYPE_NONE;
        const auto draft = common_speculative_context_params_to_llama(draft_base);
#ifdef __linux__
        // Same conditional claim as llama_new_context_with_model, without weights.
        const bool would_initialize = draft.expert_cache_sim_bytes == 0 ||
            ggml_moe_cache_sim_try_acquire(draft.expert_cache_sim_bytes);
        require(would_initialize, "draft must not fail initialization by double-claiming target shadow");
        require(!ggml_moe_cache_sim_try_acquire(target.expert_cache_sim_bytes), "single-owner rejection retained");
        ggml_moe_prefetch_stats stats{};
        ggml_moe_prefetch_get_stats(&stats);
        require(stats.cache_sim_capacity_bytes == target.expert_cache_sim_bytes, "target capacity unchanged");
#endif
        require(draft.expert_cache_sim_bytes == 0, "draft shadow disabled");
        require(draft.expert_stats_file == nullptr, "draft cannot overwrite target stats or retain temporary string");
        require(draft.n_ctx == target.n_ctx && draft.n_batch == target.n_batch, "unrelated context settings preserved");
        require(draft_base.expert_cache_sim_mib == base.expert_cache_sim_mib &&
                draft_base.expert_stats_file == base.expert_stats_file, "input unchanged");
    }
#ifdef __linux__
    ggml_moe_cache_sim_release();
    require(ggml_moe_cache_sim_try_acquire(target.expert_cache_sim_bytes), "target released normally");
    ggml_moe_cache_sim_release();
#endif
    base.expert_cache_sim_mib = 0;
    base.expert_stats_file.clear();
    const auto disabled = common_speculative_context_params_to_llama(base);
    require(disabled.expert_cache_sim_bytes == 0 && disabled.expert_stats_file == nullptr, "disabled stays disabled");
    auto override_base = base; // target disabled, draft explicitly requests telemetry
    auto cli = parse_command_line("llama-server --expert-cache-sim-mib 64 --expert-stats-file draft.json");
    require(gpt_params_parse(cli.first, cli.second, override_base), "draft CLI parses");
    free_command_line(cli.first, cli.second);
    require(override_base.expert_cache_sim_mib == 64 && override_base.expert_stats_file == "draft.json", "CLI actually sets overrides");
    const auto overridden = common_speculative_context_params_to_llama(override_base);
    require(overridden.expert_cache_sim_bytes == 0 && overridden.expert_stats_file == nullptr, "explicit draft telemetry still suppressed");
    override_base.expert_cache_sim_mib = 0;
    const auto stats_only = common_speculative_context_params_to_llama(override_base);
    require(stats_only.expert_stats_file == nullptr, "stats-only output suppressed");
    std::printf("speculative shadow regression: %d failures\n", failures);
    return failures ? 1 : 0;
}
