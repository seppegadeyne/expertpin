// Unit test for the ggml MoE prefetch-engine stats counters (issue #1).
//
// Contract under test (colibri ColiExpertStoreStats analogue, per distinct
// expert id at MoE kernel entry):
//   requests       one per distinct expert id the kernel multiplies with
//   hits           slice already resident (pinned or populated in time)
//   misses         slice not resident (demand fault ahead)
//   prefetched     expert-slice ranges covered by enqueues
//   prefetch_hits  prefetched ranges that arrived before their next use
//   defer_wait_ns  time drained inside ggml_moe_prefetch_wait
//   resident/capacity_bytes  mincore-accounted mapping residency
//
// The weights live in an anonymous mmap that is deliberately never written:
// untouched anonymous pages are non-resident, which makes hit/miss/madvise
// deterministic without touching page-cache state.
#include "ggml-moe-stats.h"
#include "ggml-moe-prefetch.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

namespace {

int failures = 0;

void require(bool condition, const std::string & message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message.c_str());
        ++failures;
    }
}

ggml_moe_prefetch_stats snapshot() {
    ggml_moe_prefetch_stats out;
    ggml_moe_prefetch_get_stats(&out);
    return out;
}

struct test_weights {
    ggml_context * ctx = nullptr;
    ggml_tensor * w = nullptr;
    void * mapping = nullptr;
    size_t mapping_size = 0;
    size_t slice = 0;

    // n_experts slices of f16 rows, backed by an untouched anonymous mmap.
    test_weights(int n_experts, int rows) {
        slice = (size_t) rows * ggml_row_size(GGML_TYPE_F16, 64);
        mapping_size = (size_t) n_experts * slice + 2 * 4096;
        mapping = mmap(nullptr, mapping_size, PROT_READ,
                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) {
            mapping = nullptr;
            return;
        }

        ctx = ggml_init({/*.mem_size   =*/ ggml_tensor_overhead() * 8,
                         /*.mem_buffer =*/ nullptr,
                         /*.no_alloc   =*/ true});
        w = ggml_new_tensor(ctx, GGML_TYPE_F16, 3, std::vector<int64_t>{64, rows, n_experts}.data());
        w->nb[1] = ggml_row_size(GGML_TYPE_F16, 64);
        w->nb[2] = slice;
        w->nb[3] = slice * (size_t) n_experts;
        w->data = mapping;
        ggml_moe_prefetch_register_mapping(mapping, mapping_size);
    }

    ~test_weights() {
        if (ctx) ggml_free(ctx);
        if (mapping) {
            ggml_moe_prefetch_unregister_mapping(mapping);
            munmap(mapping, mapping_size);
        }
    }
};

struct ids_ctx {
    ggml_context * ctx = nullptr;
    ggml_tensor * node = nullptr;

    ids_ctx(ggml_tensor * w) {
        ctx = ggml_init({ggml_tensor_overhead() * 8, nullptr, /*no_alloc =*/ false});
        node = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, std::vector<int64_t>{64, 1}.data());
        node->op = GGML_OP_MUL_MAT_ID;
        node->src[0] = w;
        node->src[2] = nullptr; // set per step
    }
    ~ids_ctx() { if (ctx) ggml_free(ctx); }

    void set_ids(const std::vector<int32_t> & ids) {
        ggml_tensor * t = ggml_new_tensor(ctx, GGML_TYPE_I32, 2,
                std::vector<int64_t>{(int64_t) ids.size(), 1}.data());
        auto * p = (int32_t *) t->data;
        for (size_t i = 0; i < ids.size(); ++i) p[i] = ids[i];
        node->src[2] = t;
    }
};

void test_per_expert_counters() {
    ggml_moe_prefetch_reset_stats();
    auto s0 = snapshot();
    require(s0.requests == 0 && s0.hits == 0 && s0.misses == 0 && s0.prefetched == 0,
            "reset zeroes all counters");

#ifdef __linux__
    test_weights tw(4, 128);
    require(tw.mapping != nullptr, "anonymous mmap for weights");
    ids_ctx ic(tw.w);

    // A) engine off: kernel entry still attributes per-expert requests/misses
    ggml_moe_prefetch_set_n_threads(0);
    ic.set_ids({0, 1});
    ggml_moe_prefetch_kernel_hook(ic.node, 0);
    auto sa = snapshot();
    require(sa.requests == 2, "engine-off: one request per distinct expert id");
    require(sa.misses == 2, "engine-off: untouched slices are misses");
    require(sa.hits == 0, "engine-off: no hits on cold mapping");

    // B) engine on: manifest-populate expert 2, sync completion
    ggml_moe_prefetch_set_n_threads(2);
    require(ggml_moe_prefetch_enabled(), "engine enabled with 2 threads");
    uint32_t selected[] = {2};
    require(ggml_moe_prefetch_experts(tw.w, selected, 1), "manifest populate succeeds");
    auto sb = snapshot();
    require(sb.prefetched == 1, "populate enqueues one slice range");
    require(sb.capacity_bytes >= 4 * tw.slice, "capacity covers the tensor mapping");
    require(sb.resident_bytes >= tw.slice, "populated slice accounted as resident");

    // C) kernel entry for the populated expert: hit + prefetch hit
    ggml_moe_prefetch_new_epoch();
    ic.set_ids({2});
    ggml_moe_prefetch_kernel_hook(ic.node, 0);
    auto sc = snapshot();
    require(sc.requests == 3, "populated-expert entry counted");
    require(sc.hits == 1, "pinned populated expert is a hit");
    require(sc.misses == 2, "miss count unchanged");
    require(sc.prefetch_hits == 1, "in-time populate attributed as prefetch hit");
    require(sc.prefetched == 2, "kernel hook re-enqueued the selected slice");

    // D) wait on drained work must not add meaningful defer time
    ggml_moe_prefetch_wait(tw.w);
    auto sd = snapshot();
    require(sd.defer_wait_ns < 100000000ULL, "idle wait adds negligible defer time");

    // E) cold experts 0 and 3: misses; their ranges enqueue for later
    ggml_moe_prefetch_new_epoch();
    ic.set_ids({0, 3});
    ggml_moe_prefetch_kernel_hook(ic.node, 0);
    auto se = snapshot();
    require(se.requests == 5, "two more per-expert requests");
    require(se.misses == 4, "cold experts 0 and 3 are misses");
    require(se.hits == 1, "hit count unchanged");
    require(se.prefetched == 4, "two more slice ranges enqueued");

    // F) drain the workers deterministically
    ggml_moe_prefetch_wait(tw.w);

    // G) re-enter with expert 3: the step-E prefetch arrived in time
    ggml_moe_prefetch_new_epoch();
    ic.set_ids({3});
    ggml_moe_prefetch_kernel_hook(ic.node, 0);
    auto sg = snapshot();
    require(sg.requests == 6, "final entry counted");
    require(sg.hits == 2, "prefetched-then-resident expert is a hit");
    require(sg.misses == 4, "miss count unchanged");
    require(sg.prefetch_hits == 2, "second in-time prefetch attributed");
    require(sg.defer_wait_ns < 1000000000ULL, "defer wait stays well under a second");

    ggml_moe_prefetch_set_n_threads(0);
#endif
}

} // namespace

// Llama-level API: snapshot passthrough from the ggml engine counters.
void test_llama_level_getter() {
    llama_expert_store_stats_lg lg = {};
    const bool had_data = llama_get_expert_store_stats(&lg);
    const ggml_moe_prefetch_stats g = snapshot();
    require(lg.requests == g.requests, "llama getter mirrors ggml requests");
    require(lg.hits == g.hits, "llama getter mirrors ggml hits");
    require(lg.misses == g.misses, "llama getter mirrors ggml misses");
    require(lg.prefetch_hits == g.prefetch_hits, "llama getter mirrors ggml prefetch hits");
    require(lg.resident_bytes == g.resident_bytes, "llama getter mirrors resident bytes");
    require(had_data == (g.requests > 0), "llama getter reports collection state");

    llama_expert_store_stats_lg fresh = {};
    require(llama_get_expert_store_stats(&fresh) == (fresh.requests > 0),
            "getter handles a zeroed engine consistently");
}

void test_shadow_lru_budget() {
#ifdef __linux__
    ggml_moe_prefetch_reset_stats();
    ggml_moe_prefetch_set_cache_sim_capacity(0);

    test_weights tw(4, 128);
    require(tw.mapping != nullptr, "shadow LRU: anonymous mmap for weights");
    ids_ctx ic(tw.w);
    ggml_moe_prefetch_set_n_threads(0);

    // Disabled shadow mode must remain silent, not report a zero-budget miss.
    ic.set_ids({0});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0);
    auto disabled = snapshot();
    require(disabled.cache_sim_capacity_bytes == 0 && disabled.cache_sim_requests == 0,
            "shadow LRU: zero capacity disables simulation");

    ggml_moe_prefetch_reset_stats();
    ggml_moe_prefetch_set_cache_sim_capacity(2 * tw.slice);

    ic.set_ids({0});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0); // miss: [0]
    ggml_moe_prefetch_new_epoch();
    ic.set_ids({1});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0); // miss: [1,0]
    ggml_moe_prefetch_new_epoch();
    ic.set_ids({0});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0); // hit/promote: [0,1]
    ggml_moe_prefetch_new_epoch();
    ic.set_ids({2});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0); // miss/evict 1: [2,0]

    const auto simulated = snapshot();
    require(simulated.cache_sim_capacity_bytes == 2 * tw.slice,
            "shadow LRU: configured byte capacity is reported");
    require(simulated.cache_sim_resident_bytes == 2 * tw.slice,
            "shadow LRU: residency stays at capacity");
    require(simulated.cache_sim_requests == 4 && simulated.cache_sim_hits == 1 &&
            simulated.cache_sim_misses == 3,
            "shadow LRU: access counters follow the trace");
    require(simulated.cache_sim_evictions == 1 &&
            simulated.cache_sim_evicted_bytes == tw.slice,
            "shadow LRU: one least-recent expert slice is evicted");
    require(simulated.cache_sim_bypasses == 0,
            "shadow LRU: fitting expert slices are never bypassed");

    llama_expert_store_stats_lg public_stats = {};
    require(llama_get_expert_store_stats(&public_stats),
            "shadow LRU: public getter reports collected expert data");
    require(public_stats.cache_sim_capacity_bytes == simulated.cache_sim_capacity_bytes &&
            public_stats.cache_sim_resident_bytes == simulated.cache_sim_resident_bytes,
            "shadow LRU: public getter mirrors cache budget and residency");
    require(public_stats.cache_sim_requests == simulated.cache_sim_requests &&
            public_stats.cache_sim_hits == simulated.cache_sim_hits &&
            public_stats.cache_sim_misses == simulated.cache_sim_misses &&
            public_stats.cache_sim_evictions == simulated.cache_sim_evictions &&
            public_stats.cache_sim_evicted_bytes == simulated.cache_sim_evicted_bytes &&
            public_stats.cache_sim_bypasses == simulated.cache_sim_bypasses,
            "shadow LRU: public getter mirrors all cache counters");

    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();
#endif
}

void test_cpu_cplan_routes_shadow_without_prefetch() {
#ifdef __linux__
    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();

    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, false});
    require(ctx != nullptr, "cplan shadow: context allocation");
    if (!ctx) return;

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 2);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 1, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, 1);
    *(int32_t *) ids->data = 0;
    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr, "cplan shadow: CPU backend allocation");
    if (!backend) {
        ggml_free(ctx);
        return;
    }
    ggml_backend_cpu_set_n_threads(backend, 1);
    ggml_backend_cpu_set_moe_expert_prefetch(backend, false);
    ggml_moe_prefetch_set_cache_sim_capacity(2 * weights->nb[2]);

    ggml_backend_cpu_set_moe_expert_cache_sim(backend, false);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "cplan shadow: baseline graph compute");
    require(snapshot().cache_sim_requests == 0,
            "cplan shadow: disabled plan bit stays silent");

    ggml_backend_cpu_set_moe_expert_cache_sim(backend, true);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "cplan shadow: instrumented graph compute");
    const auto observed = snapshot();
    require(observed.cache_sim_requests == 1 && observed.cache_sim_misses == 1,
            "cplan shadow: enabled plan bit routes one expert slice");
    require(observed.requests == 0 && observed.prefetched == 0,
            "cplan shadow: graph route does not enable residency or prefetch lane");

    ggml_backend_free(backend);
    ggml_free(ctx);
    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();
#endif
}

void test_shadow_has_single_explicit_owner() {
#ifdef __linux__
    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();

    test_weights tw(4, 128);
    require(tw.mapping != nullptr, "shadow owner: mmap failed");
    ids_ctx ic(tw.w);

    require(ggml_moe_cache_sim_try_acquire(2 * tw.slice),
            "shadow owner: first context acquires the simulator");
    ic.set_ids({0});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0);
    require(!ggml_moe_cache_sim_try_acquire(3 * tw.slice),
            "shadow owner: second context is rejected");
    ggml_moe_prefetch_set_cache_sim_capacity(0);

    const auto retained = snapshot();
    require(retained.cache_sim_capacity_bytes == 2 * tw.slice &&
            retained.cache_sim_requests == 1,
            "shadow owner: rejected acquire does not reset owner trace");

    ggml_moe_cache_sim_release();
    const auto released = snapshot();
    require(released.cache_sim_capacity_bytes == 0 && released.cache_sim_requests == 0,
            "shadow owner: release clears capacity and trace");
    require(ggml_moe_cache_sim_try_acquire(3 * tw.slice),
            "shadow owner: another context can acquire after release");
    ggml_moe_cache_sim_release();
    ggml_moe_prefetch_reset_stats();
#endif
}

void test_shadow_accounts_exact_last_slice_bytes() {
#ifdef __linux__
    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();

    test_weights tw(2, 128);
    require(tw.mapping != nullptr, "shadow bytes: mmap failed");
    const size_t padded_stride = tw.slice + 4096;
    tw.w->nb[2] = padded_stride;
    ids_ctx ic(tw.w);
    ggml_moe_prefetch_set_cache_sim_capacity(2 * padded_stride);

    ic.set_ids({1});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0);
    const auto observed = snapshot();
    require(observed.cache_sim_resident_bytes == tw.slice,
            "shadow bytes: last expert uses the truncated tensor tail, not full stride");

    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();
#endif
}

void test_prefetch_hook_does_not_implicitly_feed_shadow() {
#ifdef __linux__
    ggml_moe_prefetch_set_n_threads(0);
    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();

    test_weights tw(4, 128);
    require(tw.mapping != nullptr, "prefetch isolation: mmap failed");
    ids_ctx ic(tw.w);
    ggml_moe_prefetch_set_cache_sim_capacity(2 * tw.slice);

    ic.set_ids({0});
    ggml_moe_prefetch_kernel_hook(ic.node, 0);
    const auto observed = snapshot();
    require(observed.cache_sim_requests == 0,
            "prefetch isolation: only the cache-sim plan hook may feed the shadow");

    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();
#endif
}

void test_shadow_only_hook_does_not_prefetch_or_touch_residency_stats() {
#ifdef __linux__
    ggml_moe_prefetch_set_n_threads(0);
    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();

    test_weights tw(4, 128);
    require(tw.mapping != nullptr, "shadow-only: mmap failed");
    ids_ctx ic(tw.w);
    const size_t stride = tw.w->nb[2];
    ggml_moe_prefetch_set_cache_sim_capacity(2 * stride);

    ic.set_ids({0});
    ggml_moe_cache_sim_kernel_hook(ic.node, 0);
    const auto observed = snapshot();

    require(observed.cache_sim_requests == 1,
            "shadow-only: one expert slice is observed");
    require(observed.cache_sim_misses == 1,
            "shadow-only: first access is a simulated miss");
    require(observed.requests == 0,
            "shadow-only: mmap residency requests remain untouched");
    require(observed.prefetched == 0,
            "shadow-only: no prefetch is recorded");

    ggml_moe_prefetch_set_cache_sim_capacity(0);
    ggml_moe_prefetch_reset_stats();
#else
    std::cout << "test-moe-prefetch-stats: shadow-only test skipped (non-Linux)\n";
#endif
}

int main() {
    // GGML_MOE_STATS=0 disables mmap-residency attribution; the explicitly
    // configured shadow test below remains independent of that kill switch.
    test_per_expert_counters();
    test_llama_level_getter();
    test_shadow_lru_budget();
    test_cpu_cplan_routes_shadow_without_prefetch();
    test_shadow_has_single_explicit_owner();
    test_shadow_accounts_exact_last_slice_bytes();
    test_prefetch_hook_does_not_implicitly_feed_shadow();
    test_shadow_only_hook_does_not_prefetch_or_touch_residency_stats();
    if (failures == 0) {
        printf("test-moe-prefetch-stats: OK\n");
        return 0;
    }
    fprintf(stderr, "test-moe-prefetch-stats: %d failure(s)\n", failures);
    return 1;
}
