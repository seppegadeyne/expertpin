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

int main() {
    // GGML_MOE_STATS=0 (checked once in the engine) disables attribution;
    // the per-expert assertions below assume the default-on configuration.
    test_per_expert_counters();
    test_llama_level_getter();
    if (failures == 0) {
        printf("test-moe-prefetch-stats: OK\n");
        return 0;
    }
    fprintf(stderr, "test-moe-prefetch-stats: %d failure(s)\n", failures);
    return 1;
}
