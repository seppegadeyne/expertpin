#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Cumulative runtime counters for the MoE expert prefetch engine (issue #1).
// They mirror the colibri ColiExpertStoreStats contract at ggml level:
//   requests         - MoE matmul kernel entries (MUL_MAT_ID / MOE_FUSED_UP_GATE)
//   hits             - kernel entries where every requested expert slice was
//                      already resident (no demand fault / cold read needed)
//   misses           - kernel entries that needed at least one cold read
//   prefetched       - expert slices covered by prefetch enqueues
//                      (selective node enqueues + manifest-resident populates;
//                      lookahead full-tensor sweeps are excluded)
//   prefetch_hits    - prefetched slices that later arrived at a kernel entry
//                      already resident (in-time prefetch)
//   defer_wait_ns    - time spent inside ggml_moe_prefetch_wait (the deferred
//                      I/O path: waiting for streaming workers to finish)
//   resident_bytes / capacity_bytes - mmapped weight bytes; capacity is the
//                      full mapping size, resident the pages mincore() reports
//                      populated at snapshot time
// The counters are advisory-only: collecting them never gates or reorders
// execution, so --resident-experts 0 behavior stays bit-identical.
struct ggml_moe_prefetch_stats {
    uint64_t requests;
    uint64_t hits;
    uint64_t misses;
    uint64_t prefetched;
    uint64_t prefetch_hits;
    uint64_t defer_wait_ns;
    uint64_t resident_bytes;
    uint64_t capacity_bytes;

    // Advisory LRU shadow for a future hard-bounded host expert cache. These
    // counters do not affect execution or OS page residency.
    uint64_t cache_sim_requests;
    uint64_t cache_sim_hits;
    uint64_t cache_sim_misses;
    uint64_t cache_sim_evictions;
    uint64_t cache_sim_evicted_bytes;
    uint64_t cache_sim_bypasses;
    uint64_t cache_sim_resident_bytes;
    uint64_t cache_sim_capacity_bytes;
};

// Copy out a consistent snapshot of the counters (atomically read, zeroed
// struct when the engine has never run).
void ggml_moe_prefetch_get_stats(struct ggml_moe_prefetch_stats * out);

// Reset all counters to zero (used by tests and at context teardown).
void ggml_moe_prefetch_reset_stats(void);

#ifdef __cplusplus
}
#endif
