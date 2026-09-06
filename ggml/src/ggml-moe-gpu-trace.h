#pragma once

#include "ggml-moe-trace.h"
#include <cstring>
#include <limits>
#include <vector>

// CUDA supplies a stream-ordered, packed IDs-only reader. CPU tests supply an
// in-memory reader. No weight data pointer is ever dereferenced here.
using ggml_moe_gpu_ids_reader = bool (*)(const ggml_tensor *, int32_t *, size_t, size_t, void *);

class ggml_moe_gpu_trace {
public:
    static constexpr size_t max_ids_bytes = 64 * 1024;
    static constexpr size_t max_total_ids_bytes = 64 * 1024 * 1024;
    static constexpr uint64_t max_entries = 100000;
    explicit ggml_moe_gpu_trace(FILE * file, uint64_t records = ggml_moe_trace::max_records,
            size_t read_budget = max_total_ids_bytes, uint64_t entries = max_entries)
        : csv_(file, records, true), budget_(std::min(read_budget, max_total_ids_bytes)),
          entries_(std::min(entries, max_entries)) {}

    void set_scope(ggml_moe_trace_scope scope) {
        active_ = scope.phase >= 1 && scope.phase <= 3 && scope.request_id >= 0 && scope.seq_id >= 0;
        if (!active_) scope.phase = 0;
        csv_.set_scope(scope);
        if (active_) ++epoch_;
    }
    void finish() { csv_.finish(); }
    void record(const ggml_tensor * w0, const ggml_tensor * w1, const ggml_tensor * ids,
            ggml_moe_gpu_ids_reader reader, void * user) {
        if (!active_) return;
        if (stopped_ || csv_.remaining() == 0) { csv_.incomplete(); stopped_ = true; return; }
        if (entries_ == 0) { reject(); return; }
        --entries_;
        if (!ids || !ids->data || ids->type != GGML_TYPE_I32 || ids->ne[0] <= 0 || ids->ne[1] <= 0 ||
                ids->ne[2] != 1 || ids->ne[3] != 1 || ids->nb[0] != sizeof(int32_t) ||
                uint64_t(ids->ne[0]) > max_ids_bytes / sizeof(int32_t)) { reject(); return; }
        const size_t row = size_t(ids->ne[0]) * sizeof(int32_t);
        if (uint64_t(ids->ne[1]) > max_ids_bytes / row || ids->nb[1] < row ||
                (uint64_t(ids->ne[1]) > 1 && ids->nb[1] > (SIZE_MAX - row) / size_t(ids->ne[1] - 1))) {
            reject(); return;
        }
        const size_t bytes = row * size_t(ids->ne[1]);
        if (bytes > budget_ || !valid_weight(w0) || (w1 && !valid_weight(w1))) { reject(); return; }
        budget_ -= bytes;
        std::vector<int32_t> selected(bytes / sizeof(int32_t));
        if (!reader || !reader(ids, selected.data(), row, size_t(ids->ne[1]), user)) { reject(); return; }
        // -1 is the runtime's pruned-expert sentinel; other invalid IDs fail closed.
        for (int32_t id : selected) {
            if (id < -1 || (id >= 0 && (id >= w0->ne[2] || (w1 && id >= w1->ne[2])))) { reject(); return; }
        }
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
        for (const ggml_tensor * w : {w0, w1}) {
            if (!w) continue;
            const uint64_t entry = entry_++;
            for (int32_t id : selected) {
                if (id < 0) continue;
                csv_.record(entry, epoch_, ids->ne[1], w->name, w->type, uint32_t(id),
                        w->nb[2], size_t(id) * w->nb[2], w->nb[2], false, true);
            }
        }
    }
private:
    static bool valid_weight(const ggml_tensor * w) {
        // Only ordinary contiguous expert slices; repacked layouts are not file offsets.
        return w && w->ne[0] > 0 && w->ne[1] > 0 && w->ne[2] > 0 && w->ne[3] == 1 &&
            w->type >= 0 && w->type < GGML_TYPE_COUNT && ggml_is_contiguous(w) &&
            w->nb[2] > 0 && uint64_t(w->ne[2]) <= SIZE_MAX / w->nb[2];
    }
    void reject() { csv_.incomplete(); stopped_ = true; }
    ggml_moe_trace csv_;
    size_t budget_;
    uint64_t entries_, entry_ = 0, epoch_ = 0;
    bool active_ = false, stopped_ = false;
};

// Process-global, single synchronized target request; independent of CPU shadow
// ownership/capacity. Environment is read once, before the first CUDA graph.
bool ggml_moe_gpu_trace_enabled();
void ggml_moe_gpu_trace_set_scope(ggml_moe_trace_scope scope);
void ggml_moe_gpu_trace_record(const ggml_tensor * w0, const ggml_tensor * w1, const ggml_tensor * ids,
        ggml_moe_gpu_ids_reader reader, void * user);
