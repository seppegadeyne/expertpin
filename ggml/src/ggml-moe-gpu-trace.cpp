#include "ggml-moe-gpu-trace.h"
#include <cstdlib>
#include <memory>
#include <mutex>

namespace {
struct gpu_trace_state {
    std::mutex mutex;
    FILE * file = nullptr;
    std::unique_ptr<ggml_moe_gpu_trace> trace;
    gpu_trace_state() {
        const char * path = std::getenv("GGML_MOE_GPU_TRACE_FILE");
        if (!path || !*path) return;
        file = std::fopen(path, "wx");
        if (!file) {
            std::fprintf(stderr, "GPU expert trace: cannot create %s; tracing disabled\n", path);
            return;
        }
        trace.reset(new ggml_moe_gpu_trace(file));
        std::fprintf(stderr, "GPU expert trace: request-only IDs, no shadow hit semantics; CUDA graphs disabled\n");
    }
    ~gpu_trace_state() {
        trace.reset();
        if (file) std::fclose(file);
    }
};
gpu_trace_state & state() { static gpu_trace_state s; return s; }
}

bool ggml_moe_gpu_trace_enabled() { return state().trace != nullptr; }
void ggml_moe_gpu_trace_set_scope(ggml_moe_trace_scope scope) {
    auto & s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.trace) s.trace->set_scope(scope);
}
void ggml_moe_gpu_trace_record(const ggml_tensor * w0, const ggml_tensor * w1, const ggml_tensor * ids,
        ggml_moe_gpu_ids_reader reader, void * user) {
    auto & s = state();
    if (!s.trace) return;
    std::lock_guard<std::mutex> lock(s.mutex);
    s.trace->record(w0, w1, ids, reader, user);
}
