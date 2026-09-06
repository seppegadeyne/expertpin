#include "ggml-cuda-transfer-trace.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>

namespace {
const char * phase_name(int phase) {
    return phase == 1 ? "prefill" : phase == 2 ? "decode" : "mixed";
}
const char * host_name(ggml_cuda_host_kind kind) {
    switch (kind) {
        case ggml_cuda_host_kind::pageable: return "pageable";
        case ggml_cuda_host_kind::pinned: return "pinned";
        case ggml_cuda_host_kind::device: return "device";
        case ggml_cuda_host_kind::managed: return "managed";
        default: return "unknown";
    }
}
// Bounded, quoted CSV name. Control characters become '?', quotes are doubled.
void csv_name(const char * name, char * out) {
    *out++ = '"';
    for (size_t i = 0; name && i < GGML_MAX_NAME && name[i]; ++i) {
        const unsigned char c = static_cast<unsigned char>(name[i]);
        if (c == '"') *out++ = '"';
        *out++ = c < 32 || c == 127 ? '?' : static_cast<char>(c);
    }
    *out++ = '"'; *out = 0;
}
struct transfer_state {
    std::mutex mutex;
    FILE * file = nullptr;
    std::unique_ptr<ggml_cuda_transfer_writer> writer;
    transfer_state() {
        const char * path = std::getenv("GGML_CUDA_TRANSFER_TRACE_FILE");
        if (!path || !*path) return;
        file = std::fopen(path, "wx");
        if (!file) {
            std::fprintf(stderr, "CUDA transfer trace: cannot create %s; disabled\n", path);
            return;
        }
        writer.reset(new ggml_cuda_transfer_writer(file));
    }
    ~transfer_state() {
        writer.reset();
        if (file && std::fclose(file) != 0) std::fprintf(stderr, "CUDA transfer trace: close failed\n");
    }
};
transfer_state & state() { static transfer_state s; return s; }
}

ggml_cuda_transfer_writer::ggml_cuda_transfer_writer(FILE * file, uint64_t limit)
    : file_(file), limit_(std::min(limit, max_records)) {
    if (!file_) { ++io_errors_; return; }
    if (std::fprintf(file_, "# cuda_transfer_trace_v1 coverage=ordinary_backend_set_get_only timing=host_API_wall_not_DMA\n"
            "id,context,request_id,seq_id,phase,pos_min,pos_max,backend_api,direction,tensor,offset_bytes,size_bytes,device,host_kind,pointer_status,start_monotonic_ns,copy_api_wall_ns,sync_api_wall_ns,copy_status,sync_status\n") < 0 ||
            std::fflush(file_) != 0) ++io_errors_;
}
ggml_cuda_transfer_writer::~ggml_cuda_transfer_writer() {
    if (!file_) return;
    if (std::fflush(file_) != 0) ++io_errors_;
    const auto errors = api_errors_ + pointer_errors_ + io_errors_;
    if (std::fprintf(file_, "# end admitted=%llu written=%llu dropped=%llu errors=%llu api_errors=%llu pointer_errors=%llu io_errors=%llu\n",
            (unsigned long long) admitted_, (unsigned long long) written_, (unsigned long long) dropped_,
            (unsigned long long) errors, (unsigned long long) api_errors_,
            (unsigned long long) pointer_errors_, (unsigned long long) io_errors_) < 0 || std::fflush(file_) != 0) {
        std::fprintf(stderr, "CUDA transfer trace: footer write failed; evidence incomplete\n");
    }
}
ggml_cuda_transfer_ticket ggml_cuda_transfer_writer::begin() {
    ggml_cuda_transfer_ticket ticket;
    if (scope_.request_id < 0 || scope_.seq_id < 0 || scope_.phase < 1 || scope_.phase > 3) return ticket;
    if (io_errors_ || admitted_ >= limit_) { ++dropped_; return ticket; }
    ticket.active = true; ticket.id = admitted_++; ticket.scope = scope_;
    return ticket;
}
void ggml_cuda_transfer_writer::record(const ggml_cuda_transfer_ticket & ticket, const ggml_cuda_transfer_event & e) {
    if (!ticket.active) return;
    if (e.copy_status != 0 || e.sync_status > 0) ++api_errors_;
    if (e.pointer_status > 0 || e.pointer_status == -2) ++pointer_errors_;
    if (io_errors_) { ++dropped_; return; }
    char name[2 * GGML_MAX_NAME + 3]; csv_name(e.tensor_name, name);
    // Fixed-size row guarantees a byte bound as well as a record bound.
    char row[1024];
    const int n = std::snprintf(row, sizeof(row),
        "%llu,target,%lld,%d,%s,%d,%d,%s%s,%s,%s,%zu,%zu,%d,%s,%d,%llu,%llu,%llu,%d,%d\n",
        (unsigned long long) ticket.id, (long long) ticket.scope.request_id, ticket.scope.seq_id,
        phase_name(ticket.scope.phase), ticket.scope.pos_min, ticket.scope.pos_max,
        e.h2d ? "set_tensor" : "get_tensor", e.synchronous ? "" : "_async", e.h2d ? "H2D" : "D2H", name,
        e.offset, e.size, e.device, host_name(e.host_kind), e.pointer_status,
        (unsigned long long) e.start_ns, (unsigned long long) e.copy_wall_ns, (unsigned long long) e.sync_wall_ns,
        e.copy_status, e.sync_status);
    if (n < 0 || size_t(n) >= sizeof(row) || std::fwrite(row, 1, size_t(n), file_) != size_t(n) || std::fflush(file_) != 0) {
        ++io_errors_; ++dropped_; return;
    }
    ++written_;
}
bool ggml_cuda_transfer_trace_enabled() { return state().writer != nullptr; }
void ggml_cuda_transfer_trace_set_scope(ggml_moe_trace_scope scope) {
    auto & s = state(); if (!s.writer) return;
    std::lock_guard<std::mutex> lock(s.mutex); s.writer->set_scope(scope);
}
ggml_cuda_transfer_ticket ggml_cuda_transfer_trace_begin() {
    auto & s = state(); if (!s.writer) return {};
    std::lock_guard<std::mutex> lock(s.mutex); return s.writer->begin();
}
void ggml_cuda_transfer_trace_record(const ggml_cuda_transfer_ticket & ticket, const ggml_cuda_transfer_event & event) {
    if (!ticket.active) return;
    auto & s = state(); if (!s.writer) return;
    std::lock_guard<std::mutex> lock(s.mutex); s.writer->record(ticket, event);
}
