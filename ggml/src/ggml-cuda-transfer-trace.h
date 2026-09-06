#pragma once

#include "ggml-moe-stats.h"
#include <cstdio>
#include <cstdint>

// CPU-only writer. No CUDA dependency; callers serialize access. Tickets snapshot
// the request at admission, before any CUDA work, and bound even in-flight rows.
struct ggml_cuda_transfer_ticket {
    bool active = false;
    uint64_t id = 0;
    ggml_moe_trace_scope scope = {-1, -1, 0, -1, -1};
};
enum class ggml_cuda_host_kind { unknown, pageable, pinned, device, managed };
struct ggml_cuda_transfer_event {
    const char * tensor_name = "";
    size_t offset = 0, size = 0;
    int device = -1;
    bool h2d = true, synchronous = false;
    ggml_cuda_host_kind host_kind = ggml_cuda_host_kind::unknown;
    int pointer_status = -1; // -1 unsupported, -2 prior CUDA error, -3 no query
    int copy_status = 0, sync_status = -1; // -1: sync not called
    uint64_t start_ns = 0, copy_wall_ns = 0, sync_wall_ns = 0;
};
class ggml_cuda_transfer_writer {
public:
    static constexpr uint64_t max_records = 100000;
    explicit ggml_cuda_transfer_writer(FILE * file, uint64_t limit = max_records);
    ~ggml_cuda_transfer_writer();
    ggml_cuda_transfer_writer(const ggml_cuda_transfer_writer &) = delete;
    ggml_cuda_transfer_writer & operator=(const ggml_cuda_transfer_writer &) = delete;
    void set_scope(ggml_moe_trace_scope scope) { scope_ = scope; }
    ggml_cuda_transfer_ticket begin();
    void record(const ggml_cuda_transfer_ticket & ticket, const ggml_cuda_transfer_event & event);
private:
    FILE * file_;
    uint64_t limit_, admitted_ = 0, written_ = 0, dropped_ = 0;
    uint64_t api_errors_ = 0, pointer_errors_ = 0, io_errors_ = 0;
    ggml_moe_trace_scope scope_ = {-1, -1, 0, -1, -1};
};

bool ggml_cuda_transfer_trace_enabled();
void ggml_cuda_transfer_trace_set_scope(ggml_moe_trace_scope scope);
ggml_cuda_transfer_ticket ggml_cuda_transfer_trace_begin();
void ggml_cuda_transfer_trace_record(const ggml_cuda_transfer_ticket & ticket, const ggml_cuda_transfer_event & event);
