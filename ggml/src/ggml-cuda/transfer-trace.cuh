#pragma once

#include "../ggml-cuda-transfer-trace.h"
#include <chrono>

// Included after the CUDA runtime declarations and CUDA_CHECK. No device events
// or extra synchronization: synchronous entrypoints retain their existing wait.
static void ggml_cuda_trace_copy(void * dst, const void * src, size_t size,
        cudaMemcpyKind kind, cudaStream_t stream, bool synchronous,
        const ggml_tensor * tensor, size_t offset, int device) {
    ggml_cuda_transfer_ticket ticket;
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    ticket = ggml_cuda_transfer_trace_begin();
#endif
    if (!ticket.active) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, kind, stream));
        if (synchronous) CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }
    ggml_cuda_transfer_event event;
    event.tensor_name = tensor->name; event.offset = offset; event.size = size;
    event.device = device; event.h2d = kind == cudaMemcpyHostToDevice; event.synchronous = synchronous;
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    const auto copy_status = cudaMemcpyAsync(dst, src, size, kind, stream);
    const auto copied = clock::now();
    event.start_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
    event.copy_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(copied - start).count();
    event.copy_status = int(copy_status);
    cudaError_t sync_status = cudaSuccess;
    if (synchronous && copy_status == cudaSuccess) {
        const auto sync_start = clock::now();
        sync_status = cudaStreamSynchronize(stream);
        event.sync_wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - sync_start).count();
        event.sync_status = int(sync_status);
    }
    // CUDA >=11 returns cudaMemoryTypeUnregistered for ordinary host memory.
    // Older runtimes can raise invalid-value for pageable pointers: do not probe
    // them, or HIP/MUSA. Never clear a CUDA error (it may be an async failure).
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11000
    const void * host = event.h2d ? src : dst;
    event.pointer_status = -3;
    if (host && size && copy_status == cudaSuccess && sync_status == cudaSuccess) {
        if (cudaPeekAtLastError() != cudaSuccess) {
            event.pointer_status = -2;
        } else {
            cudaPointerAttributes attr = {};
            const auto status = cudaPointerGetAttributes(&attr, host);
            event.pointer_status = int(status);
            if (status == cudaSuccess) {
                switch (attr.type) {
                    case cudaMemoryTypeUnregistered: event.host_kind = ggml_cuda_host_kind::pageable; break;
                    case cudaMemoryTypeHost: event.host_kind = ggml_cuda_host_kind::pinned; break;
                    case cudaMemoryTypeDevice: event.host_kind = ggml_cuda_host_kind::device; break;
                    case cudaMemoryTypeManaged: event.host_kind = ggml_cuda_host_kind::managed; break;
                    default: break;
                }
            }
        }
    }
#endif
    ggml_cuda_transfer_trace_record(ticket, event);
    CUDA_CHECK(copy_status);
    if (synchronous) CUDA_CHECK(sync_status);
}
