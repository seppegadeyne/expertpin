#include "ggml-cuda-transfer-trace.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#if defined(__linux__)
#include <unistd.h>
#endif
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

// Deliberately synthetic CUDA API double, NOT GPU measurements. Exercise the
// production wrapper's control flow without linking or initializing CUDA.
#define CUDART_VERSION 11000
using cudaError_t = int;
using cudaStream_t = int;
const int cudaSuccess = 0;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
enum { cudaMemoryTypeUnregistered, cudaMemoryTypeHost, cudaMemoryTypeDevice, cudaMemoryTypeManaged };
struct cudaPointerAttributes { int type; };
static int copies, syncs, queries, peeks, copy_result, sync_result, query_result, prior_error, memory_type;
static const void * queried_host;
static cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t) { ++copies; return copy_result; }
static cudaError_t cudaStreamSynchronize(cudaStream_t) { ++syncs; return sync_result; }
static cudaError_t cudaPeekAtLastError() { ++peeks; return prior_error; }
static cudaError_t cudaPointerGetAttributes(cudaPointerAttributes * attr, const void * host) {
    ++queries; queried_host = host; attr->type = memory_type; return query_result;
}
#define CUDA_CHECK(x) do { int rc = (x); if (rc) throw rc; } while (0)
#include "ggml-cuda/transfer-trace.cuh"

static std::string contents(FILE * file) {
    std::rewind(file); std::string result; char buf[1024];
    while (size_t n = std::fread(buf, 1, sizeof(buf), file)) result.append(buf, n);
    return result;
}
static void writer_tests() {
    FILE * f = std::tmpfile(); CHECK(f);
    {
        ggml_cuda_transfer_writer writer(f, 2);
        CHECK(!writer.begin().active);
        writer.set_scope({42, 0, 1, 0, 9});
        auto first = writer.begin(); CHECK(first.active);
        writer.set_scope({42, 0, 2, 10, 13});
        auto second = writer.begin(); CHECK(second.active);
        CHECK(!writer.begin().active); // bound reservations, not only emitted rows
        ggml_cuda_transfer_event e;
        e.tensor_name = "blk.1,\"weight\"\n"; e.offset = 7; e.size = 123;
        e.host_kind = ggml_cuda_host_kind::pageable; e.pointer_status = 0;
        e.start_ns = 100; e.copy_wall_ns = 20; e.sync_wall_ns = 30;
        e.synchronous = true; e.sync_status = 0;
        writer.record(first, e); // snapshot survives phase change
        e.h2d = false; e.synchronous = false; e.host_kind = ggml_cuda_host_kind::pinned;
        e.copy_status = 7; e.pointer_status = 8; e.sync_status = -1;
        writer.record(second, e);
        for (auto scope : {ggml_moe_trace_scope{-1, 0, 2, 0, 0}, {42, -1, 2, 0, 0}, {42, 0, 0, 0, 0}, {42, 0, 4, 0, 0}}) {
            writer.set_scope(scope); CHECK(!writer.begin().active);
        }
    }
    auto text = contents(f);
    CHECK(text.find("0,target,42,0,prefill,0,9,set_tensor,H2D,\"blk.1,\"\"weight\"\"?\",7,123,-1,pageable,0,100,20,30,0,0") != std::string::npos);
    CHECK(text.find("1,target,42,0,decode,10,13,get_tensor_async,D2H,") != std::string::npos);
    CHECK(text.find("# end admitted=2 written=2 dropped=1 errors=2 api_errors=1 pointer_errors=1 io_errors=0") != std::string::npos);
    CHECK(text.find("0x") == std::string::npos);
    std::fclose(f);
    // Zero cap, bounded unterminated names, mixed scope, all host kinds.
    f = std::tmpfile(); CHECK(f);
    {
        ggml_cuda_transfer_writer writer(f, 0); writer.set_scope({1, 0, 3, 1, 2});
        CHECK(!writer.begin().active);
    }
    CHECK(contents(f).find("admitted=0 written=0 dropped=1 errors=0") != std::string::npos); std::fclose(f);
    f = std::tmpfile(); CHECK(f);
    {
        ggml_cuda_transfer_writer writer(f); writer.set_scope({1, 0, 3, 1, 2});
        char name[GGML_MAX_NAME]; std::memset(name, '"', sizeof(name));
        ggml_cuda_transfer_event e; e.tensor_name = name;
        for (auto kind : {ggml_cuda_host_kind::unknown, ggml_cuda_host_kind::device, ggml_cuda_host_kind::managed}) {
            e.host_kind = kind; writer.record(writer.begin(), e);
        }
    }
    text = contents(f);
    CHECK(text.find(",mixed,") != std::string::npos);
    CHECK(text.find(",unknown,") != std::string::npos);
    CHECK(text.find(",device,") != std::string::npos);
    CHECK(text.find(",managed,") != std::string::npos);
    CHECK(text.size() < 4096); std::fclose(f);
#if defined(__linux__)
    f = std::fopen("/dev/full", "w"); CHECK(f);
    { ggml_cuda_transfer_writer writer(f); writer.set_scope({1, 0, 2, 0, 0}); CHECK(!writer.begin().active); }
    std::fclose(f);
#endif
}
int main(int argc, char ** argv) {
    writer_tests();
#if defined(__linux__)
    const bool enabled = argc == 2 && std::string(argv[1]) == "enabled";
    const bool collision = argc == 2 && std::string(argv[1]) == "collision";
    unsetenv("GGML_MOE_TRACE_FILE"); unsetenv("GGML_MOE_GPU_TRACE_FILE"); unsetenv("GGML_CUDA_TRANSFER_TRACE_FILE");
    char path[] = "cuda-transfer-test-XXXXXX";
    if (enabled || collision) {
        int fd = mkstemp(path); CHECK(fd >= 0); close(fd);
        if (!collision) CHECK(unlink(path) == 0);
        CHECK(setenv("GGML_CUDA_TRANSFER_TRACE_FILE", path, 1) == 0);
    }
    CHECK(ggml_cuda_transfer_trace_enabled() == enabled);
    CHECK(ggml_moe_trace_request_scoped() == enabled); // no CPU shadow or GPU routing required
    ggml_tensor tensor = {}; std::strcpy(tensor.name, "test.weight");
    char host[8], device[8];
    auto copy = [&](bool sync, bool h2d) {
        ggml_cuda_trace_copy(h2d ? device : host, h2d ? host : device, 8,
            h2d ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, 0, sync, &tensor, 2, 0);
    };
    copy(false, true); CHECK(copies == 1 && syncs == 0 && queries == 0 && peeks == 0);
    ggml_moe_trace_set_scope({99, 0, 2, 1, 4});
    copy(true, true); copy(false, false);
    CHECK(copies == 3 && syncs == 1 && queries == (enabled ? 2 : 0));
    if (enabled) {
        CHECK(queried_host == host); // D2H destination, not device source
        prior_error = 9; copy(false, true); CHECK(queries == 2 && prior_error == 9); prior_error = 0;
        query_result = 8; copy(false, true); CHECK(queries == 3); query_result = 0;
        for (int kind : {cudaMemoryTypeHost, cudaMemoryTypeDevice, cudaMemoryTypeManaged}) {
            memory_type = kind; copy(false, true); CHECK(queried_host == host);
        }
        copy_result = 7;
        try { copy(true, true); CHECK(false); } catch (int rc) { CHECK(rc == 7); }
        CHECK(syncs == 1); copy_result = 0;
        sync_result = 6;
        try { copy(true, false); CHECK(false); } catch (int rc) { CHECK(rc == 6); }
        sync_result = 0;
        FILE * f = std::fopen(path, "r"); CHECK(f); auto s = contents(f); std::fclose(f);
        for (const char * kind : {",pageable,", ",pinned,", ",device,", ",managed,", ",unknown,"}) CHECK(s.find(kind) != std::string::npos);
        CHECK(s.find("target,99,0,decode,1,4") != std::string::npos);
        CHECK(s.find("0x") == std::string::npos);
    }
    ggml_moe_trace_set_scope({-1, -1, 0, -1, -1});
    const int before = queries; copy(false, true); CHECK(queries == before);
    if (enabled || collision) CHECK(unlink(path) == 0);
#else
    (void) argc; (void) argv;
#endif
    std::puts("CUDA transfer CPU writer and synthetic API control-flow tests passed (no GPU measurements)");
}
