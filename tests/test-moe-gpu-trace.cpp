#include "ggml-moe-gpu-trace.h"
#include "ggml-moe-stats.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#if defined(__linux__)
#include <unistd.h>
#endif

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
static bool read_ids(const ggml_tensor * ids, int32_t * out, size_t row, size_t rows, void * user) {
    ++*static_cast<int *>(user);
    for (size_t i = 0; i < rows; ++i) std::memcpy((char *)out + i * row, (char *)ids->data + i * ids->nb[1], row);
    return true;
}
static std::string contents(FILE * f) {
    std::rewind(f); std::string s; char buf[512];
    while (size_t n = std::fread(buf, 1, sizeof(buf), f)) s.append(buf, n);
    return s;
}
int main() {
    ggml_init_params params = {1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params); CHECK(ctx);
    auto w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 4);
    auto gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 2, 4);
    ggml_set_name(w, "blk.0.ffn_up_exps.weight");
    ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
    // Deliberately invalid payload addresses: metadata/IDs only.
    w->data = gate->data = reinterpret_cast<void *>(uintptr_t(1));
    auto ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 2);
    int32_t raw[] = {3, 1, 3, 999, 1, -1, 2, 999};
    ids->data = raw; ids->nb[1] = 4 * sizeof(int32_t);
#if defined(__linux__)
    // Public server scope activates the GPU writer without any CPU shadow or
    // CPU trace. Unscoped init/draft callbacks must not touch IDs.
    {
        char path[] = "/tmp/expertpin-gpu-trace-test-XXXXXX";
        int fd = mkstemp(path); CHECK(fd >= 0); close(fd); CHECK(unlink(path) == 0);
        CHECK(setenv("GGML_MOE_GPU_TRACE_FILE", path, 1) == 0);
        CHECK(ggml_moe_trace_request_scoped()); CHECK(ggml_moe_gpu_trace_enabled());
        CHECK(unlink(path) == 0); // anonymous test evidence; writer closes at exit
        int reads = 0;
        ggml_moe_gpu_trace_record(w, gate, ids, read_ids, &reads); CHECK(reads == 0);
        ggml_moe_trace_set_scope({99, 0, 2, 1, 4});
        ggml_moe_gpu_trace_record(w, gate, ids, read_ids, &reads); CHECK(reads == 1);
        ggml_moe_trace_set_scope({-1, -1, 0, -1, -1});
        ggml_moe_gpu_trace_record(w, gate, ids, read_ids, &reads); CHECK(reads == 1);
    }
#endif
    {
        FILE * f = std::tmpfile(); CHECK(f); int reads = 0;
        {
            ggml_moe_gpu_trace trace(f);
            trace.record(w, gate, ids, read_ids, &reads); CHECK(reads == 0); // init
            trace.set_scope({42, 0, 1, 0, 9});
            trace.record(w, gate, ids, read_ids, &reads); CHECK(reads == 1); // one read for both weights
            trace.set_scope({-1, -1, 0, -1, -1});
            trace.record(w, gate, ids, read_ids, &reads); CHECK(reads == 1); // draft
            trace.set_scope({42, 0, 2, 10, 13});
            trace.record(w, nullptr, ids, read_ids, &reads); CHECK(reads == 2);
            trace.set_scope({42, 0, 4, 10, 13});
            trace.record(w, nullptr, ids, read_ids, &reads); CHECK(reads == 2);
        }
        auto s = contents(f);
        CHECK(s.find("# end written=9 dropped=0 error=0") != std::string::npos);
        CHECK(s.find("\"blk.0.ffn_up_exps.weight\",0,1,64,64,64,0,1,target,42,0,prefill,0,9") != std::string::npos);
        CHECK(s.find("target,42,0,decode,10,13") != std::string::npos);
        CHECK(s.find(",999,") == std::string::npos);
        std::fclose(f);
    }
    // Record, readback-byte and operation caps stop further reads and reject completeness.
    for (int mode = 0; mode < 9; ++mode) {
        FILE * f = std::tmpfile(); CHECK(f); int reads = 0;
        auto local = *ids;
        if (mode == 3) local.ne[0] = 20000; // per-read cap
        if (mode == 4) local.nb[0] = 8; // unsupported layout
        if (mode == 5) raw[0] = 4; // invalid expert
        if (mode == 6) local.nb[1] = SIZE_MAX; // overflow
        auto weight = *w;
        if (mode == 7) weight.nb[2] += 1; // reject noncontiguous weights
        {
            ggml_moe_gpu_trace trace(f, mode == 0 ? 1 : 100000, mode == 1 ? 1 : 64*1024*1024, mode == 2 ? 0 : 100000);
            trace.set_scope({7, 0, 2, 1, 1});
            trace.record(&weight, nullptr, &local, mode == 8 ? nullptr : read_ids, &reads);
            trace.record(&weight, nullptr, &local, read_ids, &reads);
            CHECK(reads == ((mode == 0 || mode == 5) ? 1 : 0));
        }
        auto s = contents(f); CHECK(s.find("dropped=0 error=0") == std::string::npos);
        std::fclose(f); raw[0] = 3;
    }
    // Entry caps also bound all-pruned selections, which emit no records.
    {
        FILE * f = std::tmpfile(); CHECK(f); int reads = 0;
        int32_t pruned[6] = {-1,-1,-1,-1,-1,-1}; auto local = *ids; local.data = pruned; local.nb[1] = 12;
        {
            ggml_moe_gpu_trace trace(f, 100000, 64*1024*1024, 1); trace.set_scope({8, 0, 2, 1, 1});
            trace.record(w, nullptr, &local, read_ids, &reads); trace.record(w, nullptr, &local, read_ids, &reads);
            CHECK(reads == 1);
        }
        CHECK(contents(f).find("# end written=0 dropped=1 error=1") != std::string::npos); std::fclose(f);
    }
    // Exact read budget is admitted; the following dispatch cannot read again.
    {
        FILE * f = std::tmpfile(); CHECK(f); int reads = 0;
        {
            ggml_moe_gpu_trace trace(f, 100000, 24); trace.set_scope({8, 0, 2, 1, 1});
            trace.record(w, nullptr, ids, read_ids, &reads);
            trace.record(w, nullptr, ids, read_ids, &reads); CHECK(reads == 1);
        }
        CHECK(contents(f).find("# end written=3 dropped=1 error=1") != std::string::npos); std::fclose(f);
    }
    ggml_free(ctx);
    std::puts("GPU trace CPU tests: selection, padded IDs, fused weights, dedup, scope, caps OK");
}
