#include "llama-model.h"
#include "llama-spec-features.h"
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

static void init_draft(llama_model & draft) {
    draft.arch = LLM_ARCH_QWEN4EXP;
    draft.hparams.n_embd = 4;
    draft.hparams.n_vocab = 8;
}

static void same_bytes(const ggml_tensor * a, const ggml_tensor * b) {
    GGML_ASSERT(a->type == b->type && ggml_are_same_shape(a, b));
    unsigned char x[4096], y[4096];
    const size_t size = ggml_nbytes(a);
    for (size_t offset = 0; offset < size; offset += sizeof(x)) {
        const size_t count = std::min(sizeof(x), size - offset);
        ggml_backend_tensor_get(a, x, offset, count);
        ggml_backend_tensor_get(b, y, offset, count);
        GGML_ASSERT(std::memcmp(x, y, count) == 0);
    }
}

static void test_self_contained() {
    llama_model target = {};
    target.arch = LLM_ARCH_LLAMA;
    ggml_tensor embedding = {}, output = {};
    GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(nullptr, &target));
    GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(nullptr, nullptr));
    for (int own_io = 0; own_io < 4; ++own_io) {
        llama_model draft = {};
        init_draft(draft);
        draft.tok_embd = own_io & 1 ? &embedding : nullptr;
        draft.output = own_io & 2 ? &output : nullptr;
        GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(&draft, nullptr));
        GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(&draft, &target) == (own_io == 3));
        GGML_ASSERT(draft.tok_embd == (own_io & 1 ? &embedding : nullptr));
        GGML_ASSERT(draft.output == (own_io & 2 ? &output : nullptr));
        GGML_ASSERT(draft.bufs.empty());
    }
    llama_model other = {};
    other.arch = LLM_ARCH_LLAMA;
    GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(&other, &target));
    GGML_ASSERT(!other.tok_embd && !other.output && other.bufs.empty());
}

static void test_shapes_and_partial_io() {
    llama_model target = {};
    target.arch = LLM_ARCH_QWEN4EXP;
    ggml_tensor embedding = {}, output = {}, own = {};
    embedding.ne[0] = output.ne[0] = 4;
    embedding.ne[1] = output.ne[1] = 8;
    target.tok_embd = &embedding;
    target.output = &output;
    for (int field = 0; field < 2; ++field) {
        ggml_tensor * tensor = field == 0 ? &embedding : &output;
        for (int invalid = 0; invalid < 3; ++invalid) {
            llama_model draft = {};
            init_draft(draft);
            if (invalid < 2) {
                ++tensor->ne[invalid];
            } else if (field == 0) {
                target.tok_embd = nullptr;
            } else {
                target.output = nullptr;
            }
            GGML_ASSERT(!llama_model_share_qwen4exp_mtp_tensors(&draft, &target));
            GGML_ASSERT((field == 0 ? draft.tok_embd : draft.output) == nullptr);
            GGML_ASSERT(draft.bufs.empty());
            if (invalid < 2) {
                --tensor->ne[invalid];
            }
            target.tok_embd = &embedding;
            target.output = &output;
        }
    }
    for (int own_io = 0; own_io < 3; ++own_io) {
        llama_model draft = {};
        init_draft(draft);
        draft.tok_embd = own_io == 1 ? &own : nullptr;
        draft.output = own_io == 2 ? &own : nullptr;
        draft.buft_input = llama_model::layer_buft(ggml_backend_cpu_buffer_type());
        GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(&draft, &target));
        GGML_ASSERT(draft.tok_embd == (own_io == 1 ? &own : &embedding));
        GGML_ASSERT(draft.output == (own_io == 2 ? &own : &output));
        GGML_ASSERT(draft.bufs.empty());
    }
}

static int allocations = 0;

static ggml_backend_buffer_t alternate_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    ++allocations;
    auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    GGML_ASSERT(buffer != nullptr);
    buffer->buft = buft;
    return buffer;
}

static bool nonhost(ggml_backend_buffer_type_t) {
    return false;
}

static void test_buffers(bool host_input, bool clone_output) {
    auto alternate = *ggml_backend_cpu_buffer_type();
    alternate.iface.alloc_buffer = alternate_alloc;
    // Simulate non-host placement with CPU storage; no GPU backend is needed.
    auto source_type = *ggml_backend_cpu_buffer_type();
    if (!host_input) {
        source_type.iface.is_host = nonhost;
    }
    llama_model target = {}, draft = {};
    init_draft(target);
    init_draft(draft);
    ggml_init_params params = {2 * ggml_tensor_overhead(), nullptr, true};
    auto ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);
    target.ctxs.push_back(ctx);
    target.tok_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    target.output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    auto buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
    GGML_ASSERT(buffer != nullptr);
    target.bufs.push_back(buffer);
    buffer->buft = &source_type;
    float data[32];
    for (int i = 0; i < 32; ++i) {
        data[i] = float(i + 1);
    }
    ggml_backend_tensor_set(target.tok_embd, data, 0, sizeof(data));
    ggml_backend_tensor_set(target.output, data, 0, sizeof(data));
    draft.buft_input = llama_model::layer_buft(&alternate);
    draft.buft_output = llama_model::layer_buft(clone_output ? &alternate : &source_type);
    allocations = 0;
    GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(&draft, &target));
    const int expected_allocations = (host_input ? 0 : 1) + (clone_output ? 1 : 0);
    GGML_ASSERT(allocations == expected_allocations);
    GGML_ASSERT(draft.bufs.size() == size_t(expected_allocations));
    GGML_ASSERT(draft.tok_embd == (host_input ? target.tok_embd : draft.qwen4exp_tok_embd_ptr.get()));
    GGML_ASSERT(bool(draft.qwen4exp_tok_embd_ptr) == !host_input);
    GGML_ASSERT(draft.output == (clone_output ? draft.qwen4exp_output_ptr.get() : target.output));
    GGML_ASSERT(bool(draft.qwen4exp_output_ptr) == clone_output);
    same_bytes(draft.tok_embd, target.tok_embd);
    same_bytes(draft.output, target.output);
    GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(&draft, &target));
    GGML_ASSERT(allocations == expected_allocations);
}

static void test_tied_companion(const char * path) {
    gguf_init_params metadata_params = {true, nullptr};
    auto metadata = gguf_init_from_file(path, metadata_params);
    GGML_ASSERT(metadata != nullptr);
    GGML_ASSERT(gguf_find_tensor(metadata, "token_embd.weight") >= 0);
    GGML_ASSERT(gguf_find_tensor(metadata, "output.weight") < 0);
    gguf_free(metadata);

    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.mtp = true;
    params.use_mmap = true;
    params.use_mlock = false;
    llama_model * draft = llama_model_load_from_file(path, params);
    GGML_ASSERT(draft != nullptr && draft->arch == LLM_ARCH_QWEN4EXP);
    GGML_ASSERT(llama_model_mtp_package(draft) == LLAMA_MTP_PACKAGE_COMPANION);
    GGML_ASSERT(draft->tok_embd != nullptr && draft->output != nullptr);
    same_bytes(draft->tok_embd, draft->output);
    llama_model target = {};
    target.arch = LLM_ARCH_QWEN4EXP;
    ggml_tensor foreign_output = {};
    target.output = &foreign_output;
    ggml_tensor * own_output = draft->output;
    GGML_ASSERT(llama_model_share_qwen4exp_mtp_tensors(draft, &target));
    GGML_ASSERT(draft->output == own_output);
    llama_free_model(draft);
}

int main(int argc, char ** argv) {
    if (argc == 1) {
        test_self_contained();
        test_shapes_and_partial_io();
        test_buffers(true, false);
        test_buffers(true, true);
        test_buffers(false, true);
    } else if (argc == 2 && std::strcmp(argv[1], "--self-contained") == 0) {
        test_self_contained();
    } else if (argc == 2 && std::strcmp(argv[1], "--host-input") == 0) {
        test_buffers(true, false);
    } else if (argc == 3 && std::strcmp(argv[1], "--tied-companion") == 0) {
        llama_backend_init();
        test_tied_companion(argv[2]);
        llama_backend_free();
    } else {
        std::fprintf(stderr, "Usage: %s [--self-contained | --host-input | --tied-companion path.gguf]\n"
                "Default CTest is model-free. --tied-companion runs the optional CPU loader test\n"
                "with an external Qwen4Exp companion containing token_embd.weight but no output.weight.\n",
                argv[0]);
        return 1;
    }
    std::puts("Qwen4Exp MTP IO tests passed");
    return 0;
}
