// Bounded synthetic CUDA matvec; no model files, routing, or CPU backend fallback.
// Input generation and scalar reference follow bench-expert-cpu.cpp.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#if !defined(GGML_USE_CUDA) || defined(GGML_USE_HIPBLAS) || defined(GGML_USE_MUSA)
#error "bench-expert-gpu requires the CUDA backend"
#endif

using json = nlohmann::ordered_json;
static_assert(GGML_TYPE_IQ4_NL == 20 && GGML_TYPE_IQ3_S == 21 && GGML_TYPE_IQ2_S == 22,
              "local inventory quant IDs changed");

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static float sample(uint32_t & state) {
    state = 1664525u * state + 1013904223u;
    return (static_cast<int>(state >> 16) - 32768) / 32768.0f;
}

static json measure(ggml_backend_t backend, ggml_type type, int columns, int rows,
                    size_t expected_bytes, int repetitions) {
    // Metadata only on the host; weights, input and output live in a CUDA buffer.
    const size_t graph_size = 8;
    const size_t context_bytes = 3 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({context_bytes, nullptr, true}), ggml_free);
    require(bool(ctx), "context allocation failed");
    auto * w = ggml_new_tensor_2d(ctx.get(), type, columns, rows);
    auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, columns, 1);
    auto * y = ggml_mul_mat(ctx.get(), w, x);
    auto * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, y);
    require(w->nb[2] == expected_bytes && ggml_nbytes(w) == expected_bytes,
            "expert layout does not match local GGUF inventory");
    require(ggml_backend_supports_op(backend, y), "CUDA backend does not support this quantized MUL_MAT");

    auto buft = ggml_backend_get_default_buffer_type(backend);
    require(!ggml_backend_buft_is_host(buft), "expected device buffer, not host memory");
    const size_t allocation_bound = ggml_backend_buft_get_alloc_size(buft, w)
        + ggml_backend_buft_get_alloc_size(buft, x) + ggml_backend_buft_get_alloc_size(buft, y)
        + 3 * ggml_backend_buft_get_alignment(buft);
    require(allocation_bound <= 2 * 1024 * 1024, "tensor buffer exceeds 2 MiB bound");
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "CUDA tensor allocation failed");
    require(!ggml_backend_buffer_is_host(buffer.get()), "tensors were not allocated on CUDA");
    ggml_backend_buffer_clear(buffer.get(), 0); // includes quantized row padding

    uint32_t rng = 17;
    std::vector<uint8_t> quantized(expected_bytes);
    {
        // One slice at a time, not a 512-expert bank. Source released before timing.
        std::vector<float> source(static_cast<size_t>(columns) * rows);
        for (auto & v : source) v = sample(rng);
        std::vector<float> importance(columns, 1.0f);
        const size_t written = ggml_quantize_chunk(type, source.data(), quantized.data(), 0,
                                                   rows, columns, importance.data(), nullptr);
        require(written == expected_bytes, "quantizer byte count mismatch");
        require(ggml_validate_row_data(type, quantized.data(), written), "invalid quantized payload");
    }
    std::vector<float> input(columns), output(rows), row(columns);
    for (auto & v : input) v = sample(rng);
    ggml_backend_tensor_set(w, quantized.data(), 0, expected_bytes);
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));

    // Independent scalar CPU reference over the actual quantized payload, not the
    // original F32 weights. No CPU graph backend or scheduler can execute the op.
    const auto traits = ggml_internal_get_type_traits(type);
    require(traits.to_float != nullptr, "missing dequantizer");
    std::vector<double> reference(rows);
    for (int r = 0; r < rows; ++r) {
        traits.to_float(quantized.data() + r * w->nb[1], row.data(), columns);
        double sum = 0;
        for (int c = 0; c < columns; ++c) sum += static_cast<double>(row[c]) * input[c];
        require(std::isfinite(sum), "non-finite CPU reference");
        reference[r] = sum;
    }
    double max_relative_l2_error = 0, max_absolute_error = 0;
    const auto validate = [&] {
        ggml_backend_tensor_get(y, output.data(), 0, ggml_nbytes(y)); // untimed D2H
        double squared_error = 0, squared_reference = 0;
        for (int r = 0; r < rows; ++r) {
            require(std::isfinite(output[r]), "non-finite CUDA output");
            const double delta = output[r] - reference[r];
            squared_error += delta * delta;
            squared_reference += reference[r] * reference[r];
            max_absolute_error = std::max(max_absolute_error, std::abs(delta));
        }
        require(squared_reference > 0, "degenerate reference");
        const double error = std::sqrt(squared_error / squared_reference);
        require(std::isfinite(error) && error <= 0.03, "CUDA/reference relative L2 error > 3%");
        max_relative_l2_error = std::max(max_relative_l2_error, error);
    };
    const auto compute = [&] {
        const auto status = ggml_backend_graph_compute_async(backend, graph);
        ggml_backend_synchronize(backend); // stream completion; CUDA errors are fatal in ggml
        require(status == GGML_STATUS_SUCCESS, "CUDA graph computation failed");
    };
    compute();
    validate();
    for (int i = 0; i < 3; ++i) compute();
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (int i = 0; i < repetitions; ++i) {
        ggml_backend_synchronize(backend); // nothing from setup/previous sample enters timing
        const auto start = std::chrono::steady_clock::now();
        compute(); // includes host dispatch, activation conversion and completion wait
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        require(std::isfinite(us) && us > 0, "invalid measured duration");
        samples.push_back(us);
    }
    validate();
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    return {{"ggml_type", static_cast<int>(type)}, {"quant", ggml_type_name(type)},
            {"input_columns", columns}, {"output_rows", rows}, {"slice_bytes", expected_bytes},
            {"device_tensor_buffer_bytes", ggml_backend_buffer_get_size(buffer.get())},
            {"experts_in_bank", 1}, {"tokens", 1}, {"correctness_passed", true},
            {"max_relative_l2_error", max_relative_l2_error}, {"max_absolute_error", max_absolute_error},
            {"samples_us", samples}, {"median_us", sorted[sorted.size() / 2]},
            {"minimum_us", sorted.front()}, {"maximum_us", sorted.back()}};
}

int main(int argc, char ** argv) {
    int repetitions;
    try {
        require(argc == 2, "expected REPETITIONS");
        // Exact finite choices reject signs, whitespace, overflow and long inputs.
        require(std::strcmp(argv[1], "11") == 0 || std::strcmp(argv[1], "101") == 0,
                "REPETITIONS must be 11 or 101");
        repetitions = std::strcmp(argv[1], "11") == 0 ? 11 : 101;
        const char * no_pinned = std::getenv("GGML_CUDA_NO_PINNED");
        require(no_pinned && std::strcmp(no_pinned, "1") == 0,
                "parent must set GGML_CUDA_NO_PINNED=1 before launch");
    } catch (const std::exception & e) {
        fprintf(stderr, "Usage: GGML_CUDA_NO_PINNED=1 bench-expert-gpu REPETITIONS(11|101): %s\n", e.what());
        return 2; // reject before any explicit CUDA API call
    }
    try {
        // Restrict visibility externally on multi-GPU hosts; no peer-device workload.
        require(ggml_backend_cuda_get_device_count() == 1,
                "exactly one CUDA device must be visible (use CUDA_VISIBLE_DEVICES)");
        // Fixed uncaptured dispatch: graphs=0 is ignored only in builds without graph support.
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_cuda_init(0, "graphs=0", nullptr), ggml_backend_free);
        require(bool(backend) && ggml_backend_is_cuda(backend.get()), "CUDA backend initialization failed");
        json result = {{"schema_version", 1}, {"backend", "cuda"},
            {"backend_name", ggml_backend_name(backend.get())}, {"device", 0},
            {"operation", "MUL_MAT"}, {"payload", "synthetic_quantized"}, {"seed", 17},
            {"repetitions", repetitions}, {"warmups", 3}, {"cuda_graphs", false},
            {"GGML_CUDA_NO_PINNED", "1"}, {"cache_condition", "repeated_hot_single_expert"},
            {"timing_scope", "host_wall_cuda_graph_dispatch_and_stream_synchronize"},
            {"reference", "scalar_f64_dot_dequantized_weights_f32_input"},
            {"relative_l2_error_limit", 0.03}, {"validation", "before_warmups_and_after_timing"},
            {"model_tokens_per_second", nullptr}, {"physical_vram_gib_s", nullptr},
            {"cases", json::array()}};
        result["cases"].push_back(measure(backend.get(), GGML_TYPE_IQ4_NL, 640, 2560, 921600, repetitions));
        result["cases"].push_back(measure(backend.get(), GGML_TYPE_IQ3_S, 2560, 640, 704000, repetitions));
        result["cases"].push_back(measure(backend.get(), GGML_TYPE_IQ2_S, 2560, 640, 524800, repetitions));
        backend.reset();
        ggml_quantize_free();
        puts(result.dump(2).c_str()); // no partial result on calibration failure
    } catch (const std::exception & e) {
        ggml_quantize_free();
        fprintf(stderr, "GPU calibration failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
