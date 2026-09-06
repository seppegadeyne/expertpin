// Bounded CPU-only MUL_MAT_ID microbenchmark; never opens a model or GPU backend.
#include "ggml.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static int bounded_integer(const char * text, int maximum) {
    int value = 0;
    require(text && *text, "empty integer");
    for (const char * p = text; *p; ++p) {
        require(*p >= '0' && *p <= '9', "expected decimal integer");
        const int digit = *p - '0';
        require(value <= (maximum - digit) / 10, "integer exceeds limit");
        value = 10 * value + digit;
    }
    require(value > 0 && value <= maximum, "integer outside limits");
    return value;
}

static float sample(uint32_t & state) {
    state = 1664525u * state + 1013904223u;
    return (static_cast<int>(state >> 16) - 32768) / 32768.0f;
}

static json measure(ggml_type type, int columns, int rows, size_t expected_bytes,
                    int threads, int repetitions) {
    // Two distinct experts catch incorrect ID selection; timed requests select ID 1.
    const int experts = 2;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({16 * 1024 * 1024, nullptr, false}), ggml_free);
    require(bool(ctx), "context allocation failed");
    auto * w = ggml_new_tensor_3d(ctx.get(), type, columns, rows, experts);
    auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, columns, 1, 1);
    auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
    auto * y = ggml_mul_mat_id(ctx.get(), w, x, ids);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, y);
    require(w->nb[2] == expected_bytes && ggml_nbytes(w) == experts * expected_bytes,
            "expert layout does not match local GGUF inventory");

    uint32_t rng = 17;
    std::vector<float> source(static_cast<size_t>(columns) * rows * experts);
    for (auto & v : source) v = sample(rng);
    std::vector<float> importance(columns, 1.0f);
    const size_t written = ggml_quantize_chunk(type, source.data(), w->data, 0,
                                               rows * experts, columns, importance.data(), nullptr);
    require(written == ggml_nbytes(w), "quantizer byte count mismatch");
    require(ggml_validate_row_data(type, w->data, written), "invalid quantized payload");
    source.clear();
    source.shrink_to_fit();
    auto * input = static_cast<float *>(x->data);
    for (int i = 0; i < columns; ++i) input[i] = sample(rng);

    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    require(bool(backend), "CPU backend allocation failed");
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    ggml_backend_cpu_set_moe_expert_prefetch(backend.get(), false);
    ggml_backend_cpu_set_moe_expert_cache_sim(backend.get(), false);
    const auto compute = [&] {
        require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS,
                "CPU graph computation failed");
    };

    // Independent scalar dot products over dequantized weights, not original F32
    // weights: this checks execution/ID selection rather than quantization quality.
    const auto traits = ggml_internal_get_type_traits(type);
    require(traits.to_float != nullptr, "missing dequantizer");
    std::vector<float> row(columns);
    std::vector<double> reference(rows);
    double max_error = 0;
    const auto validate = [&] {
        const auto * output = static_cast<const float *>(y->data);
        double squared_error = 0, squared_reference = 0;
        for (int r = 0; r < rows; ++r) {
            require(std::isfinite(output[r]), "non-finite kernel output");
            const double delta = output[r] - reference[r];
            squared_error += delta * delta;
            squared_reference += reference[r] * reference[r];
        }
        require(squared_reference > 0, "degenerate reference");
        const double error = std::sqrt(squared_error / squared_reference);
        require(std::isfinite(error) && error <= 0.03, "kernel/reference relative L2 error > 3%");
        max_error = std::max(max_error, error);
    };
    for (int expert = 0; expert < experts; ++expert) {
        *static_cast<int32_t *>(ids->data) = expert;
        for (int r = 0; r < rows; ++r) {
            const auto * bytes = static_cast<const char *>(w->data) + expert * w->nb[2] + r * w->nb[1];
            traits.to_float(bytes, row.data(), columns);
            double sum = 0;
            for (int c = 0; c < columns; ++c) sum += static_cast<double>(row[c]) * input[c];
            reference[r] = sum;
        }
        compute();
        validate();
    }
    // Setup/initial workspace allocation, quantization and reference are untimed.
    // Internal per-call dispatch allocations, if any, remain inside timing.
    for (int i = 0; i < 3; ++i) compute();
    std::vector<double> samples;
    samples.reserve(repetitions);
    for (int i = 0; i < repetitions; ++i) {
        const auto start = std::chrono::steady_clock::now();
        compute(); // synchronous CPU graph dispatch, including its scheduling overhead
        const double us = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - start).count();
        require(std::isfinite(us) && us > 0, "invalid measured duration");
        samples.push_back(us);
    }
    validate();
    auto sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const size_t mid = sorted.size() / 2;
    const double median = sorted.size() % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    return {{"quant", ggml_type_name(type)}, {"input_columns", columns}, {"output_rows", rows},
            {"slice_bytes", expected_bytes}, {"bank_bytes", ggml_nbytes(w)},
            {"experts_in_bank", experts}, {"selected_expert", 1}, {"tokens", 1},
            {"validated_expert_ids", {0, 1}}, {"max_relative_l2_error", max_error},
            {"samples_us", samples}, {"median_us", median},
            {"minimum_us", sorted.front()}, {"maximum_us", sorted.back()}};
}

int main(int argc, char ** argv) {
    int threads, repetitions;
    try {
        require(argc == 3, "expected THREADS REPETITIONS");
        threads = bounded_integer(argv[1], 256);
        repetitions = bounded_integer(argv[2], 10000);
    } catch (const std::exception & e) {
        fprintf(stderr, "Usage: bench-expert-cpu THREADS(1..256) REPETITIONS(1..10000): %s\n", e.what());
        return 2;
    }
    try {
        json result = {{"schema_version", 1}, {"backend", "cpu"}, {"operation", "MUL_MAT_ID"},
            {"payload", "synthetic_quantized"}, {"seed", 17}, {"threads", threads}, {"warmups", 3},
            {"cache_condition", "repeated_hot_two_expert_bank"},
            {"timing_scope", "synchronous_cpu_graph_compute_including_dispatch"},
            {"model_tokens_per_second", nullptr}, {"physical_ram_gib_s", nullptr},
            {"cases", json::array()}};
        result["cases"].push_back(measure(GGML_TYPE_IQ2_S, 2560, 640, 524800, threads, repetitions));
        result["cases"].push_back(measure(GGML_TYPE_IQ3_S, 2560, 640, 704000, threads, repetitions));
        result["cases"].push_back(measure(GGML_TYPE_IQ4_NL, 640, 2560, 921600, threads, repetitions));
        ggml_quantize_free();
        puts(result.dump(2).c_str()); // no partial JSON on failure
    } catch (const std::exception & e) {
        ggml_quantize_free();
        fprintf(stderr, "CPU calibration failed: %s\n", e.what());
        return 1;
    }
    return 0;
}
