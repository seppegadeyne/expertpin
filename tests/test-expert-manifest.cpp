#include "llama-expert-manifest.h"
#include "common.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef EXPERTPIN_TEST_FIXTURE_DIR
#error "EXPERTPIN_TEST_FIXTURE_DIR must point at tests/fixtures"
#endif

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Fn>
void require_error(Fn && fn, const std::string & expected) {
    try {
        fn();
    } catch (const std::exception & error) {
        const std::string message = error.what();
        require(message.find(expected) != std::string::npos,
                "expected error containing '" + expected + "', got '" + message + "'");
        return;
    }
    throw std::runtime_error("expected error containing '" + expected + "'");
}

void require_layer(const llama_expert_manifest & manifest, uint32_t layer, const std::vector<uint32_t> & expected) {
    const auto it = manifest.layers.find(layer);
    require(it != manifest.layers.end(), "missing layer " + std::to_string(layer));
    require(it->second == expected, "unexpected expert order for layer " + std::to_string(layer));
}

void require_layer_prefix(const llama_expert_manifest & manifest, uint32_t layer,
        const std::vector<uint32_t> & expected) {
    const auto it = manifest.layers.find(layer);
    require(it != manifest.layers.end(), "missing layer " + std::to_string(layer));
    require(it->second.size() >= expected.size(), "layer is shorter than expected prefix");
    require(std::equal(expected.begin(), expected.end(), it->second.begin()),
            "unexpected expert-order prefix for layer " + std::to_string(layer));
}

void test_bare_map_fixture() {
    const std::string fixture = std::string(EXPERTPIN_TEST_FIXTURE_DIR) + "/reap-saliency-compact.json";
    const auto manifest = llama_expert_manifest_load(fixture);

    require(manifest.layers.size() == 2, "bare map fixture should contain two layers");
    require(manifest.layers.at(0).size() == 512, "layer 0 should retain all 512 real expert IDs");
    require(manifest.layers.at(1).size() == 512, "layer 1 should retain all 512 real expert IDs");
    require_layer_prefix(manifest, 0, {447, 194, 390, 446, 87, 465, 230, 459});
    require_layer_prefix(manifest, 1, {254, 340, 282, 202, 470, 184, 223, 510});
}

void test_wrappers() {
    for (const std::string wrapper : {"layers", "pinned_experts", "kept_experts"}) {
        const auto manifest = llama_expert_manifest_parse(
                "{\"" + wrapper + "\":{\"3\":[9,4,1]},\"metadata\":{\"source\":\"test\"}}",
                "<" + wrapper + ">");
        require(manifest.layers.size() == 1, wrapper + " wrapper should contain one layer");
        require_layer(manifest, 3, {9, 4, 1});
    }
}

void test_invalid_manifests() {
    require_error([] {
        llama_expert_manifest_parse("{\"0\":[3,3]}", "<duplicate>");
    }, "duplicate expert id");

    require_error([] {
        llama_expert_manifest_parse("{\"0\":[]}", "<empty>");
    }, "must not be empty");

    require_error([] {
        llama_expert_manifest_parse("{not-json", "<broken>");
    }, "invalid JSON");

    require_error([] {
        llama_expert_manifest_parse("{\"0\":[-1]}", "<negative>");
    }, "must be >= 0");

    require_error([] {
        llama_expert_manifest_parse("{\"0\":[1.5]}", "<fractional>");
    }, "must be an integer");

    require_error([] {
        llama_expert_manifest_parse("{\"-1\":[0]}", "<negative-layer>");
    }, "invalid layer key");

    require_error([] {
        llama_expert_manifest_parse("{\"layers\":{\"0\":[1]},\"kept_experts\":{\"0\":[1]}}", "<ambiguous>");
    }, "multiple wrapper keys");

    require_error([] {
        llama_expert_manifest_load("/definitely/not/a/manifest.json");
    }, "cannot open");
}

void test_model_bounds() {
    const auto manifest = llama_expert_manifest_parse("{\"0\":[4]}", "<out-of-range>");
    require_error([&] {
        llama_expert_manifest_validate(manifest, 1, 4, 1);
    }, "outside model expert range");

    const auto bad_layer = llama_expert_manifest_parse("{\"2\":[0]}", "<bad-layer>");
    require_error([&] {
        llama_expert_manifest_validate(bad_layer, 2, 4, 1);
    }, "outside model layer range");

    const auto too_short = llama_expert_manifest_parse("{\"0\":[3,2]}", "<too-short>");
    require_error([&] {
        llama_expert_manifest_validate(too_short, 1, 4, 3);
    }, "resident expert count");
}

void test_top_k_prefix() {
    const std::string saliency_path = std::string(EXPERTPIN_TEST_FIXTURE_DIR) + "/reap-saliency-compact.json";
    const std::string kept_path = std::string(EXPERTPIN_TEST_FIXTURE_DIR) + "/reap-kept-saliency-ordered-compact.json";
    const auto saliency = llama_expert_manifest_load(saliency_path);
    const auto kept = llama_expert_manifest_load(kept_path);

    for (const auto & entry : kept.layers) {
        require(entry.second.size() == 288, "kept fixture must contain K=288 experts per layer");
        const auto selected = llama_expert_manifest_select(saliency, entry.first, 288);
        require(selected == entry.second,
                "top-K selection does not preserve saliency order for layer " + std::to_string(entry.first));
    }

    require(llama_expert_manifest_select(saliency, 0, 0).empty(), "K=0 must select no experts");
}

void test_cli_contract() {
    const std::string fixture = std::string(EXPERTPIN_TEST_FIXTURE_DIR) + "/reap-saliency-compact.json";
    std::vector<std::string> valid_args = {
        "test-expert-manifest",
        "--expert-manifest", fixture,
        "--resident-experts", "3",
        "--prefetch-experts",
    };
    std::vector<char *> valid_argv;
    for (auto & arg : valid_args) {
        valid_argv.push_back(arg.data());
    }

    gpt_params params;
    require(gpt_params_parse_ex(static_cast<int>(valid_argv.size()), valid_argv.data(), params),
            "valid expert residency CLI arguments should parse");
    require(params.expert_manifest == fixture, "--expert-manifest path was not preserved");
    require(params.resident_experts == 3, "--resident-experts value was not preserved");

    require_error([] {
        std::vector<std::string> args = {"test-expert-manifest", "--resident-experts", "1"};
        std::vector<char *> argv;
        for (auto & arg : args) argv.push_back(arg.data());
        gpt_params invalid;
        gpt_params_parse_ex(static_cast<int>(argv.size()), argv.data(), invalid);
    }, "requires --expert-manifest");

    require_error([] {
        std::vector<std::string> args = {"test-expert-manifest", "--resident-experts", "-1"};
        std::vector<char *> argv;
        for (auto & arg : args) argv.push_back(arg.data());
        gpt_params invalid;
        gpt_params_parse_ex(static_cast<int>(argv.size()), argv.data(), invalid);
    }, "must be >= 0");

    require_error([&] {
        std::vector<std::string> args = {
            "test-expert-manifest", "--expert-manifest", fixture, "--resident-experts", "1",
        };
        std::vector<char *> argv;
        for (auto & arg : args) argv.push_back(arg.data());
        gpt_params invalid;
        gpt_params_parse_ex(static_cast<int>(argv.size()), argv.data(), invalid);
    }, "requires --prefetch-experts");

    std::vector<std::string> zero_args = {"test-expert-manifest", "--resident-experts", "0"};
    std::vector<char *> zero_argv;
    for (auto & arg : zero_args) zero_argv.push_back(arg.data());
    gpt_params zero;
    require(gpt_params_parse_ex(static_cast<int>(zero_argv.size()), zero_argv.data(), zero),
            "K=0 must preserve the manifest-free default behavior");
}

void verify_real_prefix(const std::string & saliency_path, const std::string & kept_path, size_t k, bool compare_sets) {
    const auto saliency = llama_expert_manifest_load(saliency_path);
    const auto kept = llama_expert_manifest_load(kept_path);
    require(saliency.layers.size() == kept.layers.size(), "manifest layer counts differ");

    for (const auto & entry : kept.layers) {
        const auto selected = llama_expert_manifest_select(saliency, entry.first, k);
        if (compare_sets) {
            require(std::set<uint32_t>(selected.begin(), selected.end()) ==
                            std::set<uint32_t>(entry.second.begin(), entry.second.end()),
                    "top-K set differs for layer " + std::to_string(entry.first));
        } else {
            require(selected == entry.second,
                    "top-K order differs for layer " + std::to_string(entry.first));
        }
    }

    std::cout << (compare_sets ? "real REAP top-K set verified" : "real REAP top-K order verified")
              << ": layers=" << kept.layers.size() << " K=" << k << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 5 && (std::string(argv[1]) == "--verify-prefix" || std::string(argv[1]) == "--verify-set")) {
            const bool compare_sets = std::string(argv[1]) == "--verify-set";
            const size_t k = std::stoul(argv[4]);
            verify_real_prefix(argv[2], argv[3], k, compare_sets);
            return 0;
        }

        require(argc == 1, "usage: test-expert-manifest [--verify-prefix|--verify-set SALIENCY KEPT K]");
        test_bare_map_fixture();
        test_wrappers();
        test_invalid_manifests();
        test_model_bounds();
        test_top_k_prefix();
        test_cli_contract();
        std::cout << "expert manifest tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-expert-manifest: " << error.what() << '\n';
        return 1;
    }
}
