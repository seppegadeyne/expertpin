#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

struct llama_expert_manifest {
    std::map<uint32_t, std::vector<uint32_t>> layers;
    std::string source;
};

// Parse a bare layer map or an object wrapped in one of: layers,
// pinned_experts, kept_experts. Expert order is preserved verbatim.
llama_expert_manifest llama_expert_manifest_parse(
        const std::string & json_text,
        const std::string & source = "<memory>");

llama_expert_manifest llama_expert_manifest_load(const std::string & path);

// Validate manifest coordinates against a loaded model. This intentionally
// validates model-provided dimensions rather than assuming a fixed layer or
// expert count such as 48 x 512.
void llama_expert_manifest_validate(
        const llama_expert_manifest & manifest,
        uint32_t n_layers,
        uint32_t n_experts,
        int32_t resident_experts = 0);

// Return the first K expert ids exactly as ordered by the manifest.
std::vector<uint32_t> llama_expert_manifest_select(
        const llama_expert_manifest & manifest,
        uint32_t layer,
        size_t k);
