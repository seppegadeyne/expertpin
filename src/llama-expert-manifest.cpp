#include "llama-expert-manifest.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace {

using json = nlohmann::json;

[[noreturn]] void manifest_error(const std::string & source, const std::string & message) {
    throw std::runtime_error("expert manifest '" + source + "': " + message);
}

uint32_t parse_layer_key(const std::string & key, const std::string & source) {
    uint32_t layer = 0;
    if (key.empty() || key.front() == '-' || key.front() == '+') {
        manifest_error(source, "invalid layer key '" + key + "' (expected a non-negative integer)");
    }

    const char * begin = key.data();
    const char * end = begin + key.size();
    const auto result = std::from_chars(begin, end, layer);
    if (result.ec != std::errc() || result.ptr != end) {
        manifest_error(source, "invalid layer key '" + key + "' (expected a non-negative integer)");
    }
    return layer;
}

uint32_t parse_expert_id(const json & value, uint32_t layer, size_t index, const std::string & source) {
    const std::string location = "expert id at layer " + std::to_string(layer) +
            ", index " + std::to_string(index);

    if (!value.is_number_integer()) {
        manifest_error(source, location + " must be an integer");
    }

    if (value.is_number_unsigned()) {
        const uint64_t id = value.get<uint64_t>();
        if (id > std::numeric_limits<uint32_t>::max()) {
            manifest_error(source, location + " is too large");
        }
        return static_cast<uint32_t>(id);
    }

    const int64_t id = value.get<int64_t>();
    if (id < 0) {
        manifest_error(source, location + " must be >= 0");
    }
    if (static_cast<uint64_t>(id) > std::numeric_limits<uint32_t>::max()) {
        manifest_error(source, location + " is too large");
    }
    return static_cast<uint32_t>(id);
}

} // namespace

llama_expert_manifest llama_expert_manifest_parse(const std::string & json_text, const std::string & source) {
    json root;
    try {
        root = json::parse(json_text);
    } catch (const json::parse_error & error) {
        manifest_error(source, "invalid JSON: " + std::string(error.what()));
    }

    if (!root.is_object()) {
        manifest_error(source, "root must be a JSON object");
    }

    static const char * wrappers[] = {"layers", "pinned_experts", "kept_experts"};
    const json * layer_map = &root;
    const char * selected_wrapper = nullptr;
    for (const char * wrapper : wrappers) {
        if (!root.contains(wrapper)) {
            continue;
        }
        if (selected_wrapper != nullptr) {
            manifest_error(source, "multiple wrapper keys are not allowed");
        }
        selected_wrapper = wrapper;
        layer_map = &root.at(wrapper);
    }

    if (!layer_map->is_object()) {
        manifest_error(source, std::string(selected_wrapper ? selected_wrapper : "root") +
                " must contain a layer-to-expert object");
    }
    if (layer_map->empty()) {
        manifest_error(source, "layer map must not be empty");
    }

    llama_expert_manifest manifest;
    manifest.source = source;

    for (auto it = layer_map->begin(); it != layer_map->end(); ++it) {
        const uint32_t layer = parse_layer_key(it.key(), source);
        const json & ids_json = it.value();
        if (!ids_json.is_array()) {
            manifest_error(source, "layer " + std::to_string(layer) + " must contain an expert-id array");
        }
        if (ids_json.empty()) {
            manifest_error(source, "expert-id list for layer " + std::to_string(layer) + " must not be empty");
        }

        std::vector<uint32_t> ids;
        ids.reserve(ids_json.size());
        std::unordered_set<uint32_t> seen;
        for (size_t index = 0; index < ids_json.size(); ++index) {
            const uint32_t id = parse_expert_id(ids_json.at(index), layer, index, source);
            if (!seen.insert(id).second) {
                manifest_error(source, "duplicate expert id " + std::to_string(id) +
                        " in layer " + std::to_string(layer));
            }
            ids.push_back(id);
        }

        if (!manifest.layers.emplace(layer, std::move(ids)).second) {
            manifest_error(source, "duplicate layer " + std::to_string(layer));
        }
    }

    return manifest;
}

llama_expert_manifest llama_expert_manifest_load(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        manifest_error(path, "cannot open file");
    }

    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        manifest_error(path, "failed while reading file");
    }
    return llama_expert_manifest_parse(contents.str(), path);
}

void llama_expert_manifest_validate(
        const llama_expert_manifest & manifest,
        uint32_t n_layers,
        uint32_t n_experts,
        int32_t resident_experts) {
    if (resident_experts < 0) {
        manifest_error(manifest.source, "resident expert count must be >= 0");
    }

    for (const auto & entry : manifest.layers) {
        const uint32_t layer = entry.first;
        const auto & ids = entry.second;
        if (layer >= n_layers) {
            manifest_error(manifest.source, "layer " + std::to_string(layer) +
                    " is outside model layer range [0, " + std::to_string(n_layers) + ")");
        }
        if (resident_experts > static_cast<int32_t>(ids.size())) {
            manifest_error(manifest.source, "resident expert count " + std::to_string(resident_experts) +
                    " exceeds layer " + std::to_string(layer) + " list length " + std::to_string(ids.size()));
        }
        for (const uint32_t id : ids) {
            if (id >= n_experts) {
                manifest_error(manifest.source, "expert id " + std::to_string(id) +
                        " in layer " + std::to_string(layer) +
                        " is outside model expert range [0, " + std::to_string(n_experts) + ")");
            }
        }
    }
}

std::vector<uint32_t> llama_expert_manifest_select(
        const llama_expert_manifest & manifest,
        uint32_t layer,
        size_t k) {
    const auto it = manifest.layers.find(layer);
    if (it == manifest.layers.end()) {
        manifest_error(manifest.source, "layer " + std::to_string(layer) + " is not present");
    }

    const size_t count = std::min(k, it->second.size());
    return std::vector<uint32_t>(it->second.begin(), it->second.begin() + count);
}
