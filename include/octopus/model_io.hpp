#pragma once
#include "octopus/transformer.hpp"
#include <filesystem>

namespace octopus {
// Native model files contain graph nodes, operator attributes, and float32 weights.
void save_model(const ModelDefinition &model, const std::filesystem::path &path);
ModelDefinition load_json_model(const std::filesystem::path &path);
ModelDefinition load_onnx_model(const std::filesystem::path &path);
ModelDefinition load_model(const std::filesystem::path &path);
} // namespace octopus
