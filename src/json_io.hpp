#pragma once
#include "octopus/inference.hpp"
#include <nlohmann/json.hpp>

namespace octopus::detail {
using Json = nlohmann::json;
inline TensorF32 decode_tensor(const Json &object, const char *data_key, const char *shape_key) {
    const auto &dimensions = object.at(shape_key);
    const auto &values = object.at(data_key);
    if (!dimensions.is_array() || dimensions.size() > 16 || !values.is_array())
        throw std::invalid_argument("Expected a flat data array and a shape with at most 16 dimensions");
    Shape shape;
    for (const auto &dimension : dimensions) {
        if (!dimension.is_number_integer() || dimension.get<std::int64_t>() <= 0)
            throw std::invalid_argument("Shape dimensions must be positive integers");
        shape.push_back(dimension.get<std::size_t>());
    }
    if (element_count(shape) != values.size())
        throw std::invalid_argument("Input data length does not match its shape");
    std::vector<float> data;
    data.reserve(values.size());
    for (const auto &value : values) {
        if (!value.is_number())
            throw std::invalid_argument("Input values must be numbers");
        auto x = value.get<float>();
        if (!std::isfinite(x))
            throw std::invalid_argument("Input values must be finite float32 numbers");
        data.push_back(x);
    }
    return TensorF32(std::move(data), shape);
}
inline TensorMap decode_inputs(const Json &request) {
    if (!request.is_object())
        throw std::invalid_argument("Inference request must be a JSON object");
    TensorMap inputs;
    if (request.contains("inputs")) {
        if (request.contains("input_data") || request.contains("input_shape") ||
            request.contains("input_name"))
            throw std::invalid_argument("Use either inputs or the single-input fields");
        const auto &encoded = request.at("inputs");
        if (!encoded.is_object() || encoded.empty())
            throw std::invalid_argument("inputs must be a nonempty object");
        for (const auto &[name, tensor] : encoded.items())
            inputs.emplace(name, decode_tensor(tensor, "data", "shape"));
    } else
        inputs.emplace(request.value("input_name", std::string("input")),
                       decode_tensor(request, "input_data", "input_shape"));
    return inputs;
}
inline Json encode_result(const InferenceResult &result, const std::string &output_name) {
    auto found = result.outputs.find(output_name);
    if (found == result.outputs.end())
        throw std::invalid_argument("Unknown output: " + output_name);
    Json outputs = Json::object();
    for (const auto &[name, tensor] : result.outputs) {
        for (auto value : tensor.values())
            if (!std::isfinite(value))
                throw std::runtime_error("Inference produced non-finite values");
        outputs[name] = {{"data", tensor.values()}, {"shape", tensor.shape()}};
    }
    return {{"output_data", found->second.values()},
            {"output_shape", found->second.shape()},
            {"inference_time_ms", result.stats.total_time_ms},
            {"backend", result.stats.backend},
            {"outputs", outputs}};
}
inline Json parse_request(const std::string &body) {
    if (body.size() > 1024 * 1024)
        throw std::invalid_argument("Inference request exceeds 1 MiB");
    return Json::parse(body, [](int depth, Json::parse_event_t, Json &) {
        if (depth > 32)
            throw std::invalid_argument("Inference JSON nesting exceeds 32 levels");
        return true;
    });
}
} // namespace octopus::detail
