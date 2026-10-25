#include "octopus/model_io.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

namespace octopus {
namespace {
using Json = nlohmann::json;
constexpr std::uintmax_t max_model_bytes = 64 * 1024 * 1024;
Shape read_shape(const Json &values) {
    if (!values.is_array() || values.size() > 16)
        throw std::invalid_argument("Shape must be an array of at most 16 dimensions");
    Shape result;
    for (const auto &value : values) {
        if (!value.is_number_integer() || value.get<std::int64_t>() <= 0)
            throw std::invalid_argument("Shape dimensions must be positive integers");
        result.push_back(value.get<std::size_t>());
    }
    element_count(result);
    return result;
}
Json encode_attributes(const Attributes &a) {
    return {{"shape", a.shape},     {"target_shape", a.target_shape},
            {"axes", a.axes},       {"strides", a.strides},
            {"padding", a.padding}, {"axis", a.axis},
            {"epsilon", a.epsilon}, {"heads", a.heads},
            {"causal", a.causal},   {"alpha", a.alpha},
            {"beta", a.beta},       {"trans_a", a.trans_a},
            {"trans_b", a.trans_b}};
}
std::vector<std::int64_t> integer_array(const Json &values, const std::string &field) {
    if (!values.is_array() || values.size() > 16)
        throw std::invalid_argument("Invalid integer array: " + field);
    std::vector<std::int64_t> result;
    for (const auto &v : values) {
        if (!v.is_number_integer())
            throw std::invalid_argument("Expected an integer in " + field);
        result.push_back(v.get<std::int64_t>());
    }
    return result;
}
Attributes decode_attributes(const Json &object) {
    if (!object.is_object())
        throw std::invalid_argument("Node attributes must be an object");
    Attributes a;
    if (object.contains("shape"))
        a.shape = read_shape(object.at("shape"));
    if (object.contains("target_shape"))
        a.target_shape = integer_array(object.at("target_shape"), "target_shape");
    auto unsigned_array = [&](const char *key, Shape &target, bool positive) {
        if (!object.contains(key))
            return;
        target.clear();
        for (auto value : integer_array(object.at(key), key)) {
            if (value < 0 || (positive && value == 0))
                throw std::invalid_argument("Invalid dimension in " + std::string(key));
            target.push_back(static_cast<std::size_t>(value));
        }
    };
    unsigned_array("axes", a.axes, false);
    unsigned_array("strides", a.strides, true);
    unsigned_array("padding", a.padding, false);
    if (object.contains("axis")) {
        if (!object.at("axis").is_number_integer())
            throw std::invalid_argument("axis must be an integer");
        a.axis = object.at("axis").get<int>();
    }
    a.epsilon = object.value("epsilon", a.epsilon);
    if (object.contains("heads") && !object.at("heads").is_number_integer())
        throw std::invalid_argument("heads must be an integer");
    auto heads = object.value("heads", std::int64_t{1});
    if (heads < 1)
        throw std::invalid_argument("heads must be positive");
    a.heads = static_cast<std::size_t>(heads);
    a.causal = object.value("causal", false);
    a.alpha = object.value("alpha", 1.0f);
    a.beta = object.value("beta", 1.0f);
    a.trans_a = object.value("trans_a", false);
    a.trans_b = object.value("trans_b", false);
    if (!std::isfinite(a.epsilon) || a.epsilon <= 0 || !std::isfinite(a.alpha) || !std::isfinite(a.beta))
        throw std::invalid_argument("Operator scalar attributes must be finite and epsilon positive");
    return a;
}
} // namespace
void save_model(const ModelDefinition &model, const std::filesystem::path &path) {
    Json document{{"format", "octopus-model"},
                  {"version", 1},
                  {"nodes", Json::array()},
                  {"parameters", Json::object()}};
    for (const auto &node : model.graph.nodes())
        document["nodes"].push_back({{"id", node.id},
                                     {"op", op_name(node.op)},
                                     {"inputs", node.inputs},
                                     {"parameters", node.parameters},
                                     {"attributes", encode_attributes(node.attributes)}});
    for (const auto &[name, tensor] : model.parameters) {
        for (auto value : tensor.values())
            if (!std::isfinite(value))
                throw std::invalid_argument("Cannot serialize non-finite weights");
        document["parameters"][name] = {{"shape", tensor.shape()}, {"data", tensor.values()}};
    }
    std::ofstream output(path);
    if (!output)
        throw std::runtime_error("Cannot create model file: " + path.string());
    output << document.dump(2) << '\n';
    if (!output)
        throw std::runtime_error("Failed to write model file: " + path.string());
}
ModelDefinition load_json_model(const std::filesystem::path &path) {
    if (std::filesystem::file_size(path) > max_model_bytes)
        throw std::invalid_argument("Model file exceeds the 64 MiB limit");
    std::ifstream input(path);
    if (!input)
        throw std::runtime_error("Cannot read model file: " + path.string());
    auto document = Json::parse(input, [](int depth, Json::parse_event_t, Json &) {
        if (depth > 64)
            throw std::invalid_argument("Model JSON nesting exceeds 64 levels");
        return true;
    });
    if (!document.is_object() || document.value("format", std::string{}) != "octopus-model" ||
        document.at("version") != 1)
        throw std::invalid_argument("Expected octopus-model format version 1");
    GraphBuilder builder;
    const auto &nodes = document.at("nodes");
    if (!nodes.is_array() || nodes.size() > 100000)
        throw std::invalid_argument("Invalid model node list");
    for (const auto &object : nodes) {
        Node node;
        node.id = object.at("id").get<std::string>();
        node.op = parse_op(object.at("op").get<std::string>());
        node.inputs = object.value("inputs", std::vector<std::string>{});
        node.parameters = object.value("parameters", std::vector<std::string>{});
        node.attributes = decode_attributes(object.value("attributes", Json::object()));
        builder.add_node(std::move(node));
    }
    Parameters parameters;
    const auto &encoded = document.at("parameters");
    if (!encoded.is_object())
        throw std::invalid_argument("Model parameters must be an object");
    for (const auto &[name, value] : encoded.items()) {
        auto shape = read_shape(value.at("shape"));
        if (!value.at("data").is_array())
            throw std::invalid_argument("Parameter data must be an array");
        std::vector<float> data;
        data.reserve(value.at("data").size());
        for (const auto &item : value.at("data")) {
            if (!item.is_number())
                throw std::invalid_argument("Model weights must be numbers");
            auto x = item.get<float>();
            if (!std::isfinite(x))
                throw std::invalid_argument("Model weights must be finite");
            data.push_back(x);
        }
        parameters.emplace(name, TensorF32(std::move(data), shape));
    }
    return {builder.build(), std::move(parameters)};
}
ModelDefinition load_model(const std::filesystem::path &path) {
    if (path.extension() == ".onnx")
        return load_onnx_model(path);
    if (path.extension() == ".json")
        return load_json_model(path);
    throw std::invalid_argument("Supported model file extensions: .json, .onnx");
}
} // namespace octopus
