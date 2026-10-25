#include "octopus/model_io.hpp"
#include <bit>
#include <fstream>
#include <map>
#include <set>

namespace octopus {
namespace {
// Bounded protobuf wire reader for the documented ONNX inference subset.
// Field numbers follow https://github.com/onnx/onnx/blob/v1.17.0/onnx/onnx.proto3.
using Bytes = std::span<const std::uint8_t>;
class Wire {
  public:
    explicit Wire(Bytes bytes) : bytes_(bytes) {}
    bool empty() const { return position_ == bytes_.size(); }
    std::uint64_t integer() {
        std::uint64_t value = 0;
        for (unsigned shift = 0; shift < 70; shift += 7) {
            auto byte = take(1)[0];
            if (shift == 63 && byte > 1)
                throw std::invalid_argument("ONNX varint overflow");
            value |= static_cast<std::uint64_t>(byte & 127) << shift;
            if (!(byte & 128))
                return value;
        }
        throw std::invalid_argument("Invalid ONNX varint");
    }
    std::pair<unsigned, unsigned> tag() {
        auto key = integer();
        if ((key >> 3) == 0 || (key >> 3) > 536870911)
            throw std::invalid_argument("Invalid ONNX field number");
        return {static_cast<unsigned>(key >> 3), static_cast<unsigned>(key & 7)};
    }
    Bytes take(std::size_t size) {
        if (size > bytes_.size() - position_)
            throw std::invalid_argument("Truncated ONNX protobuf message");
        auto result = bytes_.subspan(position_, size);
        position_ += size;
        return result;
    }
    Bytes blob(unsigned type) {
        expect(type, 2);
        auto length = integer();
        if (length > bytes_.size() - position_)
            throw std::invalid_argument("Invalid ONNX message length");
        return take(static_cast<std::size_t>(length));
    }
    std::string text(unsigned type) {
        auto bytes = blob(type);
        return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
    }
    std::uint64_t number(unsigned type) {
        expect(type, 0);
        return integer();
    }
    float floating(unsigned type) {
        expect(type, 5);
        auto bytes = take(4);
        std::uint32_t bits = 0;
        for (unsigned i = 0; i < 4; ++i)
            bits |= static_cast<std::uint32_t>(bytes[i]) << (8 * i);
        return std::bit_cast<float>(bits);
    }
    void skip(unsigned type) {
        switch (type) {
        case 0:
            integer();
            break;
        case 1:
            take(8);
            break;
        case 2:
            blob(type);
            break;
        case 5:
            take(4);
            break;
        default:
            throw std::invalid_argument("Unsupported protobuf wire type");
        }
    }
    static void expect(unsigned actual, unsigned wanted) {
        if (actual != wanted)
            throw std::invalid_argument("Invalid ONNX protobuf field encoding");
    }

  private:
    Bytes bytes_;
    std::size_t position_ = 0;
};
void integers(Wire &wire, unsigned type, std::vector<std::int64_t> &result) {
    if (type == 0)
        result.push_back(std::bit_cast<std::int64_t>(wire.number(type)));
    else {
        Wire packed(wire.blob(type));
        while (!packed.empty())
            result.push_back(std::bit_cast<std::int64_t>(packed.integer()));
    }
}
struct Initializer {
    std::string name;
    Shape shape;
    int dtype = 0;
    std::vector<float> floats;
    std::vector<std::int64_t> ints;
};
Initializer read_initializer(Bytes bytes) {
    Wire wire(bytes);
    Initializer value;
    std::vector<std::int64_t> dimensions;
    Bytes raw;
    bool has_raw = false;
    while (!wire.empty()) {
        auto [field, type] = wire.tag();
        if (field == 1)
            integers(wire, type, dimensions);
        else if (field == 2)
            value.dtype = static_cast<int>(wire.number(type));
        else if (field == 4) {
            if (type == 5)
                value.floats.push_back(wire.floating(type));
            else {
                Wire packed(wire.blob(type));
                while (!packed.empty())
                    value.floats.push_back(packed.floating(5));
            }
        } else if (field == 7)
            integers(wire, type, value.ints);
        else if (field == 8)
            value.name = wire.text(type);
        else if (field == 9) {
            if (has_raw)
                throw std::invalid_argument("Duplicate ONNX raw_data");
            raw = wire.blob(type);
            has_raw = true;
        } else if (field == 14) {
            if (wire.number(type) != 0)
                throw std::invalid_argument("External ONNX data is unsupported");
        } else if (field == 3 || field == 5 || field == 6 || field == 10 || field == 11 || field == 13)
            throw std::invalid_argument("Unsupported ONNX tensor encoding or external data");
        else
            wire.skip(type);
    }
    if (dimensions.size() > 16 || value.name.empty())
        throw std::invalid_argument("Invalid ONNX initializer metadata");
    for (auto dimension : dimensions) {
        if (dimension <= 0)
            throw std::invalid_argument("ONNX initializer dimensions must be positive");
        value.shape.push_back(static_cast<std::size_t>(dimension));
    }
    auto count = element_count(value.shape);
    if (has_raw) {
        if (!value.floats.empty() || !value.ints.empty())
            throw std::invalid_argument("Conflicting ONNX tensor encodings");
        auto width = value.dtype == 1 ? 4u : value.dtype == 7 ? 8u : 0u;
        if (!width || raw.size() != checked_multiply(count, width))
            throw std::invalid_argument("ONNX raw tensor size or dtype mismatch");
        Wire data(raw);
        for (std::size_t i = 0; i < count; ++i) {
            if (value.dtype == 1)
                value.floats.push_back(data.floating(5));
            else {
                auto block = data.take(8);
                std::uint64_t bits = 0;
                for (unsigned j = 0; j < 8; ++j)
                    bits |= static_cast<std::uint64_t>(block[j]) << (8 * j);
                value.ints.push_back(std::bit_cast<std::int64_t>(bits));
            }
        }
    }
    if ((value.dtype != 1 && value.dtype != 7) ||
        (value.dtype == 1 ? value.floats.size() : value.ints.size()) != count ||
        (value.dtype == 1 && !value.ints.empty()) || (value.dtype == 7 && !value.floats.empty()))
        throw std::invalid_argument("Only complete FLOAT and INT64 ONNX initializers are supported");
    for (auto x : value.floats)
        if (!std::isfinite(x))
            throw std::invalid_argument("ONNX weights must be finite");
    return value;
}
struct Input {
    std::string name;
    Shape shape;
};
Input read_input(Bytes bytes) {
    Wire info(bytes);
    Input result;
    Bytes tensor_type;
    while (!info.empty()) {
        auto [field, type] = info.tag();
        if (field == 1)
            result.name = info.text(type);
        else if (field == 2) {
            Wire type_info(info.blob(type));
            while (!type_info.empty()) {
                auto [f, t] = type_info.tag();
                if (f == 1)
                    tensor_type = type_info.blob(t);
                else
                    type_info.skip(t);
            }
        } else
            info.skip(type);
    }
    if (tensor_type.empty() || result.name.empty())
        throw std::invalid_argument("ONNX input is not a named tensor");
    Wire tensor(tensor_type);
    std::uint64_t dtype = 0;
    bool has_shape = false;
    while (!tensor.empty()) {
        auto [field, type] = tensor.tag();
        if (field == 1)
            dtype = tensor.number(type);
        else if (field == 2) {
            has_shape = true;
            Wire shape(tensor.blob(type));
            while (!shape.empty()) {
                auto [f, t] = shape.tag();
                if (f != 1) {
                    shape.skip(t);
                    continue;
                }
                Wire dim(shape.blob(t));
                std::uint64_t size = 0;
                while (!dim.empty()) {
                    auto [df, dt] = dim.tag();
                    if (df == 1)
                        size = dim.number(dt);
                    else if (df == 2)
                        throw std::invalid_argument("Dynamic ONNX dimensions are not supported");
                    else
                        dim.skip(dt);
                }
                if (!size || size > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                    throw std::invalid_argument("ONNX input dimensions must be fixed and positive");
                result.shape.push_back(static_cast<std::size_t>(size));
                if (result.shape.size() > 16)
                    throw std::invalid_argument("ONNX tensor rank exceeds 16");
            }
        } else
            tensor.skip(type);
    }
    if (dtype != 1 || !has_shape)
        throw std::invalid_argument("ONNX graph inputs must be fixed-shape FLOAT tensors");
    element_count(result.shape);
    return result;
}
struct Attribute {
    int type = 0;
    float floating = 0;
    std::int64_t integer = 0;
    std::string text;
    std::vector<std::int64_t> ints;
};
std::pair<std::string, Attribute> read_attribute(Bytes bytes) {
    Wire wire(bytes);
    std::string name;
    Attribute value;
    while (!wire.empty()) {
        auto [field, type] = wire.tag();
        if (field == 1)
            name = wire.text(type);
        else if (field == 2)
            value.floating = wire.floating(type);
        else if (field == 3)
            value.integer = std::bit_cast<std::int64_t>(wire.number(type));
        else if (field == 4)
            value.text = wire.text(type);
        else if (field == 8)
            integers(wire, type, value.ints);
        else if (field == 20)
            value.type = static_cast<int>(wire.number(type));
        else if (field == 13)
            wire.skip(type);
        else
            throw std::invalid_argument("Unsupported ONNX attribute encoding");
    }
    if (name.empty() || (value.type != 1 && value.type != 2 && value.type != 3 && value.type != 7))
        throw std::invalid_argument("Unsupported ONNX attribute type");
    return {name, value};
}
struct OnnxNode {
    std::string op, domain;
    std::vector<std::string> inputs, outputs;
    std::map<std::string, Attribute> attributes;
};
OnnxNode read_node(Bytes bytes) {
    Wire wire(bytes);
    OnnxNode node;
    while (!wire.empty()) {
        auto [field, type] = wire.tag();
        if (field == 1)
            node.inputs.push_back(wire.text(type));
        else if (field == 2)
            node.outputs.push_back(wire.text(type));
        else if (field == 4)
            node.op = wire.text(type);
        else if (field == 5) {
            auto [name, a] = read_attribute(wire.blob(type));
            if (!node.attributes.emplace(name, std::move(a)).second)
                throw std::invalid_argument("Duplicate ONNX attribute");
        } else if (field == 7)
            node.domain = wire.text(type);
        else if (field == 8) {
            if (!wire.text(type).empty())
                throw std::invalid_argument("ONNX function overloads are unsupported");
        } else
            wire.skip(type);
    }
    if (!node.domain.empty() && node.domain != "ai.onnx")
        throw std::invalid_argument("Unsupported ONNX operator domain");
    if (node.outputs.size() != 1 || node.outputs[0].empty())
        throw std::invalid_argument("ONNX nodes must have one nonempty output");
    return node;
}
const Attribute *attribute(const OnnxNode &node, const std::string &name, int type) {
    auto found = node.attributes.find(name);
    if (found == node.attributes.end())
        return nullptr;
    if (found->second.type != type)
        throw std::invalid_argument("ONNX attribute type mismatch: " + name);
    return &found->second;
}
std::int64_t integer_attr(const OnnxNode &n, const std::string &name, std::int64_t fallback) {
    auto a = attribute(n, name, 2);
    return a ? a->integer : fallback;
}
float float_attr(const OnnxNode &n, const std::string &name, float fallback) {
    auto a = attribute(n, name, 1);
    return a ? a->floating : fallback;
}
Shape shape_attr(const OnnxNode &n, const std::string &name, Shape fallback, bool positive) {
    auto a = attribute(n, name, 7);
    if (!a)
        return fallback;
    Shape result;
    for (auto x : a->ints) {
        if (x < 0 || (positive && x == 0))
            throw std::invalid_argument("Invalid ONNX dimension attribute");
        result.push_back(static_cast<std::size_t>(x));
    }
    return result;
}
void allowed_attributes(const OnnxNode &n, std::set<std::string> names) {
    for (const auto &[name, value] : n.attributes) {
        (void)value;
        if (!names.contains(name))
            throw std::invalid_argument("Unsupported ONNX attribute: " + n.op + "." + name);
    }
}
} // namespace

ModelDefinition load_onnx_model(const std::filesystem::path &path) {
    auto size = std::filesystem::file_size(path);
    if (!size || size > 64 * 1024 * 1024)
        throw std::invalid_argument("ONNX file must be between 1 byte and 64 MiB");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    std::ifstream input(path, std::ios::binary);
    if (!input.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size())))
        throw std::runtime_error("Cannot read ONNX model");
    Wire model(bytes);
    Bytes graph_bytes;
    std::uint64_t opset = 0, ir = 0;
    while (!model.empty()) {
        auto [field, type] = model.tag();
        if (field == 1)
            ir = model.number(type);
        else if (field == 7) {
            if (!graph_bytes.empty())
                throw std::invalid_argument("Duplicate ONNX graph");
            graph_bytes = model.blob(type);
        } else if (field == 8) {
            Wire version(model.blob(type));
            std::string domain;
            std::uint64_t number = 0;
            while (!version.empty()) {
                auto [f, t] = version.tag();
                if (f == 1)
                    domain = version.text(t);
                else if (f == 2)
                    number = version.number(t);
                else
                    version.skip(t);
            }
            if (domain.empty() || domain == "ai.onnx") {
                if (opset)
                    throw std::invalid_argument("Duplicate ONNX opset import");
                opset = number;
            }
        } else if (field == 20 || field == 25)
            throw std::invalid_argument("ONNX training graphs and local functions are unsupported");
        else
            model.skip(type);
    }
    if (ir < 3 || ir > 10 || opset < 13 || opset > 22 || graph_bytes.empty())
        throw std::invalid_argument("Supported ONNX versions: IR 3-10, standard opset 13-22");
    std::vector<OnnxNode> nodes;
    std::vector<Input> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, Initializer> initializers;
    Wire graph(graph_bytes);
    while (!graph.empty()) {
        auto [field, type] = graph.tag();
        if (field == 1) {
            nodes.push_back(read_node(graph.blob(type)));
            if (nodes.size() > 100000)
                throw std::invalid_argument("Too many ONNX nodes");
        } else if (field == 5) {
            auto value = read_initializer(graph.blob(type));
            auto name = value.name;
            if (!initializers.emplace(name, std::move(value)).second)
                throw std::invalid_argument("Duplicate ONNX initializer");
        } else if (field == 11)
            inputs.push_back(read_input(graph.blob(type)));
        else if (field == 12) {
            Wire value(graph.blob(type));
            std::string name;
            while (!value.empty()) {
                auto [f, t] = value.tag();
                if (f == 1)
                    name = value.text(t);
                else
                    value.skip(t);
            }
            if (name.empty())
                throw std::invalid_argument("Empty ONNX output name");
            outputs.push_back(name);
        } else if (field == 15)
            throw std::invalid_argument("Sparse ONNX initializers are unsupported");
        else
            graph.skip(type);
    }
    GraphBuilder builder;
    Parameters parameters;
    std::map<std::string, std::string> values;
    auto bind = [&](const std::string &name, const std::string &id) {
        if (!values.emplace(name, id).second)
            throw std::invalid_argument("Duplicate ONNX value name: " + name);
    };
    for (const auto &value : inputs) {
        if (initializers.contains(value.name))
            continue;
        builder.input(value.name, value.shape);
        bind(value.name, value.name);
    }
    std::size_t constant_id = 0;
    for (const auto &[name, value] : initializers)
        if (value.dtype == 1) {
            auto id = "@onnx/constant/" + std::to_string(constant_id++);
            parameters.emplace(id, TensorF32(value.floats, value.shape));
            builder.constant(id, id);
            bind(name, id);
        }
    // Resolve names before graph construction so serialized node order need not be topological.
    for (std::size_t i = 0; i < nodes.size(); ++i)
        bind(nodes[i].outputs[0], "@onnx/node/" + std::to_string(i));
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto &source = nodes[i];
        Node node;
        node.id = values.at(source.outputs[0]);
        auto names = source.inputs;
        if (source.op == "Reshape") {
            allowed_attributes(source, {"allowzero"});
            if (integer_attr(source, "allowzero", 0) != 0 || names.size() != 2)
                throw std::invalid_argument("Unsupported ONNX Reshape mode");
            auto shape = initializers.find(names[1]);
            if (shape == initializers.end() || shape->second.dtype != 7 || shape->second.shape.size() != 1)
                throw std::invalid_argument("ONNX Reshape requires a constant INT64 shape vector");
            node.op = Op::Reshape;
            node.attributes.target_shape = shape->second.ints;
            names.pop_back();
        } else if (source.op == "Conv") {
            allowed_attributes(source, {"strides", "pads", "dilations", "group", "auto_pad", "kernel_shape"});
            auto auto_pad = attribute(source, "auto_pad", 3);
            if ((auto_pad && auto_pad->text != "NOTSET") || integer_attr(source, "group", 1) != 1 ||
                shape_attr(source, "dilations", {1, 1}, true) != Shape{1, 1})
                throw std::invalid_argument(
                    "ONNX Conv supports group=1, unit dilation, and explicit padding only");
            auto pads = shape_attr(source, "pads", {0, 0, 0, 0}, false);
            if (pads.size() != 4 || pads[0] != pads[2] || pads[1] != pads[3])
                throw std::invalid_argument("ONNX Conv requires symmetric 2D padding");
            node.attributes.strides = shape_attr(source, "strides", {1, 1}, true);
            node.attributes.padding = {pads[0], pads[1]};
            if (source.attributes.contains("kernel_shape")) {
                if (names.size() < 2 || !initializers.contains(names[1]))
                    throw std::invalid_argument("Conv kernel_shape validation requires constant weights");
                const auto &shape = initializers.at(names[1]).shape;
                if (shape.size() != 4 ||
                    shape_attr(source, "kernel_shape", {}, true) != Shape{shape[2], shape[3]})
                    throw std::invalid_argument("ONNX Conv kernel shape mismatch");
            }
            node.op = Op::Conv2D;
        } else if (source.op == "Gemm") {
            allowed_attributes(source, {"alpha", "beta", "transA", "transB"});
            node.op = Op::Gemm;
            node.attributes.alpha = float_attr(source, "alpha", 1);
            node.attributes.beta = float_attr(source, "beta", 1);
            auto ta = integer_attr(source, "transA", 0), tb = integer_attr(source, "transB", 0);
            if ((ta != 0 && ta != 1) || (tb != 0 && tb != 1))
                throw std::invalid_argument("Invalid Gemm transpose attribute");
            node.attributes.trans_a = ta == 1;
            node.attributes.trans_b = tb == 1;
        } else if (source.op == "Softmax" || source.op == "Gather") {
            allowed_attributes(source, {"axis"});
            auto axis = integer_attr(source, "axis", source.op == "Gather" ? 0 : -1);
            if (axis < std::numeric_limits<int>::min() || axis > std::numeric_limits<int>::max())
                throw std::invalid_argument("ONNX axis exceeds integer range");
            node.attributes.axis = static_cast<int>(axis);
            if (source.op == "Gather") {
                if (axis != 0)
                    throw std::invalid_argument("ONNX Gather supports axis=0 only");
                throw std::invalid_argument("ONNX Gather requires integer graph inputs, which are not "
                                            "supported; use native Embedding");
            }
            node.op = Op::Softmax;
        } else if (source.op == "Transpose") {
            allowed_attributes(source, {"perm"});
            node.op = Op::Transpose;
            node.attributes.axes = shape_attr(source, "perm", {}, false);
        } else if (source.op == "LayerNormalization") {
            allowed_attributes(source, {"axis", "epsilon", "stash_type"});
            if (opset < 17 || integer_attr(source, "axis", -1) != -1 ||
                integer_attr(source, "stash_type", 1) != 1)
                throw std::invalid_argument(
                    "ONNX LayerNormalization supports float32 normalization on the last axis");
            node.op = Op::LayerNorm;
            node.attributes.epsilon = float_attr(source, "epsilon", 1e-5f);
        } else if (source.op == "Gelu") {
            allowed_attributes(source, {"approximate"});
            auto approximation = attribute(source, "approximate", 3);
            if (opset < 20 || !approximation || approximation->text != "tanh")
                throw std::invalid_argument("ONNX Gelu requires opset >=20 and approximate=tanh");
            node.op = Op::Gelu;
        } else {
            allowed_attributes(source, {});
            static const std::map<std::string, Op> operators{
                {"Add", Op::Add},   {"Sub", Op::Sub},         {"Mul", Op::Mul},   {"MatMul", Op::MatMul},
                {"Relu", Op::Relu}, {"Sigmoid", Op::Sigmoid}, {"Tanh", Op::Tanh}, {"Identity", Op::Identity}};
            auto found = operators.find(source.op);
            if (found == operators.end())
                throw std::invalid_argument("Unsupported ONNX operator: " + source.op);
            node.op = found->second;
        }
        if (!std::isfinite(node.attributes.alpha) || !std::isfinite(node.attributes.beta) ||
            !std::isfinite(node.attributes.epsilon) || node.attributes.epsilon <= 0)
            throw std::invalid_argument("Invalid ONNX scalar attributes");
        for (const auto &name : names) {
            auto found = values.find(name);
            if (found == values.end())
                throw std::invalid_argument("Unsupported or unresolved ONNX input: " + name);
            node.inputs.push_back(found->second);
        }
        builder.add_node(std::move(node));
    }
    for (const auto &name : outputs) {
        auto found = values.find(name);
        if (found == values.end())
            throw std::invalid_argument("Unresolved ONNX graph output: " + name);
        builder.output(name, found->second);
    }
    return {builder.build(), std::move(parameters)};
}
} // namespace octopus
