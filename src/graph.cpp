#include "octopus/graph.hpp"
#include <deque>
#include <utility>

namespace octopus {
namespace {
constexpr std::string_view names[] = {"Input",     "Output", "Constant",  "Identity", "Dense",   "Conv2D",
                                      "Relu",      "Gelu",   "Sigmoid",   "Tanh",     "Softmax", "LayerNorm",
                                      "MatMul",    "Add",    "Sub",       "Mul",      "Reshape", "Transpose",
                                      "Embedding", "Gather", "Attention", "Dropout",  "Gemm"};
void check_arity(const Node &node) {
    auto n = node.inputs.size();
    bool valid = false;
    switch (node.op) {
    case Op::Input:
    case Op::Constant:
        valid = n == 0;
        break;
    case Op::Add:
    case Op::Sub:
    case Op::Mul:
    case Op::MatMul:
    case Op::Gather:
        valid = n == 2;
        break;
    case Op::Attention:
        valid = n == 3 || n == 4;
        break;
    case Op::Gemm:
        valid = n == 2 || n == 3;
        break;
    case Op::Conv2D:
    case Op::LayerNorm:
        valid = n >= 1 && n <= 3;
        break;
    default:
        valid = n == 1;
        break;
    }
    if (!valid)
        throw std::invalid_argument("Invalid input count for node '" + node.id + "'");
    if (node.op == Op::Input)
        element_count(node.attributes.shape);
    if (node.op == Op::Constant && node.parameters.size() != 1)
        throw std::invalid_argument("Constant nodes require exactly one parameter");
    auto count = node.parameters.size();
    bool valid_parameters = false;
    switch (node.op) {
    case Op::Constant:
    case Op::Embedding:
        valid_parameters = count == 1;
        break;
    case Op::Dense:
        valid_parameters = count == 1 || count == 2;
        break;
    case Op::LayerNorm:
    case Op::Conv2D:
        valid_parameters = count <= 2;
        break;
    default:
        valid_parameters = count == 0;
        break;
    }
    if (!valid_parameters)
        throw std::invalid_argument("Invalid parameter count for node '" + node.id + "'");
    if ((node.op == Op::LayerNorm || node.op == Op::Conv2D) && count && n != 1)
        throw std::invalid_argument("Use either parameter weights or input weights for node '" + node.id +
                                    "'");
}
} // namespace
std::string_view op_name(Op op) {
    auto index = static_cast<std::size_t>(op);
    if (index >= std::size(names))
        throw std::invalid_argument("Invalid operator enum");
    return names[index];
}
Op parse_op(std::string_view name) {
    for (std::size_t i = 0; i < std::size(names); ++i)
        if (names[i] == name)
            return static_cast<Op>(i);
    throw std::invalid_argument("Unsupported operator: " + std::string(name));
}
Graph::Graph(std::vector<Node> nodes) : nodes_(std::move(nodes)) {
    std::vector<std::size_t> indegree(nodes_.size());
    std::vector<std::vector<std::size_t>> consumers(nodes_.size());
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
        const auto &n = nodes_[i];
        if (n.id.empty() || !index_.emplace(n.id, i).second)
            throw std::invalid_argument("Empty or duplicate node name: " + n.id);
        check_arity(n);
        if (n.op == Op::Input)
            inputs_.push_back(n.id);
        if (n.op == Op::Output)
            outputs_.push_back(n.id);
    }
    if (inputs_.empty() || outputs_.empty())
        throw std::invalid_argument("A graph needs input and output nodes");
    for (std::size_t i = 0; i < nodes_.size(); ++i)
        for (const auto &input : nodes_[i].inputs) {
            auto source = index_.find(input);
            if (source == index_.end())
                throw std::invalid_argument("Unknown graph input: " + input);
            consumers[source->second].push_back(i); // Preserve repeated edges such as Add(x, x).
            ++indegree[i];
        }
    std::deque<std::size_t> ready;
    for (std::size_t i = 0; i < indegree.size(); ++i)
        if (indegree[i] == 0)
            ready.push_back(i);
    while (!ready.empty()) {
        auto i = ready.front();
        ready.pop_front();
        order_.push_back(i);
        for (auto consumer : consumers[i])
            if (--indegree[consumer] == 0)
                ready.push_back(consumer);
    }
    if (order_.size() != nodes_.size())
        throw std::invalid_argument("The graph contains a dependency cycle");
}
const Node &Graph::node(const std::string &id) const {
    auto found = index_.find(id);
    if (found == index_.end())
        throw std::invalid_argument("Unknown node: " + id);
    return nodes_[found->second];
}
std::string GraphBuilder::add_node(Node node) {
    auto id = node.id;
    nodes_.push_back(std::move(node));
    return id;
}
std::string GraphBuilder::input(std::string name, Shape shape) {
    Node node{std::move(name), Op::Input, {}, {}, {}};
    node.attributes.shape = std::move(shape);
    return add_node(std::move(node));
}
std::string GraphBuilder::output(std::string name, std::string input) {
    return add_node({std::move(name), Op::Output, {std::move(input)}, {}, {}});
}
std::string GraphBuilder::constant(std::string name, std::string parameter) {
    return add_node({std::move(name), Op::Constant, {}, {std::move(parameter)}, {}});
}
std::string GraphBuilder::dense(std::string name, std::string input, bool bias) {
    Node node{name, Op::Dense, {std::move(input)}, {name + ".weight"}, {}};
    if (bias)
        node.parameters.push_back(name + ".bias");
    return add_node(std::move(node));
}
std::string GraphBuilder::unary(std::string name, Op op, std::string input, Attributes attributes) {
    return add_node({std::move(name), op, {std::move(input)}, {}, std::move(attributes)});
}
std::string GraphBuilder::binary(std::string name, Op op, std::string a, std::string b) {
    return add_node({std::move(name), op, {std::move(a), std::move(b)}, {}, {}});
}
std::string GraphBuilder::layer_norm(std::string name, std::string input, Shape shape, float epsilon) {
    Node node{name, Op::LayerNorm, {std::move(input)}, {name + ".weight", name + ".bias"}, {}};
    node.attributes.shape = std::move(shape);
    node.attributes.epsilon = epsilon;
    return add_node(std::move(node));
}
} // namespace octopus
