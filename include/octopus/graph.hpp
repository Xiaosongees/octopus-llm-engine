#pragma once
#include "octopus/tensor.hpp"
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace octopus {

enum class Op {
    Input,
    Output,
    Constant,
    Identity,
    Dense,
    Conv2D,
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
    Softmax,
    LayerNorm,
    MatMul,
    Add,
    Sub,
    Mul,
    Reshape,
    Transpose,
    Embedding,
    Gather,
    Attention,
    Dropout,
    Gemm
};
std::string_view op_name(Op op);
Op parse_op(std::string_view name);

struct Attributes {
    Shape shape;
    std::vector<std::int64_t> target_shape;
    Shape axes;
    Shape strides{1, 1};
    Shape padding{0, 0};
    int axis = -1;
    float epsilon = 1e-5f;
    std::size_t heads = 1;
    bool causal = false;
    float alpha = 1.0f;
    float beta = 1.0f;
    bool trans_a = false;
    bool trans_b = false;
};

struct Node {
    std::string id;
    Op op = Op::Identity;
    std::vector<std::string> inputs;
    std::vector<std::string> parameters;
    Attributes attributes;
};

class Graph {
  public:
    explicit Graph(std::vector<Node> nodes);
    const std::vector<Node> &nodes() const noexcept { return nodes_; }
    const std::vector<std::size_t> &execution_order() const noexcept { return order_; }
    const std::vector<std::string> &input_names() const noexcept { return inputs_; }
    const std::vector<std::string> &output_names() const noexcept { return outputs_; }
    const Node &node(const std::string &id) const;

  private:
    std::vector<Node> nodes_;
    std::unordered_map<std::string, std::size_t> index_;
    std::vector<std::size_t> order_;
    std::vector<std::string> inputs_, outputs_;
};

class GraphBuilder {
  public:
    std::string add_node(Node node);
    std::string input(std::string name, Shape shape);
    std::string output(std::string name, std::string input);
    std::string constant(std::string name, std::string parameter);
    std::string dense(std::string name, std::string input, bool bias = true);
    std::string unary(std::string name, Op op, std::string input, Attributes attributes = {});
    std::string binary(std::string name, Op op, std::string a, std::string b);
    std::string layer_norm(std::string name, std::string input, Shape shape, float epsilon = 1e-5f);
    Graph build() const { return Graph(nodes_); }

  private:
    std::vector<Node> nodes_;
};

} // namespace octopus
