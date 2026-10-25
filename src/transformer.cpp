#include "octopus/transformer.hpp"
#include <utility>

namespace octopus {
void TransformerConfig::validate() const {
    if (!d_model || !n_heads || d_model % n_heads || !n_layers || !d_ff || !vocab_size || !max_seq_len)
        throw std::invalid_argument(
            "Transformer dimensions must be positive and d_model divisible by n_heads");
    if (!std::isfinite(epsilon) || epsilon <= 0)
        throw std::invalid_argument("Transformer epsilon must be positive");
    if (vocab_size > 16777216 || max_seq_len > 16777216)
        throw std::invalid_argument("Float32 token indices require vocabulary and position counts <= 2^24");
}
TransformerEncoder::TransformerEncoder(TransformerConfig config) : config_(config) {
    config_.validate();
}
ModelDefinition TransformerEncoder::build(std::size_t batch, std::size_t sequence, std::uint32_t seed) const {
    if (!batch || !sequence || sequence > config_.max_seq_len)
        throw std::invalid_argument("Invalid transformer batch or sequence length");
    const auto &c = config_;
    GraphBuilder graph;
    Parameters parameters;
    std::mt19937 generator(seed);
    auto random_weight = [&](const std::string &name, const Shape &shape, float scale) {
        std::uniform_real_distribution<float> distribution(-scale, scale);
        std::vector<float> data(element_count(shape));
        for (auto &value : data)
            value = distribution(generator);
        parameters.emplace(name, TensorF32(std::move(data), shape));
    };
    auto dense = [&](const std::string &name, const std::string &input, std::size_t in, std::size_t out) {
        random_weight(name + ".weight", {out, in}, std::sqrt(6.0f / static_cast<float>(in + out)));
        parameters.emplace(name + ".bias", TensorF32::zeros({out}));
        return graph.dense(name, input);
    };
    auto norm = [&](const std::string &name, const std::string &input) {
        parameters.emplace(name + ".weight", TensorF32::ones({c.d_model}));
        parameters.emplace(name + ".bias", TensorF32::zeros({c.d_model}));
        return graph.layer_norm(name, input, {c.d_model}, c.epsilon);
    };
    graph.input("tokens", {batch, sequence});
    random_weight("token_embedding.weight", {c.vocab_size, c.d_model}, 0.02f);
    random_weight("position_embedding.weight", {c.max_seq_len, c.d_model}, 0.02f);
    graph.add_node({"token_embedding", Op::Embedding, {"tokens"}, {"token_embedding.weight"}, {}});
    std::vector<float> positions(sequence);
    std::iota(positions.begin(), positions.end(), 0.0f);
    parameters.emplace("positions", TensorF32(std::move(positions), {1, sequence}));
    graph.constant("positions", "positions");
    graph.add_node({"position_embedding", Op::Embedding, {"positions"}, {"position_embedding.weight"}, {}});
    auto current = graph.binary("embeddings", Op::Add, "token_embedding", "position_embedding");
    for (std::size_t i = 0; i < c.n_layers; ++i) {
        const auto prefix = "block_" + std::to_string(i);
        auto n1 = norm(prefix + ".norm1", current);
        auto q = dense(prefix + ".q_proj", n1, c.d_model, c.d_model);
        auto k = dense(prefix + ".k_proj", n1, c.d_model, c.d_model);
        auto v = dense(prefix + ".v_proj", n1, c.d_model, c.d_model);
        Attributes attributes;
        attributes.heads = c.n_heads;
        attributes.causal = c.causal;
        auto attention = graph.add_node({prefix + ".attention", Op::Attention, {q, k, v}, {}, attributes});
        auto projected = dense(prefix + ".out_proj", attention, c.d_model, c.d_model);
        auto residual = graph.binary(prefix + ".residual1", Op::Add, current, projected);
        auto n2 = norm(prefix + ".norm2", residual);
        auto hidden = dense(prefix + ".ffn.fc1", n2, c.d_model, c.d_ff);
        auto activated = graph.unary(prefix + ".ffn.activation", c.use_gelu ? Op::Gelu : Op::Relu, hidden);
        auto output = dense(prefix + ".ffn.fc2", activated, c.d_ff, c.d_model);
        current = graph.binary(prefix + ".residual2", Op::Add, residual, output);
    }
    auto normalized = norm("final_norm", current);
    graph.output("output", normalized);
    return {graph.build(), std::move(parameters)};
}
ModelStats model_stats(const Parameters &parameters) {
    ModelStats result;
    for (const auto &[name, tensor] : parameters) {
        (void)name;
        if (tensor.size() > std::numeric_limits<std::size_t>::max() - result.total_parameters)
            throw std::invalid_argument("Parameter count overflows size_t");
        result.total_parameters += tensor.size();
    }
    result.weight_bytes = checked_multiply(result.total_parameters, sizeof(float));
    return result;
}
} // namespace octopus
