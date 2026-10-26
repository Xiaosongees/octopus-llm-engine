#include "octopus/inference.hpp"
#include <chrono>
#include <utility>

namespace octopus {
InferenceEngine::InferenceEngine(Graph graph, Parameters parameters, BackendType backend)
    : graph_(std::move(graph)) {
    if (backend != BackendType::Cpu)
        throw std::invalid_argument("CUDA is not implemented; use --backend cpu");
    std::vector<std::string> dense_weights;
    for (const auto &node : graph_.nodes()) {
        for (const auto &name : node.parameters)
            if (!parameters.contains(name))
                throw std::invalid_argument("Missing model parameter: " + name);
        if ((node.op == Op::Dense || node.op == Op::Embedding) && node.parameters.empty())
            throw std::invalid_argument("Missing weights for node: " + node.id);
        if (node.op == Op::Conv2D && node.parameters.empty() && node.inputs.size() < 2)
            throw std::invalid_argument("Missing Conv2D weights: " + node.id);
        if (node.op == Op::Dense)
            dense_weights.push_back(node.parameters[0]);
    }
    backend_ = std::make_shared<CpuBackend>(std::move(parameters), dense_weights);
}
InferenceResult InferenceEngine::run(const TensorMap &inputs) const {
    using Clock = std::chrono::steady_clock;
    auto start = Clock::now();
    TensorMap values;
    if (inputs.size() != graph_.input_names().size())
        throw std::invalid_argument("Provide exactly the model's declared inputs");
    for (const auto &name : graph_.input_names()) {
        auto found = inputs.find(name);
        if (found == inputs.end())
            throw std::invalid_argument("Missing input: " + name);
        if (found->second.shape() != graph_.node(name).attributes.shape)
            throw std::invalid_argument("Input shape mismatch: " + name);
        values.emplace(name, found->second);
    }
    std::unordered_map<std::string, std::size_t> uses;
    for (const auto &node : graph_.nodes())
        for (const auto &input : node.inputs)
            ++uses[input];
    InferenceResult result;
    for (auto index : graph_.execution_order()) {
        const auto &node = graph_.nodes()[index];
        if (node.op == Op::Input)
            continue;
        std::vector<TensorF32> arguments;
        for (const auto &input : node.inputs)
            arguments.push_back(values.at(input));
        auto node_start = Clock::now();
        try {
            values.insert_or_assign(node.id, backend_->forward(node, arguments));
        } catch (const std::exception &error) {
            throw std::invalid_argument("Node '" + node.id + "' (" + std::string(op_name(node.op)) +
                                        "): " + error.what());
        }
        result.stats.operator_times_ms[node.id] =
            std::chrono::duration<double, std::milli>(Clock::now() - node_start).count();
        ++result.stats.nodes_executed;
        if (node.op == Op::Output)
            result.outputs.emplace(node.id, values.at(node.id));
        for (const auto &input : node.inputs)
            if (--uses.at(input) == 0)
                values.erase(input);
        if (!uses.contains(node.id))
            values.erase(node.id);
    }
    result.stats.backend = backend_->name();
    result.stats.total_time_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    return result;
}
TensorF32 InferenceEngine::infer(const std::string &name, const TensorF32 &input,
                                 const std::string &output) const {
    auto result = run({{name, input}});
    auto selected = output.empty() ? graph_.output_names().front() : output;
    auto found = result.outputs.find(selected);
    if (found == result.outputs.end())
        throw std::invalid_argument("Unknown output: " + selected);
    return found->second;
}
InferenceEngine make_sample_engine() {
    GraphBuilder builder;
    builder.input("input", {1, 4});
    builder.dense("fc1", "input");
    builder.unary("relu1", Op::Relu, "fc1");
    builder.output("output", "relu1");
    Parameters parameters{{"fc1.weight", TensorF32({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}, {3, 4})},
                          {"fc1.bias", TensorF32({0.5f, -0.5f, 1}, {3})}};
    return InferenceEngine(builder.build(), std::move(parameters));
}
} // namespace octopus
