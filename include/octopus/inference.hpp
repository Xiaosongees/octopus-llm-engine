#pragma once
#include "octopus/backend.hpp"
#include <map>

namespace octopus {
using TensorMap = std::unordered_map<std::string, TensorF32>;
struct InferenceStats {
    double total_time_ms = 0;
    std::map<std::string, double> operator_times_ms;
    std::size_t nodes_executed = 0;
    std::string backend = "cpu";
};
struct InferenceResult {
    TensorMap outputs;
    InferenceStats stats;
};

// A loaded model is immutable. Each run owns its inputs, outputs, and scratch data.
class InferenceEngine {
  public:
    InferenceEngine(Graph graph, Parameters parameters = {}, BackendType backend = BackendType::Cpu);
    InferenceResult run(const TensorMap &inputs) const;
    TensorF32 infer(const std::string &name, const TensorF32 &input, const std::string &output = "") const;
    const Graph &graph() const noexcept { return graph_; }
    std::string backend_name() const { return backend_->name(); }

  private:
    Graph graph_;
    std::shared_ptr<const Backend> backend_;
};

InferenceEngine make_sample_engine();
} // namespace octopus
