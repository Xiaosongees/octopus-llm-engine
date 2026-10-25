#pragma once
#include "octopus/graph.hpp"

namespace octopus {
using Parameters = std::unordered_map<std::string, TensorF32>;
enum class BackendType { Cpu, Cuda };
BackendType parse_backend(const std::string &name);

TensorF32 embedding(const TensorF32 &indices, const TensorF32 &table);
TensorF32 conv2d(const TensorF32 &input, const TensorF32 &weight, const TensorF32 *bias = nullptr,
                 Shape strides = {1, 1}, Shape padding = {0, 0});
TensorF32 scaled_dot_product_attention(const TensorF32 &query, const TensorF32 &key, const TensorF32 &value,
                                       const TensorF32 *mask = nullptr, bool causal = false);

class Backend {
  public:
    virtual ~Backend() = default;
    virtual std::string name() const = 0;
    virtual TensorF32 forward(const Node &node, std::span<const TensorF32> inputs) const = 0;
};

class CpuBackend final : public Backend {
  public:
    explicit CpuBackend(Parameters parameters, const std::vector<std::string> &dense_weights = {});
    std::string name() const override { return "cpu"; }
    TensorF32 forward(const Node &node, std::span<const TensorF32> inputs) const override;
    const TensorF32 &parameter(const std::string &name) const;

  private:
    Parameters parameters_;
    Parameters transposed_weights_;
};

} // namespace octopus
