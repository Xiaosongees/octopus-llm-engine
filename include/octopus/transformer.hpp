#pragma once
#include "octopus/inference.hpp"

namespace octopus {
struct TransformerConfig {
    std::size_t d_model = 64;
    std::size_t n_heads = 4;
    std::size_t n_layers = 2;
    std::size_t d_ff = 128;
    std::size_t vocab_size = 1000;
    std::size_t max_seq_len = 32;
    float epsilon = 1e-5f;
    bool use_gelu = true;
    bool causal = false;
    void validate() const;
};

struct ModelStats {
    std::size_t total_parameters = 0;
    std::size_t weight_bytes = 0;
};
struct ModelDefinition {
    Graph graph;
    Parameters parameters;
};

class TransformerEncoder {
  public:
    explicit TransformerEncoder(TransformerConfig config = {});
    ModelDefinition build(std::size_t batch, std::size_t sequence, std::uint32_t seed = 42) const;
    const TransformerConfig &config() const noexcept { return config_; }

  private:
    TransformerConfig config_;
};
ModelStats model_stats(const Parameters &parameters);
} // namespace octopus
