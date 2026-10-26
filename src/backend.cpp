#include "octopus/backend.hpp"
#include <utility>

namespace octopus {
BackendType parse_backend(const std::string &name) {
    if (name == "cpu")
        return BackendType::Cpu;
    if (name == "cuda")
        return BackendType::Cuda;
    throw std::invalid_argument("Unsupported backend: " + name);
}

TensorF32 embedding(const TensorF32 &indices, const TensorF32 &table) {
    if (table.rank() != 2)
        throw std::invalid_argument("Embedding table must have rank 2");
    auto width = table.shape()[1];
    auto shape = indices.shape();
    shape.push_back(width);
    std::vector<float> output(element_count(shape));
    for (std::size_t i = 0; i < indices.size(); ++i) {
        double token = indices.values()[i];
        if (!std::isfinite(token) || token < 0 || token >= static_cast<double>(table.shape()[0]) ||
            std::floor(token) != token)
            throw std::invalid_argument("Embedding indices must be in-range nonnegative integers");
        auto offset = static_cast<std::size_t>(token) * width;
        std::copy_n(table.values().begin() + static_cast<std::ptrdiff_t>(offset), width,
                    output.begin() + static_cast<std::ptrdiff_t>(i * width));
    }
    return TensorF32(std::move(output), shape);
}

TensorF32 conv2d(const TensorF32 &input, const TensorF32 &weight, const TensorF32 *bias, Shape strides,
                 Shape padding) {
    if (input.rank() != 4 || weight.rank() != 4 || strides.size() != 2 || padding.size() != 2 ||
        !strides[0] || !strides[1])
        throw std::invalid_argument("Conv2D expects NCHW input, OIHW weights, and positive 2D strides");
    const auto &x = input.shape();
    const auto &w = weight.shape();
    if (x[1] != w[1])
        throw std::invalid_argument("Conv2D channel counts differ");
    if (bias && bias->shape() != Shape{w[0]})
        throw std::invalid_argument("Conv2D bias shape mismatch");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max());
    if (padding[0] > (limit - x[2]) / 2 || padding[1] > (limit - x[3]) / 2)
        throw std::invalid_argument("Conv2D padding is too large");
    auto padded_h = x[2] + 2 * padding[0], padded_w = x[3] + 2 * padding[1];
    if (padded_h < w[2] || padded_w < w[3])
        throw std::invalid_argument("Conv2D kernel exceeds padded input");
    Shape shape{x[0], w[0], (padded_h - w[2]) / strides[0] + 1, (padded_w - w[3]) / strides[1] + 1};
    std::vector<float> output(element_count(shape));
    for (std::size_t n = 0; n < x[0]; ++n)
        for (std::size_t o = 0; o < w[0]; ++o)
            for (std::size_t h = 0; h < shape[2]; ++h)
                for (std::size_t v = 0; v < shape[3]; ++v) {
                    float sum = bias ? bias->values()[o] : 0.0f;
                    for (std::size_t c = 0; c < x[1]; ++c)
                        for (std::size_t kh = 0; kh < w[2]; ++kh)
                            for (std::size_t kw = 0; kw < w[3]; ++kw) {
                                auto ih = static_cast<std::ptrdiff_t>(h * strides[0] + kh) -
                                          static_cast<std::ptrdiff_t>(padding[0]);
                                auto iw = static_cast<std::ptrdiff_t>(v * strides[1] + kw) -
                                          static_cast<std::ptrdiff_t>(padding[1]);
                                if (ih < 0 || iw < 0 || static_cast<std::size_t>(ih) >= x[2] ||
                                    static_cast<std::size_t>(iw) >= x[3])
                                    continue;
                                auto xi = ((n * x[1] + c) * x[2] + static_cast<std::size_t>(ih)) * x[3] +
                                          static_cast<std::size_t>(iw);
                                auto wi = ((o * w[1] + c) * w[2] + kh) * w[3] + kw;
                                sum += input.values()[xi] * weight.values()[wi];
                            }
                    output[((n * w[0] + o) * shape[2] + h) * shape[3] + v] = sum;
                }
    return TensorF32(std::move(output), shape);
}

TensorF32 scaled_dot_product_attention(const TensorF32 &query, const TensorF32 &key, const TensorF32 &value,
                                       const TensorF32 *mask, bool causal) {
    if (query.rank() != 4 || key.rank() != 4 || value.rank() != 4 || query.shape()[0] != key.shape()[0] ||
        query.shape()[1] != key.shape()[1] || key.shape()[0] != value.shape()[0] ||
        key.shape()[1] != value.shape()[1] || query.shape()[3] != key.shape()[3] ||
        key.shape()[2] != value.shape()[2])
        throw std::invalid_argument(
            "Attention expects compatible [batch, heads, sequence, channels] tensors");
    auto scores =
        query.matmul(key.transpose()).mul_scalar(1.0f / std::sqrt(static_cast<float>(query.shape()[3])));
    if (mask) {
        if (broadcast_shape(scores.shape(), mask->shape()) != scores.shape())
            throw std::invalid_argument("Attention mask changes score shape");
        scores = scores.add(*mask);
    }
    if (causal) {
        if (query.shape()[2] != key.shape()[2])
            throw std::invalid_argument("Causal attention requires equal query and key lengths");
        auto values = scores.values();
        auto length = query.shape()[2];
        for (std::size_t i = 0; i < values.size(); ++i)
            if (i % length > (i / length) % length)
                values[i] = -std::numeric_limits<float>::infinity();
        scores = TensorF32(std::move(values), scores.shape());
    }
    return scores.softmax().matmul(value);
}

CpuBackend::CpuBackend(Parameters parameters, const std::vector<std::string> &dense_weights)
    : parameters_(std::move(parameters)) {
    for (const auto &name : dense_weights) {
        const auto &tensor = parameter(name);
        if (tensor.rank() != 2)
            throw std::invalid_argument("Dense weight must have rank 2: " + name);
        transposed_weights_.emplace(name, tensor.transpose());
    }
}
const TensorF32 &CpuBackend::parameter(const std::string &name) const {
    auto found = parameters_.find(name);
    if (found == parameters_.end())
        throw std::invalid_argument("Missing model parameter: " + name);
    return found->second;
}
TensorF32 CpuBackend::forward(const Node &node, std::span<const TensorF32> inputs) const {
    if (inputs.size() != node.inputs.size())
        throw std::invalid_argument("Backend input count differs from node declaration");
    if (inputs.empty() && node.op != Op::Constant)
        throw std::invalid_argument("Backend node needs input tensors");
    if ((node.op == Op::Add || node.op == Op::Sub || node.op == Op::Mul || node.op == Op::MatMul ||
         node.op == Op::Gather || node.op == Op::Gemm) &&
        inputs.size() < 2)
        throw std::invalid_argument("Binary operator needs two inputs");
    if (node.op == Op::Attention && inputs.size() < 3)
        throw std::invalid_argument("Attention needs three projections");
    const auto &a = node.attributes;
    auto param = [&](std::size_t i) -> const TensorF32 & {
        if (i >= node.parameters.size())
            throw std::invalid_argument("Missing parameter name for node: " + node.id);
        return parameter(node.parameters[i]);
    };
    switch (node.op) {
    case Op::Constant:
        return param(0);
    case Op::Identity:
    case Op::Output:
    case Op::Dropout:
        return inputs[0]; // Evaluation mode.
    case Op::Dense: {
        const auto &weight = param(0);
        if (weight.rank() != 2)
            throw std::invalid_argument("Dense weight must have rank 2");
        const auto &input = inputs[0];
        if (!input.rank() || input.shape().back() != weight.shape()[1])
            throw std::invalid_argument("Dense input shape mismatch");
        const auto found = transposed_weights_.find(node.parameters[0]);
        auto weight_t = found != transposed_weights_.end() ? found->second : weight.transpose();
        auto result = input.reshape({input.size() / weight.shape()[1], weight.shape()[1]}).matmul(weight_t);
        if (node.parameters.size() > 1) {
            if (param(1).shape() != Shape{weight.shape()[0]})
                throw std::invalid_argument("Dense bias shape mismatch");
            result = result.add(param(1));
        }
        auto shape = input.shape();
        shape.back() = weight.shape()[0];
        return result.reshape(shape);
    }
    case Op::Conv2D: {
        if (node.parameters.empty() && inputs.size() < 2)
            throw std::invalid_argument("Conv2D needs weights");
        const auto &weight = node.parameters.empty() ? inputs[1] : param(0);
        const TensorF32 *bias = nullptr;
        if (node.parameters.size() > 1)
            bias = &param(1);
        else if (node.parameters.empty() && inputs.size() > 2)
            bias = &inputs[2];
        return conv2d(inputs[0], weight, bias, a.strides, a.padding);
    }
    case Op::Relu:
        return inputs[0].relu();
    case Op::Gelu:
        return inputs[0].gelu();
    case Op::Sigmoid:
        return inputs[0].sigmoid();
    case Op::Tanh:
        return inputs[0].tanh();
    case Op::Softmax:
        return inputs[0].softmax(a.axis);
    case Op::LayerNorm: {
        auto result = inputs[0].layer_norm(a.epsilon, a.shape);
        const auto *weight = !node.parameters.empty() ? &param(0) : inputs.size() > 1 ? &inputs[1] : nullptr;
        const auto *bias = node.parameters.size() > 1 ? &param(1) : inputs.size() > 2 ? &inputs[2] : nullptr;
        if (weight) {
            if (broadcast_shape(result.shape(), weight->shape()) != result.shape())
                throw std::invalid_argument("LayerNorm weight shape mismatch");
            result = result.mul(*weight);
        }
        if (bias) {
            if (broadcast_shape(result.shape(), bias->shape()) != result.shape())
                throw std::invalid_argument("LayerNorm bias shape mismatch");
            result = result.add(*bias);
        }
        return result;
    }
    case Op::MatMul:
        return inputs[0].matmul(inputs[1]);
    case Op::Add:
        return inputs[0].add(inputs[1]);
    case Op::Sub:
        return inputs[0].sub(inputs[1]);
    case Op::Mul:
        return inputs[0].mul(inputs[1]);
    case Op::Reshape: {
        Shape shape;
        std::size_t known = 1, inferred = a.target_shape.size();
        for (std::size_t i = 0; i < a.target_shape.size(); ++i) {
            auto value = a.target_shape[i];
            if (value == -1) {
                if (inferred != a.target_shape.size())
                    throw std::invalid_argument("Reshape permits only one inferred dimension");
                inferred = i;
                shape.push_back(1);
            } else {
                if (value == 0) {
                    if (i >= inputs[0].rank())
                        throw std::invalid_argument("Reshape zero dimension has no source axis");
                    value = static_cast<std::int64_t>(inputs[0].shape()[i]);
                }
                if (value < 1)
                    throw std::invalid_argument("Invalid reshape dimension");
                shape.push_back(static_cast<std::size_t>(value));
                known = checked_multiply(known, shape.back());
            }
        }
        if (inferred != a.target_shape.size()) {
            if (inputs[0].size() % known)
                throw std::invalid_argument("Reshape inferred dimension is not integral");
            shape[inferred] = inputs[0].size() / known;
        }
        return inputs[0].reshape(shape);
    }
    case Op::Transpose: {
        auto axes = a.axes;
        if (axes.empty()) {
            axes.resize(inputs[0].rank());
            std::iota(axes.rbegin(), axes.rend(), 0);
        }
        return inputs[0].permute(axes);
    }
    case Op::Embedding:
        return embedding(inputs[0], param(0));
    case Op::Gather:
        if (a.axis != 0)
            throw std::invalid_argument("Native Gather supports axis=0 only");
        return embedding(inputs[1], inputs[0]);
    case Op::Attention: {
        const auto &shape = inputs[0].shape();
        if (inputs[0].rank() != 3 || !a.heads || shape[2] % a.heads)
            throw std::invalid_argument("Attention needs [batch, sequence, channels] and a valid head count");
        auto split = [&](const TensorF32 &tensor) {
            if (tensor.rank() != 3 || tensor.shape()[0] != shape[0] || tensor.shape()[2] != shape[2])
                throw std::invalid_argument("Attention projection shape mismatch");
            return tensor.reshape({shape[0], tensor.shape()[1], a.heads, shape[2] / a.heads})
                .permute({0, 2, 1, 3});
        };
        auto result = scaled_dot_product_attention(split(inputs[0]), split(inputs[1]), split(inputs[2]),
                                                   inputs.size() > 3 ? &inputs[3] : nullptr, a.causal);
        return result.permute({0, 2, 1, 3}).reshape(shape);
    }
    case Op::Gemm: {
        if (inputs[0].rank() != 2 || inputs[1].rank() != 2)
            throw std::invalid_argument("Gemm requires rank 2");
        auto left = a.trans_a ? inputs[0].transpose() : inputs[0];
        auto right = a.trans_b ? inputs[1].transpose() : inputs[1];
        auto result = left.matmul(right).mul_scalar(a.alpha);
        if (inputs.size() > 2) {
            if (broadcast_shape(result.shape(), inputs[2].shape()) != result.shape())
                throw std::invalid_argument("Gemm bias changes output shape");
            result = result.add(inputs[2].mul_scalar(a.beta));
        }
        return result;
    }
    case Op::Input:
        break;
    }
    throw std::invalid_argument("Unsupported backend operator: " + std::string(op_name(node.op)));
}
} // namespace octopus
