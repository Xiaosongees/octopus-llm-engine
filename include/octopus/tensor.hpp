#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace octopus {

using Shape = std::vector<std::size_t>;

inline std::size_t checked_multiply(std::size_t a, std::size_t b) {
    if (b && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("Tensor dimensions overflow size_t");
    return a * b;
}

inline std::size_t element_count(const Shape &shape) {
    std::size_t result = 1;
    for (auto dim : shape) {
        if (!dim)
            throw std::invalid_argument("Tensor dimensions must be positive");
        result = checked_multiply(result, dim);
    }
    return result;
}

inline Shape strides_for(const Shape &shape) {
    Shape result(shape.size(), 1);
    for (std::size_t i = shape.size(); i > 1; --i)
        result[i - 2] = checked_multiply(result[i - 1], shape[i - 1]);
    return result;
}

inline Shape broadcast_shape(const Shape &a, const Shape &b) {
    Shape result(std::max(a.size(), b.size()), 1);
    for (std::size_t i = 0; i < result.size(); ++i) {
        auto x = i < a.size() ? a[a.size() - 1 - i] : 1;
        auto y = i < b.size() ? b[b.size() - 1 - i] : 1;
        if (x != y && x != 1 && y != 1)
            throw std::invalid_argument("Tensor shapes cannot be broadcast together");
        result[result.size() - 1 - i] = std::max(x, y);
    }
    return result;
}

inline std::size_t broadcast_offset(std::size_t index, const Shape &output, const Shape &input,
                                    const Shape &strides) {
    std::size_t offset = 0;
    for (std::size_t i = output.size(); i > 0; --i) {
        auto coord = index % output[i - 1];
        index /= output[i - 1];
        if (i > output.size() - input.size()) {
            auto j = i - 1 - (output.size() - input.size());
            if (input[j] != 1)
                offset += coord * strides[j];
        }
    }
    return offset;
}

// Immutable, contiguous tensors share storage when copied or reshaped.
template <typename T = float> class Tensor {
    static_assert(std::is_floating_point_v<T>, "Tensor supports floating-point storage");

  public:
    Tensor() : Tensor(std::vector<T>{T{0}}, Shape{}) {}
    Tensor(std::vector<T> values, Shape shape) : shape_(std::move(shape)) {
        if (values.size() != element_count(shape_))
            throw std::invalid_argument("Tensor data length does not match its shape");
        values_ = std::make_shared<const std::vector<T>>(std::move(values));
    }
    static Tensor zeros(const Shape &shape) {
        return Tensor(std::vector<T>(element_count(shape), T{0}), shape);
    }
    static Tensor ones(const Shape &shape) {
        return Tensor(std::vector<T>(element_count(shape), T{1}), shape);
    }
    static Tensor random(const Shape &shape, std::uint32_t seed = 42, T scale = T{1}) {
        std::mt19937 generator(seed);
        std::uniform_real_distribution<T> dist(-scale, scale);
        std::vector<T> values(element_count(shape));
        for (auto &value : values)
            value = dist(generator);
        return Tensor(std::move(values), shape);
    }
    const Shape &shape() const noexcept { return shape_; }
    const std::vector<T> &values() const noexcept { return *values_; }
    std::size_t size() const noexcept { return values_->size(); }
    std::size_t rank() const noexcept { return shape_.size(); }
    T at(std::span<const std::size_t> indices) const {
        if (indices.size() != rank())
            throw std::invalid_argument("Tensor index rank mismatch");
        std::size_t offset = 0;
        for (std::size_t i = 0; i < rank(); ++i) {
            if (indices[i] >= shape_[i])
                throw std::out_of_range("Tensor index is out of bounds");
            offset = offset * shape_[i] + indices[i];
        }
        return values()[offset];
    }
    Tensor reshape(const Shape &shape) const {
        if (element_count(shape) != size())
            throw std::invalid_argument("Reshape changes the element count");
        return Tensor(values_, shape);
    }
    Tensor map(const std::function<T(T)> &function) const {
        std::vector<T> result(size());
        std::transform(values().begin(), values().end(), result.begin(), function);
        return Tensor(std::move(result), shape_);
    }
    Tensor binary(const Tensor &other, const std::function<T(T, T)> &function) const {
        auto output = broadcast_shape(shape_, other.shape_);
        auto a_strides = strides_for(shape_), b_strides = strides_for(other.shape_);
        std::vector<T> result(element_count(output));
        for (std::size_t i = 0; i < result.size(); ++i)
            result[i] = function(values()[broadcast_offset(i, output, shape_, a_strides)],
                                 other.values()[broadcast_offset(i, output, other.shape_, b_strides)]);
        return Tensor(std::move(result), output);
    }
    Tensor add(const Tensor &other) const { return binary(other, std::plus<T>{}); }
    Tensor sub(const Tensor &other) const { return binary(other, std::minus<T>{}); }
    Tensor mul(const Tensor &other) const { return binary(other, std::multiplies<T>{}); }
    Tensor add_scalar(T scalar) const {
        return map([=](T x) { return x + scalar; });
    }
    Tensor mul_scalar(T scalar) const {
        return map([=](T x) { return x * scalar; });
    }
    Tensor relu() const {
        return map([](T x) { return std::max(T{0}, x); });
    }
    Tensor gelu() const {
        return map([](T x) {
            return T{0.5} * x *
                   (T{1} + std::tanh(static_cast<T>(0.7978845608028654) *
                                     (x + static_cast<T>(0.044715) * x * x * x)));
        });
    }
    Tensor sigmoid() const {
        return map([](T x) {
            if (x >= 0)
                return T{1} / (T{1} + std::exp(-x));
            auto e = std::exp(x);
            return e / (T{1} + e);
        });
    }
    Tensor tanh() const {
        return map([](T x) { return std::tanh(x); });
    }
    Tensor permute(const Shape &axes) const {
        if (axes.size() != rank())
            throw std::invalid_argument("Transpose permutation rank mismatch");
        std::vector<bool> seen(rank(), false);
        Shape output(rank());
        for (std::size_t i = 0; i < rank(); ++i) {
            if (axes[i] >= rank() || seen[axes[i]])
                throw std::invalid_argument("Invalid transpose permutation");
            seen[axes[i]] = true;
            output[i] = shape_[axes[i]];
        }
        auto strides = strides_for(shape_);
        std::vector<T> result(size());
        for (std::size_t i = 0; i < size(); ++i) {
            auto remaining = i;
            std::size_t offset = 0;
            for (std::size_t j = rank(); j > 0; --j) {
                offset += (remaining % output[j - 1]) * strides[axes[j - 1]];
                remaining /= output[j - 1];
            }
            result[i] = values()[offset];
        }
        return Tensor(std::move(result), output);
    }
    Tensor transpose() const {
        if (rank() < 2)
            throw std::invalid_argument("Transpose requires rank >= 2");
        Shape axes(rank());
        std::iota(axes.begin(), axes.end(), 0);
        std::swap(axes[rank() - 2], axes[rank() - 1]);
        return permute(axes);
    }
    // Broadcast leading batch axes; multiply the final two axes.
    Tensor matmul(const Tensor &other) const {
        if (rank() < 2 || other.rank() < 2)
            throw std::invalid_argument("MatMul requires rank >= 2");
        auto m = shape_[rank() - 2], k = shape_.back();
        auto n = other.shape_.back();
        if (k != other.shape_[other.rank() - 2])
            throw std::invalid_argument("MatMul inner dimensions differ");
        Shape a_batch(shape_.begin(), shape_.end() - 2),
            b_batch(other.shape_.begin(), other.shape_.end() - 2);
        auto batch = broadcast_shape(a_batch, b_batch);
        auto a_strides = strides_for(a_batch), b_strides = strides_for(b_batch);
        auto output = batch;
        output.push_back(m);
        output.push_back(n);
        std::vector<T> result(element_count(output), T{0});
        for (std::size_t b = 0; b < element_count(batch); ++b) {
            auto a_base = broadcast_offset(b, batch, a_batch, a_strides) * m * k;
            auto b_base = broadcast_offset(b, batch, b_batch, b_strides) * k * n;
            auto out_base = b * m * n;
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t p = 0; p < k; ++p) {
                    const T a = values()[a_base + i * k + p];
                    for (std::size_t j = 0; j < n; ++j)
                        result[out_base + i * n + j] += a * other.values()[b_base + p * n + j];
                }
        }
        return Tensor(std::move(result), output);
    }
    Tensor softmax(int axis = -1) const {
        auto selected = normalize_axis(axis);
        const auto width = shape_[selected];
        std::size_t inner = 1;
        for (std::size_t i = selected + 1; i < rank(); ++i)
            inner *= shape_[i];
        const auto outer = size() / width / inner;
        std::vector<T> result(size());
        for (std::size_t o = 0; o < outer; ++o)
            for (std::size_t j = 0; j < inner; ++j) {
                T maximum = -std::numeric_limits<T>::infinity();
                for (std::size_t w = 0; w < width; ++w) {
                    auto x = values()[(o * width + w) * inner + j];
                    if (std::isnan(x) || x == std::numeric_limits<T>::infinity())
                        throw std::invalid_argument("Softmax does not accept NaN or positive infinity");
                    maximum = std::max(maximum, x);
                }
                if (!std::isfinite(maximum))
                    throw std::invalid_argument("Softmax row is fully masked");
                T sum = 0;
                for (std::size_t w = 0; w < width; ++w) {
                    auto index = (o * width + w) * inner + j;
                    result[index] = std::exp(values()[index] - maximum);
                    sum += result[index];
                }
                for (std::size_t w = 0; w < width; ++w)
                    result[(o * width + w) * inner + j] /= sum;
            }
        return Tensor(std::move(result), shape_);
    }
    Tensor layer_norm(T epsilon = static_cast<T>(1e-5), Shape normalized_shape = {}) const {
        if (!rank())
            throw std::invalid_argument("LayerNorm requires rank >= 1");
        if (!std::isfinite(epsilon) || epsilon <= 0)
            throw std::invalid_argument("LayerNorm epsilon must be positive");
        if (normalized_shape.empty())
            normalized_shape = {shape_.back()};
        if (normalized_shape.size() > rank() ||
            !std::equal(normalized_shape.rbegin(), normalized_shape.rend(), shape_.rbegin()))
            throw std::invalid_argument("LayerNorm shape must match trailing dimensions");
        const auto width = element_count(normalized_shape);
        std::vector<T> result(size());
        for (std::size_t base = 0; base < size(); base += width) {
            double mean = 0, variance = 0;
            for (std::size_t i = 0; i < width; ++i)
                mean += values()[base + i];
            mean /= static_cast<double>(width);
            for (std::size_t i = 0; i < width; ++i) {
                auto d = values()[base + i] - mean;
                variance += d * d;
            }
            const auto denominator = std::sqrt(variance / static_cast<double>(width) + epsilon);
            for (std::size_t i = 0; i < width; ++i)
                result[base + i] = static_cast<T>((values()[base + i] - mean) / denominator);
        }
        return Tensor(std::move(result), shape_);
    }
    Tensor sum_axis(int axis) const {
        auto selected = normalize_axis(axis);
        Shape output = shape_;
        output.erase(output.begin() + static_cast<std::ptrdiff_t>(selected));
        auto input_strides = strides_for(shape_);
        std::vector<T> result(element_count(output), 0);
        for (std::size_t i = 0; i < size(); ++i) {
            std::size_t index = 0;
            for (std::size_t j = 0; j < rank(); ++j)
                if (j != selected)
                    index = index * shape_[j] + (i / input_strides[j]) % shape_[j];
            result[index] += values()[i];
        }
        return Tensor(std::move(result), output);
    }

  private:
    Tensor(std::shared_ptr<const std::vector<T>> values, Shape shape)
        : shape_(std::move(shape)), values_(std::move(values)) {}
    std::size_t normalize_axis(int axis) const {
        auto value = static_cast<std::int64_t>(axis);
        if (value < 0)
            value += static_cast<std::int64_t>(rank());
        if (value < 0 || value >= static_cast<std::int64_t>(rank()))
            throw std::invalid_argument("Tensor axis is out of range");
        return static_cast<std::size_t>(value);
    }
    Shape shape_;
    std::shared_ptr<const std::vector<T>> values_;
};

using TensorF32 = Tensor<float>;
using TensorF64 = Tensor<double>;

} // namespace octopus
