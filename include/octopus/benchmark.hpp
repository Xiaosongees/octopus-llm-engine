#pragma once
#include "octopus/inference.hpp"
#include <atomic>
#include <chrono>
#include <iomanip>
#include <ostream>

namespace octopus {
inline void benchmark(std::ostream &output, std::size_t iterations = 100, std::size_t matrix_size = 32) {
    if (!iterations || iterations > 1000000 || !matrix_size || matrix_size > 1024)
        throw std::invalid_argument("Benchmark iterations must be 1-1000000 and matrix size 1-1024");
    auto a = TensorF32::random({matrix_size, matrix_size}, 1);
    auto b = TensorF32::random({matrix_size, matrix_size}, 2);
    auto engine = make_sample_engine();
    TensorMap inputs{{"input", TensorF32({1, 2, 3, 4}, {1, 4})}};
    auto measure = [&](const std::string &name, const std::function<TensorF32()> &function) {
        for (int i = 0; i < 3; ++i)
            (void)function();
        std::vector<double> samples;
        samples.reserve(iterations);
        double checksum = 0;
        for (std::size_t i = 0; i < iterations; ++i) {
            auto start = std::chrono::steady_clock::now();
            auto result = function();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            samples.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            checksum += result.values().front();
        }
        if (!std::isfinite(checksum))
            throw std::runtime_error("Non-finite benchmark output");
        auto total = std::accumulate(samples.begin(), samples.end(), 0.0);
        std::sort(samples.begin(), samples.end());
        auto p95 = samples[static_cast<std::size_t>(std::ceil(static_cast<double>(iterations) * 0.95)) - 1];
        output << name << ": median_ms=" << samples[iterations / 2] << " p95_ms=" << p95
               << " throughput_per_sec=" << static_cast<double>(iterations) * 1000 / std::max(total, 1e-9)
               << " checksum=" << checksum << '\n';
    };
    output << std::fixed << std::setprecision(6) << "Octopus CPU benchmark; iterations=" << iterations
           << "; matrix_size=" << matrix_size << "; warmup=3\n";
    measure("matmul", [&] { return a.matmul(b); });
    measure("relu", [&] { return a.relu(); });
    measure("softmax", [&] { return a.softmax(); });
    measure("sample_mlp", [&] { return engine.run(inputs).outputs.at("output"); });
}
} // namespace octopus
