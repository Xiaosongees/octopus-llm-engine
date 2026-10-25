#include "octopus/inference.hpp"
#include <cmath>
#include <iostream>

int main() {
    try {
        auto engine = octopus::make_sample_engine();
        auto output = engine.infer("input", octopus::TensorF32({1, 2, 3, 4}, {1, 4}));
        const std::vector<float> expected{1.5f, 1.5f, 8.0f};
        if (output.shape() != octopus::Shape{1, 3})
            throw std::runtime_error("Unexpected output shape");
        for (std::size_t i = 0; i < expected.size(); ++i)
            if (std::abs(output.values()[i] - expected[i]) > 1e-6f)
                throw std::runtime_error("Unexpected model output");
        std::cout << "Sample MLP output: [";
        for (std::size_t i = 0; i < output.size(); ++i)
            std::cout << (i ? ", " : "") << output.values()[i];
        std::cout << "]\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
