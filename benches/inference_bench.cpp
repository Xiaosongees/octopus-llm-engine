#include "octopus/benchmark.hpp"
#include <charconv>
#include <iostream>
#include <string_view>

int main(int argc, char **argv) {
    try {
        std::size_t iterations = 100;
        if (argc > 2)
            throw std::invalid_argument("Usage: inference_bench [iterations]");
        if (argc == 2) {
            std::string_view text(argv[1]);
            auto result = std::from_chars(text.data(), text.data() + text.size(), iterations);
            if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
                throw std::invalid_argument("Invalid iteration count");
        }
        octopus::benchmark(std::cout, iterations);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
