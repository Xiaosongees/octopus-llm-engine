#include "json_io.hpp"
#include "octopus/benchmark.hpp"
#include "octopus/model_io.hpp"
#ifdef OCTOPUS_HAS_SERVER
#include "octopus/server.hpp"
#endif
#include <charconv>
#include <fstream>
#include <iostream>
#include <set>
#include <thread>

namespace {
using namespace octopus;
void help() {
    std::cout << "Octopus LLM Engine " << OCTOPUS_VERSION
              << " (C++20, CPU)\n\n"
                 "Usage: octopus <command> [options]\n\n"
                 "  info                     Show engine capabilities\n"
                 "  demo                     Run the deterministic sample MLP\n"
                 "  transformer-demo         Run a randomly initialized tiny encoder\n"
                 "  make-model --output FILE Save the sample MLP as a native JSON model\n"
                 "  infer --model FILE --input FILE\n"
                 "                           Run a JSON or supported ONNX model\n"
                 "  benchmark [--iterations 100] [--size 32]\n"
                 "                           Measure CPU operators and model inference\n"
                 "  serve (--sample | --model FILE) [--host 127.0.0.1] [--port 8080]\n"
                 "        [--threads 4]      Start the HTTP API (port 0 selects a free port)\n\n"
                 "Global option: --backend cpu. CUDA is explicitly unsupported.\n"
                 "Transformer demo options: --seed 42 --causal\n"
                 "Use --help to show this message.\n";
}
struct Options {
    std::map<std::string, std::string> values;
    bool has(const std::string &key) const { return values.contains(key); }
    std::string get(const std::string &key, const std::string &fallback = "") const {
        auto i = values.find(key);
        return i == values.end() ? fallback : i->second;
    }
    std::size_t number(const std::string &key, std::size_t fallback, std::size_t maximum) const {
        if (!has(key))
            return fallback;
        auto text = get(key);
        std::size_t value = 0;
        auto result = std::from_chars(text.data(), text.data() + text.size(), value);
        if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value > maximum)
            throw std::invalid_argument("Invalid numeric option: " + key);
        return value;
    }
    std::string required(const std::string &key) const {
        if (!has(key) || get(key).empty())
            throw std::invalid_argument("Required option: " + key);
        return get(key);
    }
};
ModelDefinition sample_model() {
    GraphBuilder graph;
    graph.input("input", {1, 4});
    graph.dense("fc1", "input");
    graph.unary("relu1", Op::Relu, "fc1");
    graph.output("output", "relu1");
    return {graph.build(),
            {{"fc1.weight", TensorF32({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1}, {3, 4})},
             {"fc1.bias", TensorF32({0.5f, -0.5f, 1}, {3})}}};
}
} // namespace
int main(int argc, char **argv) {
    using namespace octopus;
    try {
        if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
            help();
            return 0;
        }
        std::string command = argv[1];
        std::map<std::string, std::set<std::string>> commands{
            {"info", {}},
            {"demo", {}},
            {"transformer-demo", {"--seed", "--causal"}},
            {"make-model", {"--output"}},
            {"infer", {"--model", "--input"}},
            {"benchmark", {"--iterations", "--size"}},
            {"serve", {"--model", "--sample", "--host", "--port", "--threads"}}};
        if (!commands.contains(command))
            throw std::invalid_argument("Unknown command: " + command);
        auto allowed = commands.at(command);
        allowed.insert("--backend");
        Options options;
        for (int i = 2; i < argc; ++i) {
            std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                help();
                return 0;
            }
            if (!allowed.contains(key) || options.has(key))
                throw std::invalid_argument("Unknown or duplicate option: " + key);
            if (key == "--sample" || key == "--causal")
                options.values.emplace(key, "true");
            else {
                if (++i >= argc)
                    throw std::invalid_argument("Missing value for option: " + key);
                options.values.emplace(key, argv[i]);
            }
        }
        if (parse_backend(options.get("--backend", "cpu")) != BackendType::Cpu)
            throw std::invalid_argument("CUDA is not implemented; use --backend cpu");
        if (command == "info") {
            std::cout << "Octopus LLM Engine " << OCTOPUS_VERSION << "\nLanguage: C++20\nBackend: CPU\n"
                      << "Hardware concurrency: " << std::thread::hardware_concurrency() << "\n"
                      << "Storage: float32 and float64 tensors; float32 graph execution\n"
                      << "Models: native JSON and a validated ONNX subset\n"
                      << "CUDA, quantization, KV cache, and pretrained text generation: not implemented\n";
        } else if (command == "make-model") {
            save_model(sample_model(), options.required("--output"));
            std::cout << "Saved sample model to " << options.get("--output") << '\n';
        } else if (command == "demo") {
            auto engine = make_sample_engine();
            auto result = engine.run({{"input", TensorF32({1, 2, 3, 4}, {1, 4})}});
            std::cout << detail::encode_result(result, "output").dump(2) << '\n';
        } else if (command == "transformer-demo") {
            TransformerConfig config;
            config.d_model = 16;
            config.n_heads = 4;
            config.n_layers = 1;
            config.d_ff = 32;
            config.vocab_size = 32;
            config.causal = options.has("--causal");
            auto model = TransformerEncoder(config).build(
                1, 4,
                static_cast<std::uint32_t>(
                    options.number("--seed", 42, std::numeric_limits<std::uint32_t>::max())));
            auto engine = InferenceEngine(std::move(model.graph), std::move(model.parameters));
            auto result = engine.run({{"tokens", TensorF32({1, 2, 3, 4}, {1, 4})}});
            std::cout << detail::encode_result(result, "output").dump(2) << '\n';
        } else if (command == "benchmark") {
            benchmark(std::cout, options.number("--iterations", 100, 1000000),
                      options.number("--size", 32, 1024));
        } else if (command == "infer") {
            auto model = load_model(options.required("--model"));
            InferenceEngine engine(std::move(model.graph), std::move(model.parameters));
            auto path = options.required("--input");
            if (std::filesystem::file_size(path) > 1024 * 1024)
                throw std::invalid_argument("Input file exceeds 1 MiB");
            std::ifstream input(path);
            if (!input)
                throw std::runtime_error("Cannot read input file");
            std::string body((std::istreambuf_iterator<char>(input)), {});
            auto request = detail::parse_request(body);
            auto output = request.value("output_name", engine.graph().output_names().front());
            std::cout << detail::encode_result(engine.run(detail::decode_inputs(request)), output).dump(2)
                      << '\n';
        } else if (command == "serve") {
#ifdef OCTOPUS_HAS_SERVER
            if (options.has("--sample") == options.has("--model"))
                throw std::invalid_argument("Select exactly one of --sample and --model FILE");
            auto engine = [&] {
                if (options.has("--sample"))
                    return make_sample_engine();
                auto model = load_model(options.required("--model"));
                return InferenceEngine(std::move(model.graph), std::move(model.parameters));
            }();
            serve(engine, options.get("--host", "127.0.0.1"),
                  static_cast<int>(options.number("--port", 8080, 65535)),
                  options.number("--threads", 4, 64));
#else
            throw std::invalid_argument("HTTP support was disabled at build time");
#endif
        }
        return 0;
    } catch (const std::exception &exception) {
        std::cerr << "Error: " << exception.what() << '\n';
        return 1;
    }
}
