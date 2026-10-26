# Octopus LLM Engine

A C++20 inference engine for studying and extending tensor operations, computation graphs, and Transformer execution on the CPU.

The project provides a reusable C++ library, a command-line application, a local HTTP API, numerical tests, and reproducible benchmark commands. It is an experimental inference runtime, not a pretrained chat application.

## Implemented capabilities

| Area | Current implementation |
| --- | --- |
| Tensors | Immutable shared storage, float32/float64, reshape, broadcasting, transpose, batched matrix multiplication, reductions |
| Operators | Dense, NCHW Conv2D, ReLU, tanh-approximate GELU, Sigmoid, Tanh, Softmax, LayerNorm, Embedding, multi-head attention |
| Graph execution | Dependency validation, cycle detection, deterministic topological ordering, parameter loading, per-run statistics, intermediate lifetime tracking |
| Transformer | Token and learned position embeddings, pre-normalized blocks, residual connections, feed-forward networks, optional causal masking |
| Model files | Versioned native JSON format and an explicitly supported ONNX protobuf subset |
| Interfaces | C++ library, CLI, HTTP health/model/inference endpoints |
| Verification | CTest, numerical regression tests, CLI/model tests, HTTP/concurrency tests, optional ONNX Runtime comparison |

CUDA, quantization, KV caching, operator fusion, tokenizers, pretrained checkpoint adapters, and autoregressive text generation are **not implemented**. The CPU kernels are portable reference implementations; no performance advantage over other runtimes is claimed. See [supported formats and limitations](docs/model-formats.md).

## Build

Requirements:

- A C++20 compiler: GCC 11+, Clang 14+, or a recent MSVC toolset.
- CMake 3.24 or newer.
- Python 3.9+ when building tests; it is used only for integration tests.
- Internet access on the first configuration to fetch two checksum-pinned headers, or an offline dependency directory.

```sh
git clone https://github.com/Xiaosongees/octopus-llm-engine.git
cd octopus-llm-engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

On Linux and macOS, executables are normally placed directly in `build/`. With the Visual Studio generator on Windows, use `build/Release/octopus.exe`, `build/Release/basic_inference.exe`, and `build/Release/inference_bench.exe`. The examples below use the Linux/macOS executable paths.

For an offline build, place the exact headers listed in [third-party dependencies](THIRD_PARTY.md) at `DEPS/nlohmann/json.hpp` and `DEPS/httplib.h`, then configure with:

```sh
cmake -S . -B build -DOCTOPUS_DEPENDENCY_DIR=/absolute/path/to/DEPS
```

Build options:

| Option | Default | Purpose |
| --- | --- | --- |
| `OCTOPUS_BUILD_TESTS` | `ON` | Build and register tests; requires Python |
| `OCTOPUS_BUILD_SERVER` | `ON` | Build HTTP support; disable for a smaller CLI/library build |
| `OCTOPUS_ENABLE_SANITIZERS` | `OFF` | Enable AddressSanitizer and UndefinedBehaviorSanitizer with GCC/Clang |
| `OCTOPUS_DEPENDENCY_DIR` | Empty | Use pre-downloaded, checksum-verified headers |

## Run a complete inference example

```sh
./build/octopus info
./build/octopus demo
./build/basic_inference
```

The deterministic sample model computes `ReLU(x W^T + b)`. For input `[1, 2, 3, 4]`, its output is `[1.5, 1.5, 8.0]` with shape `[1, 3]`. Real weights are loaded before inference.

```sh
./build/octopus make-model --output sample.json
./build/octopus infer --model sample.json --input examples/input.json
./build/octopus transformer-demo --seed 42 --causal
```

The Transformer demo runs a small randomly initialized encoder. Its output is a tensor of hidden states, not generated text. Repeated runs with the same seed and toolchain are deterministic; random distribution implementations may differ across standard libraries.

## Use the C++ library

Link your CMake target against `octopus::engine` after adding this project with `add_subdirectory`.

```cpp
#include <octopus/inference.hpp>
#include <iostream>

int main() {
    using namespace octopus;
    GraphBuilder graph;
    graph.input("x", {2, 2});
    graph.dense("fc", "x");
    graph.output("y", "fc");

    Parameters weights{
        {"fc.weight", TensorF32({1, 0, 0, 1}, {2, 2})},
        {"fc.bias", TensorF32({1, 2}, {2})}
    };
    InferenceEngine engine(graph.build(), std::move(weights));
    auto output = engine.infer("x", TensorF32({3, 4, 5, 6}, {2, 2}), "y");
    for (float value : output.values()) std::cout << value << ' ';
    // Expected: 4 6 6 8
}
```

The loaded model and its parameters are immutable. `run()` takes a complete named input map and returns a new output map and statistics. Concurrent calls do not share mutable input or output caches. See [architecture](docs/architecture.md).

## HTTP API

```sh
./build/octopus serve --sample --host 127.0.0.1 --port 8080 --threads 4
# Alternatively:
./build/octopus serve --model sample.json
```

The server defaults to localhost and requires an explicitly selected model. Its bounded worker pool allows independent requests to run concurrently. Request bodies are limited to 1 MiB.

```sh
curl http://127.0.0.1:8080/v1/health
curl http://127.0.0.1:8080/v1/model/info
curl -X POST http://127.0.0.1:8080/v1/inference \
  -H 'Content-Type: application/json' \
  --data-binary @examples/input.json
```

The response includes:

```json
{
  "output_data": [1.5, 1.5, 8.0],
  "output_shape": [1, 3],
  "backend": "cpu"
}
```

It also contains measured `inference_time_ms` and all named `outputs`. Invalid shapes, missing inputs, and malformed requests receive HTTP 400. Incorrect content types receive 415 and oversized bodies receive 413. See [the API contract](docs/http-api.md).

The HTTP interface is intended for trusted local development. Authentication, TLS termination, deployment-wide admission control, and tenant isolation must be provided by a deployment layer before exposing it publicly.

## Benchmark

```sh
./build/inference_bench 1000
./build/octopus benchmark --iterations 1000 --size 64
```

Both commands warm up each case, execute real calculations, consume the outputs, and report median latency, p95 latency, throughput, and a checksum. Results depend on compiler settings and hardware. See [benchmark methodology](docs/benchmarks.md) for collecting comparable measurements.

## Tests

```sh
ctest --test-dir build -C Release --output-on-failure
```

The tests cover tensor arithmetic, broadcasting, batch dimensions, numerical stability, convolution, masking, graph cycles and repeated edges, real parameter loading, deterministic Transformer execution, model serialization, ONNX import, CLI errors, HTTP status codes, and concurrent request isolation.

For optional differential checks against ONNX Runtime:

```sh
python -m pip install numpy==2.2.6 onnx==1.18.0 onnxruntime==1.22.0
python tests/test_onnx_reference.py ./build/octopus
```

The optional packages are test-only dependencies. An optional GitHub Actions configuration for Windows, Linux, and sanitizers is supplied in `ci/github-actions.yml`. See [CI setup](docs/ci.md) to activate it with a credential authorized to manage workflows.

## Repository layout

```text
include/octopus/   Public tensor, graph, backend, model, and inference APIs
src/              CPU kernels, graph runtime, model import/export, CLI, HTTP server
examples/         Runnable C++ example and JSON request
benches/          Executable benchmark driver
tests/            C++ numerical tests and Python integration/reference checks
cmake/            Checksum-pinned dependency setup
docs/             Architecture, model format, API, and benchmark documentation
website/          Static English project overview
ci/               Optional cross-platform build, test, and sanitizer configuration
```

## Development direction

Near-term work should focus on larger reference-model coverage, optimized CPU matrix kernels, dynamic shapes, integer tensor inputs, and pretrained checkpoint adapters. GPU execution, quantization, and KV caching should be added behind explicit capability checks and validated against a numerical reference.

See [contribution guidelines](CONTRIBUTING.md). This C++ rewrite replaces the original Rust implementation and is an API-breaking change; see [migration notes](docs/migration.md).

## License

The project is licensed under the [MIT License](LICENSE). External dependency licenses are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
