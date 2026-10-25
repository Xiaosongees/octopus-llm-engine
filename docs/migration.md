# Migration from the Rust prototype

Version 0.2 replaces the Rust implementation with a C++20 library and CMake build. Cargo files, Rust sources, and the previous hard-coded chart asset have been removed. Git history preserves the original implementation.

| Previous concept | C++ replacement |
| --- | --- |
| Cargo build/test | CMake build and CTest |
| Tensor/ndarray | Immutable `octopus::Tensor<T>` with contiguous shared storage |
| GraphBuilder | `octopus::GraphBuilder` and validated `octopus::Graph` |
| Mutable engine input/output cache | `InferenceEngine::run(input_map)` returning an independent result |
| Implicit empty backend parameters | Explicit graph and complete parameter map at construction |
| Axum server | Optional cpp-httplib server using the same immutable engine |
| Criterion | Release-mode benchmark executable with measured latency and checksums |
| Placeholder ONNX loader | Bounded protobuf importer with explicit operator/version support |

CLI commands `info`, `benchmark`, and `serve` remain available, with updated options. The server requires `--sample` or `--model FILE`, accepts `--host` and `--port`, and binds to localhost by default. See `octopus --help` for the complete interface.

The HTTP endpoint paths remain `/v1/health`, `/v1/model/info`, and `/v1/inference`. Single-input inference uses the actual `input_data`/`input_shape` fields, and the new `inputs` object supports multiple named tensors. Errors now use appropriate client/server status codes.

This is an API-breaking rewrite. The Rust builder signatures and library imports are not preserved. No pretrained-model compatibility is implied by the migration. Placeholder GPU and quantization claims have been removed from the documentation.
