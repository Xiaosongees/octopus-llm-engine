# Third-party dependencies

The production implementation uses two upstream single-header libraries. CMake downloads them into the build directory and verifies their SHA-256 digests on every configuration. Header copyright notices remain intact. No third-party source is copied into the repository.

| Dependency | Version | License | Purpose |
| --- | --- | --- | --- |
| [JSON for Modern C++](https://github.com/nlohmann/json/tree/v3.12.0) | 3.12.0 | MIT | Native model files and HTTP JSON |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib/tree/v0.63.0) | 0.63.0 | MIT | Optional HTTP transport |

## Exact offline files

- `nlohmann/json.hpp`: https://raw.githubusercontent.com/nlohmann/json/v3.12.0/single_include/nlohmann/json.hpp
  - SHA-256: `aaf127c04cb31c406e5b04a63f1ae89369fccde6d8fa7cdda1ed4f32dfc5de63`
- `httplib.h`: https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.63.0/httplib.h
  - SHA-256: `9021001a719768893a8a6e4051c56107690e8b8e49b41e5f903534fa19ae9afb`

Python's standard library is used for integration tests. NumPy, ONNX, and ONNX Runtime are optional test-only dependencies for differential numerical validation; they are not linked into the engine.

The ONNX reader implements protobuf field encodings described by the [ONNX 1.17 schema](https://github.com/onnx/onnx/blob/v1.17.0/onnx/onnx.proto3). It does not vendor or link ONNX or protobuf code.
