# Benchmark methodology

Build in Release mode before collecting results. Run `octopus benchmark --iterations 1000 --size 64` or `inference_bench 1000` on an otherwise idle machine.

Each case performs three warmup calls. Measured calls use initialized input data, execute the real kernel or complete sample graph, and consume an output value. The benchmark reports median and p95 latency in milliseconds, throughput in completed calls per second, and a checksum. Throughput uses the sum of measured durations and excludes warmup.

Cases are matrix multiplication, ReLU, last-axis Softmax, and a deterministic single-request MLP. The executable driver uses matrix size 32; the CLI accepts sizes from 1 to 1024. The small MLP is an integration baseline rather than a representative language-model throughput workload.

Record these fields when publishing results:

1. Commit SHA and operating system.
2. CPU model, memory, and power configuration.
3. Compiler version, optimization flags, and build configuration.
4. Tensor shapes, dtype, iteration count, and warmup count.
5. Whether timings include allocation, graph dispatch, or transport overhead.
6. Thread settings and the exact command used.

The current CPU kernels are sequential. Concurrent HTTP request throughput is a separate measurement and should not be described as graph-level operator parallelism.

For comparisons with NumPy, PyTorch, ONNX Runtime, or another runtime, use identical numerical operations and precision, equivalent threading settings, and validated outputs. Publish the comparison script and raw results. No cross-runtime speedup is asserted in this repository.
