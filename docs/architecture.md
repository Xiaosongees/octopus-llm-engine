# Architecture

## Data model

`Tensor<T>` stores a contiguous row-major array in an immutable `shared_ptr<const vector<T>>`. Copying a tensor or reshaping it shares initialized storage. Arithmetic allocates a new result. No public uninitialized-memory constructor exists. Dimensions are positive, scalar tensors use an empty shape, and element-count overflow is rejected before allocation.

Tensor operations support float32 and float64. Graph execution uses float32 tensors. Embedding indices are represented as float32 numbers and must be finite, integral, and in range. Transformer vocabulary and position counts cannot exceed 2^24, the range in which consecutive integer IDs remain exactly representable.

## Graph preparation

`GraphBuilder` creates named nodes, parameter references, and input declarations. `Graph` rejects duplicate identifiers, unresolved edges, invalid operator arities, and dependency cycles. Kahn's algorithm uses adjacency lists and preserves repeated edges, including `Add(x, x)`. Construction costs O(V + E).

`InferenceEngine` receives a complete graph and parameter map. Missing parameters fail at construction. Operator-specific weight and intermediate shapes are checked when the relevant kernel executes. CPU weights are immutable after loading. CUDA selection throws an explicit unsupported-backend error.

## Execution

Each `run()` validates a complete input map, allocates its own intermediate map, and executes nodes in a deterministic topological order. Reference counts identify the last use of each intermediate so it can be released during execution. Outputs and timings belong to the returned `InferenceResult`.

The graph itself executes sequentially. The implementation does not claim graph-level parallel scheduling, SIMD-specific kernels, BLAS acceleration, or operator fusion. Independent HTTP requests can execute concurrently because the loaded engine is immutable and scratch buffers are local to each run.

## Numerical implementation

- MatMul broadcasts leading batch axes and multiplies the final two axes. Rank-one vector MatMul is not supported.
- Dense flattens leading input dimensions, multiplies by a transposed rank-two weight, broadcasts bias, and restores the leading shape.
- Softmax subtracts the row maximum and supports any valid axis. NaN, positive infinity, and fully masked rows are rejected.
- LayerNorm normalizes trailing dimensions using population variance and supports broadcast affine parameters.
- Conv2D uses direct NCHW/OIHW convolution with symmetric 2D padding and positive strides. Groups and dilation are not implemented.
- Multi-head attention splits `[batch, sequence, channels]` projections into heads, computes scaled attention, applies additive/causal masks, and concatenates heads.
- Dropout is an identity operation in evaluation mode; training is outside the scope of this runtime.

## Interfaces

The CLI and HTTP server share request validation and result serialization. Models can be loaded from the native JSON format or the validated ONNX subset. Both interfaces invoke the same engine and CPU kernels.

The HTTP worker pool has a fixed thread limit and a bounded pending queue. Default limits reduce accidental resource usage but do not provide multi-tenant scheduling, per-model operation budgets, or authentication. Deploy behind an appropriate application boundary if remote access is required.
