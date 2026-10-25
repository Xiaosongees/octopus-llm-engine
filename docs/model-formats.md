# Model formats and support boundaries

## Native JSON: `octopus-model`, version 1

Generate a complete example with `octopus make-model --output sample.json`.

```json
{
  "format": "octopus-model",
  "version": 1,
  "nodes": [
    {"id": "x", "op": "Input", "attributes": {"shape": [1, 2]}},
    {"id": "fc", "op": "Dense", "inputs": ["x"], "parameters": ["fc.weight", "fc.bias"]},
    {"id": "y", "op": "Output", "inputs": ["fc"]}
  ],
  "parameters": {
    "fc.weight": {"shape": [1, 2], "data": [2.0, 3.0]},
    "fc.bias": {"shape": [1], "data": [1.0]}
  }
}
```

For `x = [[4, 5]]`, this model produces `y = [[24]]`. Dense weights use `[output_features, input_features]`. All parameter values must be finite float32 values. Model files are limited to 64 MiB and tensor rank to 16 in the JSON interface.

Nodes have an `id`, `op`, optional `inputs`, optional `parameters`, and optional `attributes`. Attribute names are defined by `octopus::Attributes` in `include/octopus/graph.hpp`. Missing attributes use the documented C++ defaults. Input shape declarations are exact; dynamic axes are not supported.

Supported native operations are Input, Output, Constant, Identity, Dense, Conv2D, Relu, Gelu, Sigmoid, Tanh, Softmax, LayerNorm, MatMul, Add, Sub, Mul, Reshape, Transpose, Embedding, Gather, Attention, Dropout, and Gemm. Native Gather currently means row lookup on a rank-two table and requires an explicit `axis: 0` attribute. GELU uses the tanh approximation. Dropout is evaluation-only.

Reshape supports one `-1` inferred dimension and `0` to copy the corresponding input dimension. Other negative values and nonintegral inferred sizes are rejected. Transpose uses `axes`; an empty permutation reverses all axes.

## ONNX protobuf subset

`.onnx` files are parsed as actual protobuf messages. The importer does not substitute unrelated operations or return empty graphs for arbitrary files.

Supported model versions: IR 3-10 and standard-domain opset 13-22. Graph inputs must be fixed-shape FLOAT tensors. Weight tensors must be FLOAT, stored inline as `raw_data` or `float_data`. INT64 initializer vectors are supported for constant Reshape shapes. External tensor files, sparse tensors, custom domains, local functions, and training graphs are rejected.

| ONNX operator | Supported semantics |
| --- | --- |
| Add, Sub, Mul | Separate operations with multidirectional broadcasting |
| MatMul | Rank >=2; broadcast leading batch axes |
| Gemm | Rank-two inputs, alpha/beta, transA/transB, broadcast bias |
| Relu, Sigmoid, Tanh, Identity | Float32 input/output |
| Gelu | Opset >=20 with `approximate="tanh"` |
| Softmax | Opset >=13 semantics, configurable axis |
| LayerNormalization | Opset >=17, last-axis normalization, float32 stash, one output |
| Conv | NCHW, group 1, unit dilation, symmetric explicit 2D padding |
| Reshape | Constant rank-one INT64 shape initializer; `allowzero=0` |
| Transpose | Explicit permutation or default axis reversal |

Every imported operator must have one nonempty output. Graph output names must not collide with graph input names. Unknown operators and unsupported attributes fail explicitly. Integer graph inputs and ONNX Gather are not supported. Node serialization order does not need to be topological, but the graph must be acyclic. The importer is intentionally a subset and cannot load arbitrary exported LLM graphs.

## Transformer models

`TransformerEncoder::build(batch, sequence, seed)` returns a complete graph and initialized parameter map. The graph has one input named `tokens` with shape `[batch, sequence]` and one output named `output` with shape `[batch, sequence, d_model]`. Position IDs are internal constants.

Initialization is useful for numerical testing and architecture experiments. It does not produce trained weights. Loading a pretrained model requires an explicit parameter adapter and validation against that model's reference implementation. The project does not include a tokenizer or a language-model generation loop.
