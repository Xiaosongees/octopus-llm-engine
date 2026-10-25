# Contributing

Use English for documentation, comments, diagnostics, issue descriptions, and pull requests.

Build in a separate `build/` directory and run `ctest --test-dir build -C Release --output-on-failure` before submitting a change. Add numerical regression tests when changing kernel behavior. Tests must compare values and exercise nontrivial inputs; a shape-only check is insufficient for a numerical operator.

When adding model support, document the exact dtype, shape, attribute, and version boundaries. Reject unsupported semantics rather than substituting another operation or returning placeholder tensors. Add a reference-runtime comparison for imported operators where possible.

When changing performance, publish reproducible commands and validated outputs. Keep model weights immutable after loading and request state local to each inference call. Do not introduce unchecked allocation sizes or uninitialized typed storage.

Use the formatting style in `.clang-format`. Keep changes focused and explain the user-visible behavior and validation in the pull request description.
