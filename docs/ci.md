# Optional GitHub Actions setup

The repository includes a ready-to-use workflow template at `ci/github-actions.yml`. The template is inactive until it is installed under `.github/workflows/`.

The configuration builds on Windows and Linux with HTTP support enabled and disabled, runs CTest, compares supported ONNX operators with ONNX Runtime, and runs an address/undefined-behavior sanitizer build on Linux. These checks execute on pull requests and pushes to `main` after activation.

To activate it, copy `ci/github-actions.yml` to `.github/workflows/ci.yml`, commit the new file, and push with a GitHub credential permitted to manage workflow files. GitHub CLI OAuth credentials need the `workflow` scope in addition to repository access. Repository administrators can also install the file through GitHub's web interface if their account permissions allow it.

Without Actions, local validation remains available:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

For Linux sanitizer validation, configure a separate directory with `-DOCTOPUS_ENABLE_SANITIZERS=ON -DOCTOPUS_BUILD_SERVER=OFF`. A supplied workflow template is not evidence that a remote check has executed; report actual build and test results separately.
