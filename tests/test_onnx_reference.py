"""Optional differential validation against the official ONNX Runtime CPU backend."""
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from onnx_fixture import cases

binary = str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory)
    for name, protobuf, inputs, _, _ in cases():
        if name == "unordered_nodes":
            # ONNX's checker requires topological serialization; Octopus also accepts unordered DAGs.
            continue
        path = root / f"{name}.onnx"
        path.write_bytes(protobuf)
        onnx.checker.check_model(str(path))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        reference = session.run(None, {k: np.asarray(data, dtype=np.float32).reshape(shape)
                                      for k, (shape, data) in inputs.items()})[0]
        request = root / "input.json"
        request.write_text(json.dumps({"inputs": {k: {"shape": shape, "data": data}
                                                  for k, (shape, data) in inputs.items()}}), encoding="utf-8")
        result = subprocess.run([binary, "infer", "--model", str(path), "--input", str(request)],
                                text=True, capture_output=True, check=True, timeout=20)
        output = json.loads(result.stdout)
        actual = np.asarray(output["output_data"], dtype=np.float32).reshape(output["output_shape"])
        np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5, err_msg=name)
        print(f"PASS ONNX Runtime reference: {name}")
