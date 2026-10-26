"""End-to-end model, ONNX, and CLI tests using only the Python standard library."""
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

from onnx_fixture import cases, model, node

BINARY = str(pathlib.Path(sys.argv.pop(1)).resolve())


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def run_cli(self, *args, ok=True):
        result = subprocess.run([BINARY, *map(str, args)], text=True, capture_output=True, timeout=20)
        if ok:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Error:", result.stderr)
        return result

    def write_json(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_help_info_and_invalid_arguments(self):
        self.assertIn("C++20", self.run_cli("--help").stdout)
        self.assertIn("not implemented", self.run_cli("info").stdout)
        for args in (("no-such-command",), ("demo", "--backend", "cuda"), ("benchmark", "--iterations", "0"),
                     ("benchmark", "--size", "-1"), ("demo", "--unknown"), ("infer",), ("demo", "--backend", "cpu", "--backend", "cpu")):
            with self.subTest(args=args):
                self.run_cli(*args, ok=False)

    def test_sample_round_trip(self):
        path = self.root / "sample.json"
        self.run_cli("make-model", "--output", path)
        input_path = self.write_json("input.json", {"input_data": [1, 2, 3, 4], "input_shape": [1, 4]})
        result = json.loads(self.run_cli("infer", "--model", path, "--input", input_path).stdout)
        self.assertEqual(result["output_data"], [1.5, 1.5, 8.0])
        self.assertEqual(result["output_shape"], [1, 3])
        self.assertEqual(json.loads(self.run_cli("demo").stdout)["output_data"], result["output_data"])

    def test_missing_weights_are_rejected(self):
        path = self.root / "sample.json"
        self.run_cli("make-model", "--output", path)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["parameters"].pop("fc1.weight")
        path.write_text(json.dumps(document), encoding="utf-8")
        data = self.write_json("input.json", {"input_data": [1, 2, 3, 4], "input_shape": [1, 4]})
        result = self.run_cli("infer", "--model", path, "--input", data, ok=False)
        self.assertIn("Missing model parameter", result.stderr)

    def test_invalid_json_and_input_shape(self):
        path = self.root / "sample.json"
        self.run_cli("make-model", "--output", path)
        for request in ({"input_data": [1], "input_shape": [1, 4]}, {"input_data": [True], "input_shape": [1]},
                        {"input_data": [1], "input_shape": [-1]}, {"input_data": [1], "input_shape": [1.0]},
                        {"input_data": [1e100], "input_shape": [1]}):
            data = self.write_json("bad.json", request)
            self.run_cli("infer", "--model", path, "--input", data, ok=False)
        path.write_text("{", encoding="utf-8")
        self.run_cli("infer", "--model", path, "--input", data, ok=False)

    def test_transformer_demo_is_deterministic_and_seeded(self):
        a = json.loads(self.run_cli("transformer-demo", "--seed", "7", "--causal").stdout)
        b = json.loads(self.run_cli("transformer-demo", "--seed", "7", "--causal").stdout)
        c = json.loads(self.run_cli("transformer-demo", "--seed", "8", "--causal").stdout)
        self.assertEqual(a["output_shape"], [1, 4, 16])
        self.assertEqual(a["output_data"], b["output_data"])
        self.assertNotEqual(a["output_data"], c["output_data"])
        self.assertTrue(all(math.isfinite(x) for x in a["output_data"]))

    def test_real_onnx_protobuf_models(self):
        for name, protobuf, inputs, shape, expected in cases():
            with self.subTest(operator=name):
                path = self.root / f"{name}.onnx"
                path.write_bytes(protobuf)
                data = self.write_json("input.json", {"inputs": {k: {"shape": s, "data": d} for k, (s, d) in inputs.items()}})
                result = json.loads(self.run_cli("infer", "--model", path, "--input", data).stdout)
                self.assertEqual(result["output_shape"], shape)
                self.assertEqual(len(result["output_data"]), len(expected))
                for actual, reference in zip(result["output_data"], expected):
                    self.assertAlmostEqual(actual, reference, delta=2e-5)

    def test_onnx_rejects_unsupported_and_truncated_models(self):
        data = self.write_json("input.json", {"inputs": {"x": {"shape": [1], "data": [1]}}})
        documents = [b"junk", b"\x3a\xff\xff", model([node("UnsupportedOp", ["x"])], [("x", [1])], [1]),
                     model([node("Relu", ["x"])], [("x", [1])], [1], opset=12),
                     model([node("Relu", ["missing"])], [("x", [1])], [1])]
        for protobuf in documents:
            path = self.root / "bad.onnx"
            path.write_bytes(protobuf)
            self.run_cli("infer", "--model", path, "--input", data, ok=False)


if __name__ == "__main__":
    unittest.main()
