"""Exercise the real HTTP listener, validation, and concurrent request isolation."""
import concurrent.futures
import json
import pathlib
import queue
import subprocess
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

BINARY = str(pathlib.Path(sys.argv.pop(1)).resolve())


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = subprocess.Popen([BINARY, "serve", "--sample", "--port", "0", "--threads", "4"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = queue.Queue()
        threading.Thread(target=lambda: output.put(cls.process.stdout.readline()), daemon=True).start()
        try:
            line = output.get(timeout=10)
            if not line.startswith("Listening on http://"):
                raise RuntimeError("HTTP listener did not start: " + line)
            cls.base = line.strip().split("Listening on ", 1)[1]
            for _ in range(100):
                try:
                    with urllib.request.urlopen(cls.base + "/v1/health", timeout=1):
                        return
                except (OSError, urllib.error.URLError):
                    time.sleep(0.02)
            raise RuntimeError("HTTP listener did not become ready")
        except Exception:
            cls.process.kill()
            cls.process.communicate(timeout=5)
            raise

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.communicate(timeout=5)

    def request(self, path, body=None, content_type="application/json"):
        data = None if body is None else body.encode("utf-8") if isinstance(body, str) else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, headers={"Content-Type": content_type})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    def test_health_and_model_information(self):
        status, health = self.request("/v1/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")
        status, model = self.request("/v1/model/info")
        self.assertEqual(model["inputs"], {"input": [1, 4]})
        self.assertEqual(model["outputs"], ["output"])

    def test_successful_inference(self):
        status, result = self.request("/v1/inference", {"input_data": [1, 2, 3, 4], "input_shape": [1, 4]})
        self.assertEqual(status, 200)
        self.assertEqual(result["output_data"], [1.5, 1.5, 8.0])
        self.assertEqual(result["output_shape"], [1, 3])

    def test_malformed_requests_return_client_errors(self):
        requests = ["{", [], {"input_data": [1], "input_shape": [1, 4]},
                    {"input_data": [1, 2, 3, 4], "input_shape": [2, 2]},
                    {"input_data": [1, 2, 3, 4], "input_shape": [1, 4], "input_name": "missing"},
                    {"input_data": [1, 2, 3, 4], "input_shape": [1, 4], "output_name": "missing"},
                    {"input_data": [True], "input_shape": [1]},
                    {"input_data": [1], "input_shape": [18446744073709551615, 2]}]
        for request in requests:
            with self.subTest(request=request):
                status, result = self.request("/v1/inference", request)
                self.assertEqual(status, 400)
                self.assertIn("error", result)
        status, _ = self.request("/v1/inference", "{}", "text/plain")
        self.assertEqual(status, 415)
        status, _ = self.request("/does-not-exist")
        self.assertEqual(status, 404)

    def test_request_size_limit(self):
        status, _ = self.request("/v1/inference", " " * (1024 * 1024 + 1))
        self.assertEqual(status, 413)

    def test_concurrent_requests_are_isolated(self):
        def infer(value):
            status, result = self.request("/v1/inference", {"input_data": [value] * 4, "input_shape": [1, 4]})
            self.assertEqual(status, 200)
            self.assertEqual(result["output_data"], [value + 0.5, value - 0.5, value * 2 + 1])
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(infer, range(1, 25)))


if __name__ == "__main__":
    unittest.main()
