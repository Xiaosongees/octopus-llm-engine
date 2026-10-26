# HTTP API

Start a loaded model with `octopus serve --sample` or `octopus serve --model FILE`.

Options: `--host` defaults to `127.0.0.1`, `--port` defaults to `8080`, and `--threads` defaults to `4`. Port `0` asks the OS for a free port. Worker count must be between 1 and 64. The selected listener address is printed to standard output.

## `GET /v1/health`

Returns HTTP 200 with `status`, `version`, and `uptime_secs`. Model loading happens before the server starts, so a process cannot report a healthy server with an unloaded model.

## `GET /v1/model/info`

Returns HTTP 200 with `model_loaded`, `node_count`, `backend`, named input shapes in `inputs`, and ordered output names in `outputs`.

## `POST /v1/inference`

Content type: `application/json`. Maximum body size: 1 MiB. Tensor rank limit: 16. Values must be finite numbers representable in float32; booleans and numeric strings are rejected.

Single-input request:

```json
{
  "input_name": "input",
  "input_shape": [1, 4],
  "input_data": [1, 2, 3, 4],
  "output_name": "output"
}
```

`input_name` defaults to `input`. `output_name` defaults to the first declared graph output. Data is a flat row-major array, not nested arrays.

Multi-input request:

```json
{
  "inputs": {
    "left": {"shape": [1, 2], "data": [1, 2]},
    "right": {"shape": [1, 2], "data": [3, 4]}
  }
}
```

Provide exactly the model's declared inputs. Do not combine `inputs` with single-input fields. Each request is independent; omitted inputs are never reused from an earlier request.

Successful response:

```json
{
  "backend": "cpu",
  "inference_time_ms": 0.01,
  "output_data": [1.5, 1.5, 8.0],
  "output_shape": [1, 3],
  "outputs": {
    "output": {"data": [1.5, 1.5, 8.0], "shape": [1, 3]}
  }
}
```

The timing value above is illustrative; actual responses report measured execution time, excluding network transfer, JSON parsing, and response encoding.

## Errors and operational limits

Errors use a JSON object with an `error` field.

| Status | Meaning |
| --- | --- |
| 400 | Invalid JSON, shape, input name, output name, or operator input |
| 404 | Unknown endpoint |
| 413 | Body exceeds 1 MiB |
| 415 | Content type is not JSON |
| 500 | Unexpected inference failure or non-finite model output |
| 503 | Memory allocation failure during inference |

Read and write timeouts are five seconds. The worker queue holds up to 64 waiting requests; transport-level queue rejection may close the connection instead of returning JSON. Timeouts do not cancel CPU kernels that have already started. There is no authentication or permissive cross-origin policy. Use a reverse proxy and application-level limits for deployment outside a trusted local environment.
