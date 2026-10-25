"""Create small real ONNX protobuf fixtures without requiring a Python package."""
import struct


def varint(value):
    value &= (1 << 64) - 1
    data = bytearray()
    while value >= 128:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)
    return bytes(data)


def integer(field, value):
    return varint(field << 3) + varint(value)


def blob(field, value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return varint((field << 3) | 2) + varint(len(value)) + value


def attribute(name, value):
    if isinstance(value, float):
        return blob(1, name) + integer(20, 1) + varint((2 << 3) | 5) + struct.pack("<f", value)
    if isinstance(value, int):
        return blob(1, name) + integer(20, 2) + integer(3, value)
    if isinstance(value, str):
        return blob(1, name) + integer(20, 3) + blob(4, value)
    return blob(1, name) + integer(20, 7) + blob(8, b"".join(varint(x) for x in value))


def node(op, inputs, output="y", attributes=None):
    return (b"".join(blob(1, name) for name in inputs) + blob(2, output)
            + blob(4, op) + b"".join(blob(5, attribute(k, v)) for k, v in (attributes or {}).items()))


def value_info(name, shape, dtype=1):
    dimensions = b"".join(blob(1, integer(1, value)) for value in shape)
    tensor_type = integer(1, dtype) + blob(2, dimensions)
    return blob(1, name) + blob(2, blob(1, tensor_type))


def initializer(name, shape, data, dtype=1, raw=True):
    result = blob(1, b"".join(varint(x) for x in shape)) + integer(2, dtype) + blob(8, name)
    if dtype == 1:
        packed = struct.pack("<" + "f" * len(data), *data)
        return result + blob(9 if raw else 4, packed)
    if raw:
        return result + blob(9, struct.pack("<" + "q" * len(data), *data))
    return result + blob(7, b"".join(varint(x) for x in data))


def model(nodes, inputs, output_shape, initializers=(), opset=17):
    graph = (blob(2, "octopus_fixture") + b"".join(blob(1, value) for value in nodes)
             + b"".join(blob(5, value) for value in initializers)
             + b"".join(blob(11, value_info(name, shape)) for name, shape in inputs)
             + blob(12, value_info("y", output_shape)))
    return integer(1, 9) + blob(7, graph) + blob(8, integer(2, opset))


def cases():
    """Return protobuf, named input tensors, and independently known output values."""
    result = []
    for op, expected in (("Add", [5, 9]), ("Sub", [3, 5]), ("Mul", [4, 14])):
        result.append((op, model([node(op, ["x", "w"])], [("x", [1, 2])], [1, 2],
                               [initializer("w", [2], [1, 2])]), {"x": ([1, 2], [4, 7])}, [1, 2], expected))
    result.append(("Gemm", model([node("Gemm", ["x", "w", "b"], attributes={"transB": 1, "alpha": 2.0, "beta": 3.0})],
                                [("x", [1, 2])], [1, 1], [initializer("w", [1, 2], [3, 4], raw=False), initializer("b", [1], [5])]),
                   {"x": ([1, 2], [1, 2])}, [1, 1], [37]))
    result.append(("MatMul", model([node("MatMul", ["x", "w"])], [("x", [1, 2])], [1, 1],
                                  [initializer("w", [2, 1], [3, 4])]), {"x": ([1, 2], [1, 2])}, [1, 1], [11]))
    result.append(("Conv", model([node("Conv", ["x", "w"], attributes={"kernel_shape": [2, 2]})],
                                [("x", [1, 1, 3, 3])], [1, 1, 2, 2], [initializer("w", [1, 1, 2, 2], [1, 1, 1, 1])]),
                   {"x": ([1, 1, 3, 3], list(range(1, 10)))}, [1, 1, 2, 2], [12, 16, 24, 28]))
    result.append(("Softmax", model([node("Softmax", ["x"], attributes={"axis": 0})], [("x", [2, 2])], [2, 2]),
                   {"x": ([2, 2], [1000, 1000, 1000, 1000])}, [2, 2], [0.5] * 4))
    result.append(("Transpose", model([node("Transpose", ["x"], attributes={"perm": [1, 0]})], [("x", [2, 3])], [3, 2]),
                   {"x": ([2, 3], [1, 2, 3, 4, 5, 6])}, [3, 2], [1, 4, 2, 5, 3, 6]))
    result.append(("Reshape", model([node("Reshape", ["x", "shape"])], [("x", [2, 3])], [3, 2],
                                   [initializer("shape", [2], [3, -1], dtype=7, raw=False)]),
                   {"x": ([2, 3], [1, 2, 3, 4, 5, 6])}, [3, 2], [1, 2, 3, 4, 5, 6]))
    result.append(("LayerNormalization", model([node("LayerNormalization", ["x", "scale", "bias"])], [("x", [2, 2])], [2, 2],
                                             [initializer("scale", [2], [2, 3]), initializer("bias", [2], [1, 2])]),
                   {"x": ([2, 2], [1, 3, 2, 4])}, [2, 2], [-0.99999, 4.999985, -0.99999, 4.999985]))
    # Resolve an edge to a producer serialized after its consumer.
    result.append(("unordered_nodes", model([node("Relu", ["hidden"]), node("Sub", ["x", "w"], "hidden")],
                                           [("x", [2])], [2], [initializer("w", [2], [2, 4])]),
                   {"x": ([2], [1, 7])}, [2], [0, 3]))
    return result
