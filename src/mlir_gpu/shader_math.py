"""Device f32 math; these functions build MLIR, not host Python results."""

from .frontend import VECTOR_SIZES, Node, Type, Value, _node, _operands


def _floating(value: Value | float) -> Value:
    result = value if isinstance(value, Value) else Value(_node(value, Type.F32))
    if result.node.type not in (Type.F32, *VECTOR_SIZES):
        raise TypeError("Shader math requires f32 scalar/vector values.")
    return result


def _known_components(node: Node) -> tuple[float | None, ...]:
    if node.op == "constant" and isinstance(node.literal, (int, float)):
        return (float(node.literal),)
    if node.op in ("vec2", "vec3", "vec4"):
        return tuple(_known_components(arg)[0] for arg in node.args)
    if node.op == "broadcast":
        return _known_components(node.args[0]) * VECTOR_SIZES[node.type]
    return (None,) * VECTOR_SIZES.get(node.type, 1)


def _unary(operation: str, value: Value | float) -> Value:
    node = _floating(value).node
    for component in _known_components(node):
        if component is not None and (
            (operation == "sqrt" and component < 0)
            or (operation in ("log", "log2") and component <= 0)
        ):
            raise ValueError(f"{operation} argument is outside its portable domain.")
    return Value(Node(operation, node.type, (node,)))


def sin(value: Value | float) -> Value:
    return _unary("sin", value)


def cos(value: Value | float) -> Value:
    return _unary("cos", value)


def sqrt(value: Value | float) -> Value:
    return _unary("sqrt", value)


def exp(value: Value | float) -> Value:
    return _unary("exp", value)


def exp2(value: Value | float) -> Value:
    return _unary("exp2", value)


def log(value: Value | float) -> Value:
    return _unary("log", value)


def log2(value: Value | float) -> Value:
    return _unary("log2", value)


def floor(value: Value | float) -> Value:
    return _unary("floor", value)


def pow(base: Value | float, exponent: Value | float) -> Value:
    left, right = _operands(_floating(base), _floating(exponent), broadcast=True)
    for value, power in zip(_known_components(left), _known_components(right)):
        if value is not None and (value < 0 or (value == 0 and power is not None and power <= 0)):
            raise ValueError("pow requires a nonnegative base and a positive exponent at zero.")
    return Value(Node("pow", left.type, (left, right)))
