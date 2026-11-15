"""Device f32 math; these functions build MLIR, not host Python results."""

from .frontend import (
    VECTOR_SIZES,
    Node,
    Type,
    Value,
    _node,
    _operands,
    load_texture,
    maximum,
    minimum,
    sample_texture,
    sample_texture_level,
    select,
    vec2,
    vec3,
)


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
            or (operation in ("rsqrt", "log", "log2") and component <= 0)
        ):
            raise ValueError(f"{operation} argument is outside its portable domain.")
    return Value(Node(operation, node.type, (node,)))


def sin(value: Value | float) -> Value:
    return _unary("sin", value)


def cos(value: Value | float) -> Value:
    return _unary("cos", value)


def sqrt(value: Value | float) -> Value:
    return _unary("sqrt", value)


def rsqrt(value: Value | float) -> Value:
    return _unary("rsqrt", value)


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


def _vector(value: Value, size: int | None = None) -> Value:
    if not isinstance(value, Value) or value.node.type not in VECTOR_SIZES:
        raise TypeError("Expected an f32 device vector.")
    if size is not None and VECTOR_SIZES[value.node.type] != size:
        raise TypeError(f"Expected an f32 vector with {size} components.")
    return value


def dot(lhs: Value, rhs: Value) -> Value:
    left, right = _vector(lhs), _vector(rhs)
    if left.node.type != right.node.type:
        raise TypeError("dot requires matching vector shapes.")
    result = left[0] * right[0]
    for index in range(1, VECTOR_SIZES[left.node.type]):
        result = result + left[index] * right[index]
    return result


def cross(lhs: Value, rhs: Value) -> Value:
    a, b = _vector(lhs, 3), _vector(rhs, 3)
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)


def length(value: Value) -> Value:
    return sqrt(dot(value, value))


def normalize(value: Value) -> Value:
    magnitude = length(value)
    return value / select(magnitude == 0.0, 1.0, magnitude)


def rotate2(value: Value, angle: Value | float) -> Value:
    point, radians = _vector(value, 2), _floating(angle)
    if radians.node.type != Type.F32:
        raise TypeError("Rotation angle must be an f32 scalar.")
    s, c = sin(radians), cos(radians)
    return vec2(c * point.x - s * point.y, s * point.x + c * point.y)


def clamp(value: Value | float, lower: Value | float, upper: Value | float) -> Value:
    low, high = _operands(_floating(lower), _floating(upper), broadcast=True)
    for a, b in zip(_known_components(low), _known_components(high)):
        if a is not None and b is not None and a > b:
            raise ValueError("clamp lower bound must not exceed its upper bound.")
    return minimum(maximum(_floating(value), Value(low)), Value(high))


def mix(lhs: Value | float, rhs: Value | float, weight: Value | float) -> Value:
    left, right = _operands(_floating(lhs), _floating(rhs), broadcast=True)
    return Value(left) + (Value(right) - Value(left)) * _floating(weight)


def smoothstep(edge0: Value | float, edge1: Value | float, value: Value | float) -> Value:
    low, high = _operands(_floating(edge0), _floating(edge1), broadcast=True)
    for a, b in zip(_known_components(low), _known_components(high)):
        if a is not None and b is not None and a >= b:
            raise ValueError("smoothstep requires increasing edges.")
    t = clamp((_floating(value) - Value(low)) / (Value(high) - Value(low)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def fract(value: Value | float) -> Value:
    operand = _floating(value)
    return operand - floor(operand)


def mod(value: Value | float, period: Value | float) -> Value:
    x, y = _operands(_floating(value), _floating(period), broadcast=True)
    for component in _known_components(y):
        if component is not None and component <= 0:
            raise ValueError("Modulo period must be positive.")
    return Value(x) - Value(y) * floor(Value(x) / Value(y))
