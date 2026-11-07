from __future__ import annotations

import inspect
import math
import re
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Mapping


class Type(Enum):
    F32 = "f32"
    U32 = "i32"
    BOOL = "i1"
    VEC2 = "vector<2xf32>"
    VEC3 = "vector<3xf32>"
    VEC4 = "vector<4xf32>"

    @property
    def schema_name(self) -> str:
        return {
            Type.F32: "f32", Type.U32: "u32", Type.BOOL: "bool",
            Type.VEC2: "f32x2", Type.VEC3: "f32x3", Type.VEC4: "f32x4",
        }[self]

    @classmethod
    def from_schema(cls, name: str) -> Type:
        for typ in cls:
            if typ.schema_name == name:
                return typ
        raise ValueError(f"Unsupported shader interface type: {name!r}.")


@dataclass(frozen=True)
class Node:
    op: str
    type: Type
    args: tuple[Node, ...] = ()
    literal: float | int | bool | str | None = None


class Value:
    """A traced device value, not an ordinary Python number."""

    __hash__ = None

    def __init__(self, node: Node):
        self.node = node

    def __bool__(self) -> bool:
        raise TypeError("Device values cannot control Python if/loops; use select().")

    def __add__(self, other: Value | float | int) -> Value:
        return _binary("add", self, other)

    def __radd__(self, other: Value | float | int) -> Value:
        return _binary("add", other, self)

    def __sub__(self, other: Value | float | int) -> Value:
        return _binary("sub", self, other)

    def __rsub__(self, other: Value | float | int) -> Value:
        return _binary("sub", other, self)

    def __mul__(self, other: Value | float | int) -> Value:
        return _binary("mul", self, other)

    def __rmul__(self, other: Value | float | int) -> Value:
        return _binary("mul", other, self)

    def __getitem__(self, index: int) -> Value:
        sizes = {Type.VEC2: 2, Type.VEC3: 3, Type.VEC4: 4}
        size = sizes.get(self.node.type)
        if size is None or type(index) is not int or not 0 <= index < size:
            raise TypeError("Vector extraction requires an in-range constant integer component.")
        return Value(Node("extract", Type.F32, (self.node,), literal=index))

    def __eq__(self, other: object) -> Value:
        if not isinstance(other, (Value, float, int)):
            raise TypeError("Device equality requires a device value or number.")
        lhs, rhs = _operands(self, other)
        if lhs.type not in (Type.U32, Type.F32, Type.BOOL):
            raise TypeError("Equality currently supports scalars only.")
        return Value(Node("eq", Type.BOOL, (lhs, rhs)))


def _literal(value: float | int | bool, expected: Type | None = None) -> Node:
    if isinstance(value, bool):
        node = Node("constant", Type.BOOL, literal=value)
    elif isinstance(value, int) and expected != Type.F32:
        if not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("Unsigned integer literals must fit in 32 bits.")
        node = Node("constant", Type.U32, literal=value)
    else:
        try:
            number = float(value)
            number = struct.unpack("<f", struct.pack("<f", number))[0]
        except OverflowError as error:
            raise ValueError("Floating-point literals must fit in f32.") from error
        if not math.isfinite(number):
            raise ValueError("Floating-point literals must be finite f32 values.")
        node = Node("constant", Type.F32, literal=number)
    if expected is not None and node.type != expected:
        raise TypeError(f"Expected {expected.name}, got {node.type.name}.")
    return node


def _node(value: Value | float | int, expected: Type | None = None) -> Node:
    if isinstance(value, Value):
        if expected is not None and value.node.type != expected:
            raise TypeError(f"Expected {expected.name}, got {value.node.type.name}.")
        return value.node
    if not isinstance(value, (int, float)):
        raise TypeError(f"Unsupported device operand: {type(value).__name__}.")
    return _literal(value, expected)


def _operands(lhs: Value | float | int, rhs: Value | float | int) -> tuple[Node, Node]:
    expected = lhs.node.type if isinstance(lhs, Value) else (
        rhs.node.type if isinstance(rhs, Value) else None
    )
    left = _node(lhs, expected)
    right = _node(rhs, left.type)
    return left, right


def _binary(op: str, lhs: Value | float | int, rhs: Value | float | int) -> Value:
    left, right = _operands(lhs, rhs)
    if left.type not in (Type.F32, Type.U32, Type.VEC2, Type.VEC3, Type.VEC4):
        raise TypeError("Arithmetic requires matching floating-point or unsigned values.")
    return Value(Node(op, left.type, (left, right)))


def vec4(x: Value | float, y: Value | float, z: Value | float, w: Value | float) -> Value:
    return Value(Node("vec4", Type.VEC4, tuple(_node(v, Type.F32) for v in (x, y, z, w))))


def vec2(x: Value | float, y: Value | float) -> Value:
    return Value(Node("vec2", Type.VEC2, tuple(_node(v, Type.F32) for v in (x, y))))


def vec3(x: Value | float, y: Value | float, z: Value | float) -> Value:
    return Value(Node("vec3", Type.VEC3, tuple(_node(v, Type.F32) for v in (x, y, z))))


def select(condition: Value, on_true: Value | float | int, on_false: Value | float | int) -> Value:
    predicate = _node(condition, Type.BOOL)
    left, right = _operands(on_true, on_false)
    return Value(Node("select", left.type, (predicate, left, right)))


@dataclass(frozen=True)
class Vertex:
    position: Value
    color: Value | None = None
    varyings: Mapping[str, Value] = field(default_factory=dict)


@dataclass(frozen=True)
class Draw:
    vertex_count: int = 3
    topology: str = "triangle-list"

    def __post_init__(self):
        if self.topology != "triangle-list":
            raise ValueError("The preview currently supports triangle-list topology only.")
        if type(self.vertex_count) is not int or not 0 <= self.vertex_count <= 0xFFFFFFFF or self.vertex_count % 3:
            raise ValueError("Triangle-list vertex_count must be a nonnegative multiple of three fitting in u32.")

    def to_dict(self) -> dict:
        return {"topology": self.topology, "vertex_count": self.vertex_count}


@dataclass(frozen=True)
class ShaderIO:
    name: str
    type: Type
    location: int | None = None
    builtin: str | None = None
    interpolation: str = "perspective"

    def to_dict(self) -> dict:
        result = {"name": self.name, "type": self.type.schema_name}
        if self.builtin is not None:
            result["builtin"] = self.builtin
        else:
            result.update(location=self.location, interpolation=self.interpolation)
        return result


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str
    inputs: tuple[Node, ...]
    outputs: tuple[Node, ...]
    input_interface: tuple[ShaderIO, ...]
    output_interface: tuple[ShaderIO, ...]
    source_file: str
    source_line: int

    def to_dict(self) -> dict:
        nodes: list[dict] = []
        ids: dict[int, int] = {}
        input_ids = {id(node): index for index, node in enumerate(self.inputs)}

        def encode(node: Node) -> int:
            key = id(node)
            if key not in ids:
                operands = [encode(argument) for argument in node.args]
                operation = "vector" if node.op in ("vec2", "vec3", "vec4") else node.op
                data = {"op": operation, "type": node.type.schema_name, "args": operands}
                if operation == "input":
                    data["input"] = input_ids[key]
                elif operation in ("constant", "extract"):
                    data["literal"] = node.literal
                ids[key] = len(nodes)
                nodes.append(data)
            return ids[key]

        outputs = [
            {**interface.to_dict(), "value": encode(node)}
            for interface, node in zip(self.output_interface, self.outputs)
        ]
        return {
            "name": self.name, "stage": self.kind, "nodes": nodes,
            "inputs": [interface.to_dict() for interface in self.input_interface],
            "outputs": outputs, "source_file": self.source_file, "source_line": self.source_line,
        }


@dataclass(frozen=True)
class Entry:
    function: Callable
    kind: str

    def trace(self, varying_interface: tuple[ShaderIO, ...] | None = None) -> Stage:
        name = self.function.__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid entry-point name: {name!r}.")
        parameters = list(inspect.signature(self.function).parameters.values())
        if any(parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
               for parameter in parameters):
            raise TypeError("Entry points require positional device arguments.")
        if self.kind == "vertex":
            if len(parameters) != 1:
                raise TypeError("A vertex entry point must take one positional vertex-index argument.")
            interfaces = (ShaderIO("vertex_index", Type.U32, builtin="vertex_index"),)
        elif varying_interface is None:
            if len(parameters) > 1:
                raise TypeError("Trace multi-input fragment shaders through Pipeline to resolve their interfaces.")
            interfaces = tuple(ShaderIO(parameter.name, Type.VEC4, location=0) for parameter in parameters)
        else:
            available = {interface.name: interface for interface in varying_interface}
            resolved = []
            for parameter in parameters:
                interface = available.get(parameter.name)
                if interface is None and len(parameters) == 1 and len(available) == 1:
                    interface = next(iter(available.values()))
                if interface is None:
                    raise ValueError(f"Fragment input {parameter.name!r} has no matching vertex varying.")
                resolved.append(interface)
            interfaces = tuple(resolved)
        input_nodes = tuple(Node("input", interface.type, literal=interface.name) for interface in interfaces)
        result = self.function(*(Value(node) for node in input_nodes))
        if self.kind == "vertex":
            if not isinstance(result, Vertex):
                raise TypeError("A vertex entry point must return Vertex(position, color=..., varyings=...).")
            varyings = dict(result.varyings)
            if result.color is not None:
                if "color" in varyings:
                    raise ValueError("Specify color only once in the vertex interface.")
                varyings = {"color": result.color, **varyings}
            outputs = (_node(result.position, Type.VEC4), *(_node(value) for value in varyings.values()))
            output_interface = (
                ShaderIO("position", Type.VEC4, builtin="position"),
                *(ShaderIO(label, node.type, location=index) for index, (label, node) in
                  enumerate(zip(varyings, outputs[1:]))),
            )
        else:
            if not isinstance(result, (Value, Mapping)):
                raise TypeError("A fragment entry point must return a device value or named render-target mapping.")
            targets = dict(result) if isinstance(result, Mapping) else {"target0": result}
            outputs = tuple(_node(value) for value in targets.values())
            output_interface = tuple(
                ShaderIO(label, node.type, location=index)
                for index, (label, node) in enumerate(zip(targets, outputs))
            )
        for interface in (*interfaces, *output_interface):
            if not isinstance(interface.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", interface.name):
                raise ValueError(f"Invalid stage interface name: {interface.name!r}.")
            if interface.builtin is None and interface.type not in (Type.F32, Type.VEC2, Type.VEC3, Type.VEC4):
                raise TypeError("The bootstrap frontend currently supports f32 scalar/vector rasterization interfaces.")
        visited: set[int] = set()
        owned_inputs = {id(node) for node in input_nodes}

        def validate(node: Node) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            if node.op == "input" and id(node) not in owned_inputs:
                raise ValueError("A stage cannot capture another stage's device input.")
            if node.op not in {"input", "constant", "vec2", "vec3", "vec4", "extract", "select", "eq", "add", "sub", "mul"}:
                raise ValueError(f"Unsupported device operation: {node.op}.")
            for arg in node.args:
                validate(arg)

        for output in outputs:
            validate(output)
        return Stage(name, self.kind, input_nodes, outputs, interfaces, output_interface,
                     self.function.__code__.co_filename, self.function.__code__.co_firstlineno)


def vertex(function: Callable) -> Entry:
    return Entry(function, "vertex")


def fragment(function: Callable) -> Entry:
    return Entry(function, "fragment")


@dataclass(frozen=True)
class Pipeline:
    vertex: Entry
    fragment: Entry
    draw: Draw = field(default_factory=Draw)

    def trace(self) -> tuple[Stage, Stage]:
        if self.vertex.kind != "vertex" or self.fragment.kind != "fragment":
            raise TypeError("Pipeline requires a @vertex entry followed by a @fragment entry.")
        if not isinstance(self.draw, Draw):
            raise TypeError("Pipeline draw configuration must be a Draw object.")
        vertex_stage = self.vertex.trace()
        fragment_stage = self.fragment.trace(tuple(
            interface for interface in vertex_stage.output_interface if interface.builtin is None
        ))
        return vertex_stage, fragment_stage
