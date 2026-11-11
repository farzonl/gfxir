from __future__ import annotations

import ast
import builtins
import inspect
import operator
import textwrap
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Callable

from . import sdf, shader_math
from .frontend import (
    ComputePipeline, Entry, GlobalInvocationId, LocalInvocationId, Node, Pipeline,
    Type, Value, Vertex, WorkgroupId, _node, _normalize_buffer_element_type,
    atomic_add, atomic_and, atomic_compare_exchange, atomic_exchange,
    atomic_max, atomic_min, atomic_or, atomic_sub, atomic_umax, atomic_umin,
    atomic_xor, full_barrier, global_invocation_id, local_invocation_id,
    maximum, minimum, select, shared_array, storage_barrier,
    workgroup_barrier, workgroup_id,
)
from .frontend import i32, u32, vec2, vec3, vec4
from .parameters import _ParameterValues


_VECTOR_BY_SIZE = {2: Type.VEC2, 3: Type.VEC3, 4: Type.VEC4}
_VECTOR_CONSTRUCTORS = {2: vec2, 3: vec3, 4: vec4}


def _lane(vector: Value, index: int) -> Value:
    return getattr(vector, "xyzw"[index])


@dataclass(frozen=True)
class MatrixValue:
    """A device matrix with C columns, each a vector of R components (GLSL/WGSL ``matCxR``)."""
    columns: tuple[Value, ...]
    is_device_matrix = True

    def __post_init__(self):
        if not isinstance(self.columns, tuple) or not 2 <= len(self.columns) <= 4:
            raise TypeError("MatrixValue requires a tuple of 2 to 4 column vectors.")
        first = self.columns[0]
        if not isinstance(first, Value) or first.node.type not in _VECTOR_BY_SIZE.values():
            raise TypeError("MatrixValue columns must be Values of type VEC2, VEC3 or VEC4.")
        for col in self.columns:
            if not isinstance(col, Value) or col.node.type != first.node.type:
                raise TypeError("MatrixValue columns must all be Values of the same vector type.")

    @property
    def column_count(self) -> int:
        return len(self.columns)

    @property
    def row_count(self) -> int:
        return {v: k for k, v in _VECTOR_BY_SIZE.items()}[self.columns[0].node.type]

    @property
    def type(self) -> Type:
        return Type(f"mat{self.column_count}x{self.row_count}")

    def __getitem__(self, index: int) -> Value:
        if type(index) is not int or not 0 <= index < self.column_count:
            raise IndexError(f"Matrix column index out of range (expected 0..{self.column_count - 1}).")
        return self.columns[index]

    def __len__(self) -> int:
        return self.column_count

    def __bool__(self) -> bool:
        raise TypeError("Device values cannot control Python if/loops; use select().")

    def __mul__(self, other: Value | MatrixValue) -> Value | MatrixValue:
        if isinstance(other, Value):
            # matCxR * vecC -> vecR: linear combination of the columns.
            if other.node.type != _VECTOR_BY_SIZE[self.column_count]:
                raise TypeError(
                    f"{self.type.value} * vector requires {_VECTOR_BY_SIZE[self.column_count].name}, "
                    f"got {other.node.type.name}."
                )
            result = self.columns[0] * _lane(other, 0)
            for i in range(1, self.column_count):
                result = result + self.columns[i] * _lane(other, i)
            return result
        if isinstance(other, MatrixValue):
            # (CaxRa) * (CbxRb) requires Ca == Rb and yields CbxRa.
            if self.column_count != other.row_count:
                raise TypeError(f"Cannot multiply {self.type.value} by {other.type.value}: "
                                f"inner dimensions {self.column_count} and {other.row_count} differ.")
            return MatrixValue(tuple(self * col for col in other.columns))
        return NotImplemented

    def __rmul__(self, other: Value) -> Value:
        if isinstance(other, Value):
            # vecR * matCxR -> vecC: dot product of the vector with each column.
            if other.node.type != _VECTOR_BY_SIZE[self.row_count]:
                raise TypeError(
                    f"Vector * {self.type.value} requires {_VECTOR_BY_SIZE[self.row_count].name}, "
                    f"got {other.node.type.name}."
                )
            comps = []
            for col in self.columns:
                total = _lane(other, 0) * _lane(col, 0)
                for i in range(1, self.row_count):
                    total = total + _lane(other, i) * _lane(col, i)
                comps.append(total)
            return _VECTOR_CONSTRUCTORS[self.column_count](*comps)
        return NotImplemented


_ATOMIC_RMW_KINDS = {
    atomic_add: "add",
    atomic_sub: "sub",
    atomic_min: "min",
    atomic_max: "max",
    atomic_umin: "umin",
    atomic_umax: "umax",
    atomic_and: "and",
    atomic_or: "or",
    atomic_xor: "xor",
    atomic_exchange: "xchg",
}

_BARRIER_SCOPES = {
    workgroup_barrier: "workgroup",
    storage_barrier: "device",
    full_barrier: "all",
}


_INTRINSICS = {
    Node, Value, Vertex, i32, u32, vec2, vec3, vec4, select, maximum, minimum,
    global_invocation_id, local_invocation_id, workgroup_id,
}
for _module in (shader_math, sdf):
    _INTRINSICS.update(
        value for name, value in vars(_module).items()
        if not name.startswith("_") and isinstance(value, FunctionType)
        and value.__module__ == _module.__name__
    )


@dataclass(frozen=True)
class _Ref:
    name: str
    type: str


@dataclass
class _LoopExit:
    values: dict[str, Value]
    mode: int | Value
    returned: object = None


@dataclass(frozen=True)
class _Next:
    action: Callable[[], object]


class _Builder:
    def __init__(self, engine, arguments=(), captures=None):
        self.engine = engine
        self.block = {"name": "entry", "arguments": list(arguments), "operations": []}
        self.cache = dict(captures or {})

    def raw(self, op, args=(), types=(), attributes=None, regions=()):
        refs = [_Ref(self.engine.name(), typ) for typ in types]
        item = {
            "op": op, "args": [arg.name for arg in args],
            "results": [{"name": ref.name, "type": ref.type} for ref in refs],
            **self.engine.location(),
        }
        if attributes:
            item["attributes"] = attributes
        if regions:
            item["regions"] = [{"blocks": [region.block]} for region in regions]
        self.block["operations"].append(item)
        return refs

    def bind(self, ref, typ):
        node = Node("ssa", typ, literal=ref.name)
        self.cache[id(node)] = ref
        self.engine.owned.add(id(node))
        self.engine.nodes[id(node)] = node
        return Value(node)

    def value(self, op, args, typ, attributes=None):
        ref, = self.raw(op, [self.emit(arg.node) for arg in args], [typ.schema_name], attributes)
        return self.bind(ref, typ)

    def emit(self, node):
        key = id(node)
        if key in self.cache:
            return self.cache[key]
        self.engine.nodes[key] = node
        if node.op in ("input", "ssa"):
            raise ValueError("A stage cannot capture another stage's device input or a region-local SSA value.")
        args = [self.emit(arg) for arg in node.args]
        typ = node.type.schema_name if isinstance(node.type, Type) else str(node.type)
        attrs = None
        floating = node.args and node.args[0].type in (Type.F32, Type.VEC2, Type.VEC3, Type.VEC4)
        signed = node.args and node.args[0].type == Type.I32
        if node.op == "constant":
            op, attrs = "arith.constant", {"value": {"kind": "literal", "type": typ, "value": node.literal}}
        elif node.op == "builtin":
            op, attrs = "gfx.builtin", {"kind": {"kind": "builtin", "value": node.literal}}
        elif node.op in ("vec2", "vec3", "vec4"):
            op = "vector.from_elements"
        elif node.op == "extract":
            op, attrs = "vector.extract", {"static_position": {"kind": "i64_array", "value": [node.literal]}}
        elif node.op in ("eq", "cmp"):
            op = "arith.cmpf" if floating else "arith.cmpi"
            attrs = {"predicate": {"kind": "cmpf" if floating else "cmpi",
                                    "value": node.literal if node.op == "cmp" else "oeq" if floating else "eq"}}
        elif node.op in ("add", "sub", "mul"):
            op = f"arith.{node.op}{'f' if floating else 'i'}"
        elif node.op in ("min", "max"):
            op = f"arith.{node.op}{'numf' if floating else 'si' if signed else 'ui'}"
        elif node.op == "cast":
            self.cache[key] = args[0]
            return args[0]
        else:
            if node.op in ("floordiv", "mod", "shl", "shr") or (
                node.op in ("and", "or", "xor") and node.type in (Type.I32, Type.U32)
            ):
                self.engine.structured = True
            op = {
                "broadcast": "vector.broadcast", "select": "arith.select",
                "div": "arith.divf", "neg": "arith.negf" if floating else "arith.subi",
                "floordiv": "arith.divsi" if signed else "arith.divui",
                "mod": "arith.remsi" if signed else "arith.remui",
                "shl": "arith.shli",
                "shr": "arith.shrsi" if signed else "arith.shrui",
                "abs": "math.absf" if floating else "math.absi",
                "and": "arith.andi", "or": "arith.ori", "xor": "arith.xori",
                **{name: f"math.{name}" for name in ("sin", "cos", "sqrt", "rsqrt", "exp", "exp2", "log", "log2", "floor")},
                "pow": "math.powf",
            }.get(node.op)
            if op is None:
                raise ValueError(f"Unsupported device operation: {node.op}.")
            if node.op == "neg" and not floating:
                args.insert(0, self.emit(_node(0, node.type)))
        ref, = self.raw(op, args, [typ], attrs)
        self.cache[key] = ref
        return ref


def _mlir_element_type(elem_type: Any) -> str:
    if isinstance(elem_type, Type):
        if elem_type in (Type.I32, Type.U32):
            return "i32"
        return elem_type.value
    return str(getattr(elem_type, "schema_name", elem_type))


@dataclass(frozen=True)
class _BufferProxy:
    name: str
    ref: _Ref
    ptr: Value
    element_type: Any
    access: Any
    set: int = 0
    binding: int = 0
    shape: tuple[int, ...] | None = None
    alignment: int | None = None

    @property
    def element_schema_name(self) -> str:
        return _mlir_element_type(self.element_type)

    @property
    def storage_class(self) -> str:
        return "global"


@dataclass(frozen=True)
class _SharedArrayProxy:
    name: str
    ref: _Ref
    element_type: Type
    count: int
    shape: tuple[int, ...]

    @property
    def element_schema_name(self) -> str:
        return _mlir_element_type(self.element_type)

    @property
    def storage_class(self) -> str:
        return "workgroup"

    @property
    def alignment(self) -> int | None:
        return None


def _has_nonlocal_jump(statements: Sequence[ast.stmt], in_loop: bool = False) -> bool:
    for stmt in statements:
        if isinstance(stmt, ast.Return):
            return True
        if isinstance(stmt, (ast.Break, ast.Continue)):
            if not in_loop:
                return True
        elif isinstance(stmt, ast.If):
            if _has_nonlocal_jump(stmt.body, in_loop) or _has_nonlocal_jump(stmt.orelse, in_loop):
                return True
        elif isinstance(stmt, (ast.For, ast.While)):
            if _has_nonlocal_jump(stmt.body, True) or _has_nonlocal_jump(stmt.orelse, True):
                return True
    return False


class ASTLowering:
    def __init__(self, function, inputs, interfaces, parameters=None, workgroup_size=None):
        self.serial = 0
        self.owned = set()
        self.nodes = {id(node): node for node in inputs}
        self.structured = False
        self.stack = []
        self.scope = object()
        self.file = function.__code__.co_filename
        self.line = function.__code__.co_firstlineno
        self.column = 1
        self.workgroup_size = workgroup_size
        self.builder = _Builder(self)
        self.root = self.builder
        for node, interface in zip(inputs, interfaces):
            attrs = (
                {"kind": {"kind": "builtin", "value": interface.builtin}}
                if interface.builtin else
                {"location": {"kind": "literal", "type": "i32", "value": interface.location},
                 "name": {"kind": "string", "value": interface.name},
                 "interpolation": {"kind": "interpolation", "value": interface.interpolation}}
            )
            ref, = self.root.raw("gfx.builtin" if interface.builtin else "gfx.input",
                                 types=[node.type.schema_name], attributes=attrs)
            self.root.cache[id(node)] = ref
        parameters = parameters or {}
        blocks = list(parameters.values())
        if blocks and any(block != blocks[0] for block in blocks):
            self.fail("This parameter ABI supports one matching block per shader module.")
        self.parameters = blocks[0] if blocks else None

        from .frontend import BufferBinding, BufferType, AccessMode, resolve_annotation
        self.buffer_bindings = {}
        binding_counter = 0
        for parameter in inspect.signature(function).parameters.values():
            annotation = resolve_annotation(function, parameter.annotation)
            if isinstance(annotation, BufferType):
                annotation = BufferBinding(
                    element_type=annotation.element_type,
                    access=annotation.access,
                    storage_class=annotation.storage_class,
                    set=annotation.set,
                    binding=annotation.binding,
                    shape=annotation.shape,
                    alignment=annotation.alignment,
                )
            if isinstance(annotation, BufferBinding):
                b_idx = annotation.binding if annotation.binding is not None else binding_counter
                binding_counter = max(binding_counter + 1, b_idx + 1)
                self.buffer_bindings[parameter.name] = BufferBinding(
                    element_type=annotation.element_type,
                    access=annotation.access,
                    storage_class=annotation.storage_class,
                    set=annotation.set,
                    binding=b_idx,
                    name=parameter.name,
                    shape=annotation.shape,
                    alignment=annotation.alignment,
                )

        arguments = []
        raster = iter(inputs)
        compute_builtins = {
            GlobalInvocationId: "global_invocation_id",
            LocalInvocationId: "local_invocation_id",
            WorkgroupId: "workgroup_id",
        }
        for parameter in inspect.signature(function).parameters.values():
            if parameter.name in self.buffer_bindings:
                b_spec = self.buffer_bindings[parameter.name]
                elem_val = _mlir_element_type(b_spec.element_type)
                ptr_type = f"!gfx.ptr<{elem_val}, global>"
                mode_str = b_spec.access.value if isinstance(b_spec.access, AccessMode) else str(b_spec.access)
                ref, = self.root.raw(
                    "gfx.buffer_binding",
                    [],
                    types=[ptr_type],
                    attributes={
                        "set": {"kind": "literal", "type": "i32", "value": b_spec.set},
                        "binding": {"kind": "literal", "type": "i32", "value": b_spec.binding},
                        "name": {"kind": "string", "value": parameter.name},
                        "access": {"kind": "access_mode", "value": mode_str},
                    },
                )
                ptr_val = self.root.bind(ref, b_spec.element_type)
                self.root.cache[id(ptr_val.node)] = ref
                arguments.append(
                    _BufferProxy(
                        name=parameter.name,
                        ref=ref,
                        ptr=ptr_val,
                        element_type=b_spec.element_type,
                        access=b_spec.access,
                        set=b_spec.set,
                        binding=b_spec.binding,
                        shape=b_spec.shape,
                        alignment=b_spec.alignment,
                    )
                )
                self.structured = True
                continue
            resolved_ann = resolve_annotation(function, parameter.annotation)
            if resolved_ann in compute_builtins:
                if workgroup_size is None:
                    self.fail("Compute invocation/workgroup ID parameters require a @compute entry point.")
                ref, = self.root.raw(
                    "gfx.builtin",
                    [],
                    types=["vector<3xi32>"],
                    attributes={"kind": {"kind": "builtin", "value": compute_builtins[resolved_ann]}},
                )
                arguments.append(self.root.bind(ref, "vector<3xi32>"))
                self.structured = True
                continue
            if parameter.name not in parameters:
                if workgroup_size is not None:
                    self.fail(
                        f"Compute entry parameter {parameter.name!r} requires a Buffer, ParameterBlock, "
                        "GlobalInvocationId, LocalInvocationId, or WorkgroupId annotation."
                    )
                arguments.append(Value(next(raster)))
                continue
            block = parameters[parameter.name]
            values = {}
            for name, typ in block.fields:
                if typ.is_matrix:
                    cols = tuple(
                        self.root.value("gfx.parameter", [], typ.matrix_column_type, {
                            "block": {"kind": "string", "value": block.name},
                            "field": {"kind": "string", "value": f"{name}_{i}"},
                        })
                        for i in range(typ.matrix_shape[0])
                    )
                    values[name] = MatrixValue(cols)
                else:
                    values[name] = self.root.value("gfx.parameter", [], typ, {
                        "block": {"kind": "string", "value": block.name},
                        "field": {"kind": "string", "value": name},
                    })
            arguments.append(_ParameterValues(values))
            self.structured = True
        self.result = self.call(function, arguments, {})

    def name(self):
        result = f"v{self.serial}"
        self.serial += 1
        return result

    def location(self):
        return {"source_file": self.file, "source_line": self.line, "source_column": self.column}

    def fail(self, message):
        error = TypeError(f"{self.file}:{self.line}:{self.column}: {message}")
        error._gfx_source_location = True
        raise error

    def located(self, action):
        try:
            return action()
        except (TypeError, ValueError, IndexError, KeyError, AttributeError, ZeroDivisionError, OverflowError) as error:
            if getattr(error, "_gfx_source_location", False):
                raise
            located = type(error)(f"{self.file}:{self.line}:{self.column}: {error}")
            located._gfx_source_location = True
            raise located from error

    def resolve(self, value):
        while isinstance(value, _Next):
            value = value.action()
        return value

    def lift(self, value, typ=None):
        if isinstance(value, Value):
            return Value(_node(value, typ))
        if typ is None:
            typ = Type.BOOL if isinstance(value, bool) else Type.I32 if isinstance(value, int) else Type.F32
        return Value(_node(value, typ))

    def pair(self, left, right):
        if isinstance(left, _LoopExit) and isinstance(right, _LoopExit):
            a, b = left.returned, right.returned
            if a is None and b is not None:
                a = self.zero(b)
            if b is None and a is not None:
                b = self.zero(a)
            values, rebuild_values = self.pair(left.values, right.values)
            modes, rebuild_modes = self.pair(self.lift(left.mode, Type.I32), self.lift(right.mode, Type.I32))
            returns, rebuild_returns = self.pair(a, b)
            count = len(values)
            return values + modes + returns, lambda items: _LoopExit(
                rebuild_values(items[:count]), rebuild_modes(items[count:count + 1]),
                rebuild_returns(items[count + 1:]),
            )
        if isinstance(left, MatrixValue) and isinstance(right, MatrixValue) and left.type == right.type:
            values, rebuild = self.pair(left.columns, right.columns)
            return values, lambda items: MatrixValue(rebuild(items))
        if isinstance(left, Vertex) and isinstance(right, Vertex):
            values, rebuild = self.pair(
                (left.position, left.color, dict(left.varyings)),
                (right.position, right.color, dict(right.varyings)),
            )
            return values, lambda items: Vertex(*rebuild(items))
        if isinstance(left, dict) and isinstance(right, dict):
            if list(left) != list(right):
                self.fail("Device branches must define the same value/field names.")
            values, rebuild = self.pair(tuple(left.values()), tuple(right.values()))
            return values, lambda items: dict(zip(left, rebuild(items)))
        if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right):
            pairs = [self.pair(a, b) for a, b in zip(left, right)]
            values = [item for items, _ in pairs for item in items]
            def rebuild(items):
                result, offset = [], 0
                for leaves, callback in pairs:
                    result.append(callback(items[offset:offset + len(leaves)]))
                    offset += len(leaves)
                return tuple(result)
            return values, rebuild
        if left is None and right is None:
            return [], lambda items: None
        typ = left.node.type if isinstance(left, Value) else right.node.type if isinstance(right, Value) else None
        try:
            a, b = self.lift(left, typ), self.lift(right, typ)
        except (TypeError, ValueError) as error:
            self.fail(f"Incompatible device branch values: {error}")
        if a.node.type != b.node.type:
            self.fail("Device branches must merge matching scalar/vector types.")
        return [(a, b)], lambda items: items[0]

    def zero(self, value):
        if isinstance(value, Value):
            if value.node.type in (Type.VEC2, Type.VEC3, Type.VEC4):
                from .frontend import vec2, vec3, vec4
                return {Type.VEC2: vec2, Type.VEC3: vec3, Type.VEC4: vec4}[value.node.type](0)
            return self.lift(False if value.node.type == Type.BOOL else 0, value.node.type)
        if isinstance(value, MatrixValue):
            return MatrixValue(self.zero(value.columns))
        if isinstance(value, Vertex):
            return Vertex(self.zero(value.position), self.zero(value.color), self.zero(dict(value.varyings)))
        if isinstance(value, dict):
            return {key: self.zero(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(self.zero(item) for item in value)
        return None if value is None else self.zero(self.lift(value))

    def choose(self, condition, yes, no):
        if not isinstance(condition, Value):
            if type(condition) not in (bool, int, float, str, tuple, dict, type(None)):
                self.fail("Compile-time conditions require scalar or literal collection constants.")
            return yes() if condition else no()
        if condition.node.type != Type.BOOL:
            self.fail("Device conditions require bool values, not numeric truthiness.")
        self.structured = True
        parent = self.builder
        cond = parent.emit(condition.node)
        branches = [_Builder(self, captures=parent.cache) for _ in range(2)]
        try:
            self.builder = branches[0]
            left = self.resolve(yes())
            self.builder = branches[1]
            right = self.resolve(no())
            leaves, rebuild = self.pair(left, right)
            types = []
            for index, branch in enumerate(branches):
                args = [branch.emit(pair[index].node) for pair in leaves]
                branch.raw("scf.yield", args)
                if index == 0:
                    types = [value.node.type for value, _ in leaves]
        finally:
            self.builder = parent
        refs = parent.raw("scf.if", [cond], [typ.schema_name for typ in types], regions=branches)
        return rebuild([parent.bind(ref, typ) for ref, typ in zip(refs, types)])

    def call(self, function, args, kwargs):
        if not callable(function) and not isinstance(function, (_Lambda, _Function)):
            self.fail("Calls must resolve to static Python helpers or registered shader intrinsics.")
        if function is shared_array:
            return self._call_shared_array(args, kwargs)
        if function in _BARRIER_SCOPES:
            return self._call_barrier(function, args, kwargs)
        if function in _ATOMIC_RMW_KINDS:
            return self._call_atomic_rmw(function, args, kwargs)
        if function is atomic_compare_exchange:
            return self._call_atomic_cas(args, kwargs)
        if function in (builtins.abs, minimum, maximum, builtins.min, builtins.max):
            if function is builtins.abs and (len(args) != 1 or type(args[0]) not in (Value, int, float, bool)):
                self.fail("abs requires a scalar or device numeric operand.")
            if function in (builtins.min, builtins.max) and any(isinstance(arg, Value) for arg in args):
                if len(args) != 2 or kwargs:
                    self.fail("Device min/max require two scalar/vector operands.")
                function = minimum if function is builtins.min else maximum
            return function(*args, **kwargs)
        if function in (builtins.int, builtins.float) and len(args) == 1 and isinstance(args[0], Value):
            value = args[0]
            if kwargs:
                self.fail("Device numeric conversions take one positional operand.")
            target = Type.I32 if function is builtins.int else Type.F32
            if value.node.type == target:
                return value
            if value.node.type == Type.U32 and target == Type.I32:
                return i32(value)
            op = {
                (Type.F32, Type.I32): "arith.fptosi",
                (Type.I32, Type.F32): "arith.sitofp",
                (Type.U32, Type.F32): "arith.uitofp",
            }.get((value.node.type, target))
            if op is None:
                self.fail("Unsupported explicit scalar numeric conversion.")
            self.structured = True
            return self.builder.value(op, [value], target)
        module = getattr(function, "__module__", "")
        bound = getattr(function, "__self__", None)
        intrinsic = function in _INTRINSICS if isinstance(function, (FunctionType, type)) else False
        if intrinsic or isinstance(bound, Value) and getattr(function, "__func__", None) is Value.with_components:
            def callback(arg):
                if isinstance(arg, (FunctionType, _Lambda, _Function)):
                    return lambda *a, **k: self.call(arg, list(a), k)
                return arg
            return function(*(callback(arg) for arg in args),
                            **{key: callback(arg) for key, arg in kwargs.items()})
        if module == "math" and all(isinstance(arg, (int, float)) for arg in args):
            return function(*args, **kwargs)
        if function in (builtins.int, builtins.float, builtins.len, builtins.tuple):
            if function in (builtins.int, builtins.float):
                if not all(type(arg) in (int, float, bool, str) for arg in args):
                    self.fail("Compile-time numeric conversion requires literal scalar operands.")
            elif len(args) != 1 or type(args[0]) not in (tuple, list, dict, str, range):
                self.fail("Compile-time collection operations require literal collections.")
            return function(*args, **kwargs)
        if function is builtins.range:
            if kwargs or not all(type(arg) is int for arg in args):
                self.fail("Device range bounds are only valid in a direct for ... in range(...) loop.")
            return range(*args)
        if function is builtins.reversed and len(args) == 1 and isinstance(args[0], (tuple, range)) and not kwargs:
            return tuple(reversed(args[0]))
        if isinstance(function, _Lambda):
            if kwargs or len(args) != len(function.parameters):
                self.fail("Lambda helper arguments must match its positional parameters.")
            previous = self.file, self.line, self.column, self.offset, self.indent, self.scope
            self.file, self.offset, self.indent = function.file, function.offset, function.indent
            self.scope = object()
            try:
                return self.expression(function.body, {**function.env, **dict(zip(function.parameters, args))})
            finally:
                self.file, self.line, self.column, self.offset, self.indent, self.scope = previous
        if isinstance(function, _Function):
            if function in self.stack:
                self.fail("Recursive device helper calls are unsupported.")
            parameters = function.definition.args
            positional = [arg.arg for arg in (*parameters.posonlyargs, *parameters.args)]
            if len(args) > len(positional):
                self.fail("Too many positional helper arguments.")
            values = dict(zip(positional, args))
            valid_keywords = {arg.arg for arg in (*parameters.args, *parameters.kwonlyargs)}
            for name, value in kwargs.items():
                if name not in valid_keywords or name in values:
                    self.fail("Unknown, positional-only, or duplicate helper keyword argument.")
                values[name] = value
            for name in (*positional, *(arg.arg for arg in parameters.kwonlyargs)):
                if name not in values:
                    if name not in function.defaults:
                        self.fail(f"Missing helper argument {name!r}.")
                    values[name] = function.defaults[name]
            previous = self.file, self.line, self.column, self.offset, self.indent, self.scope
            self.file, self.offset, self.indent = function.file, function.offset, function.indent
            self.scope = object()
            self.stack.append(function)
            try:
                return self.execute(function.definition.body, {**function.env, **values},
                                    lambda env: None, lambda value, env: value)
            finally:
                self.stack.pop()
                self.file, self.line, self.column, self.offset, self.indent, self.scope = previous
        if not isinstance(function, FunctionType):
            self.fail("Calls must resolve to static Python helpers or registered shader intrinsics.")
        if function in self.stack:
            self.fail("Recursive device helper calls are unsupported.")
        try:
            lines, start = inspect.getsourcelines(function)
        except (OSError, TypeError) as error:
            self.fail(f"Helper source is unavailable; define it in a Python source file: {error}")
        parsed = ast.parse(textwrap.dedent("".join(lines)))
        definitions = [item for item in parsed.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
            self.fail("Only ordinary, statically defined device helper functions are supported.")
        definition = definitions[0]
        try:
            parameters = inspect.signature(function).bind(*args, **kwargs)
        except TypeError as error:
            self.fail(str(error))
        parameters.apply_defaults()
        closure = inspect.getclosurevars(function)
        env = {**vars(builtins), **function.__globals__, **closure.nonlocals, **parameters.arguments}
        for name in function.__code__.co_varnames:
            if name not in parameters.arguments:
                env.pop(name, None)
        previous = self.file, self.line, self.column, getattr(self, "offset", 0), getattr(self, "indent", 0), self.scope
        self.file = function.__code__.co_filename
        self.offset = start - 1
        self.indent = len(lines[0]) - len(lines[0].lstrip())
        self.scope = object()
        self.stack.append(function)
        try:
            return self.execute(definition.body, env, lambda env: None, lambda value, env: value)
        finally:
            self.stack.pop()
            self.file, self.line, self.column, self.offset, self.indent, self.scope = previous

    def expression(self, node, env):
        self.line = self.offset + node.lineno
        self.column = self.indent + node.col_offset + 1
        return self.located(lambda: self._expression(node, env))

    def _expression(self, node, env):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                self.fail(f"Unresolved or possibly unbound local name {node.id!r}.")
            return env[node.id]
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                self.fail("Private attributes and Python introspection are unsupported in device source.")
            owner = self.expression(node.value, env)
            if not isinstance(owner, (Value, Node, ModuleType, Type, Vertex, Entry, Pipeline, ComputePipeline, _ParameterValues, MatrixValue)) and owner is not Type:
                self.fail("Dynamic attributes on arbitrary Python objects are unsupported.")
            return getattr(owner, node.attr)
        if isinstance(node, (ast.Tuple, ast.List)):
            return tuple(self.expression(item, env) for item in node.elts)
        if isinstance(node, ast.Dict):
            result = {}
            for key, value in zip(node.keys, node.values):
                if key is None:
                    fields = self.expression(value, env)
                    if type(fields) is not dict:
                        self.fail("Mapping expansion requires a compile-time field dictionary.")
                    result.update(fields)
                else:
                    result[self.expression(key, env)] = self.expression(value, env)
            return result
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            values = []
            def expand(index, scope):
                if index == len(node.generators):
                    values.append(self.expression(node.elt, scope))
                    return
                generator = node.generators[index]
                iterable = self.expression(generator.iter, scope)
                if generator.is_async or type(iterable) not in (tuple, list, range):
                    self.fail("Comprehensions require compile-time literal collections/ranges.")
                for item in iterable:
                    nested = dict(scope)
                    self.assign(generator.target, item, nested)
                    included = True
                    for condition in generator.ifs:
                        test = self.expression(condition, nested)
                        if isinstance(test, Value):
                            self.fail("Device-dependent comprehension filters cannot determine a static collection shape.")
                        if not test:
                            included = False
                            break
                    if included:
                        expand(index + 1, nested)
            expand(0, dict(env))
            return tuple(values)
        if isinstance(node, ast.Subscript):
            owner = self.expression(node.value, env)
            if isinstance(owner, _BufferProxy):
                access_str = owner.access.value if hasattr(owner.access, "value") else str(owner.access)
                if access_str == "write_only":
                    self.fail(f"Cannot read from write-only buffer '{owner.name}'.")
                index = self.expression(node.slice, env)
                return self._memory_load(owner, index)
            if isinstance(owner, _SharedArrayProxy):
                index = self.expression(node.slice, env)
                return self._memory_load(owner, index)
            if type(owner) not in (Value, MatrixValue, tuple, list, dict, str, range):
                self.fail("Subscripts require device vectors or compile-time literal collections.")
            return owner[self.expression(node.slice, env)]
        if isinstance(node, ast.Lambda):
            if node.args.vararg or node.args.kwarg or node.args.defaults or node.args.kwonlyargs:
                self.fail("Lambda helpers require plain positional arguments.")
            parameters = tuple(arg.arg for arg in (*node.args.posonlyargs, *node.args.args))
            captured = frozenset(item.id for item in ast.walk(node.body)
                                 if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                                 and item.id not in parameters)
            return _Lambda(parameters, node.body, dict(env), captured, self.scope,
                           self.file, self.offset, self.indent)
        if isinstance(node, ast.Call):
            function = self.expression(node.func, env)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    value = self.expression(arg.value, env)
                    if not isinstance(value, tuple):
                        self.fail("Argument expansion requires a compile-time tuple.")
                    args.extend(value)
                else:
                    args.append(self.expression(arg, env))
            if any(keyword.arg is None for keyword in node.keywords):
                self.fail("Dynamic keyword expansion is unsupported.")
            kwargs = {keyword.arg: self.expression(keyword.value, env) for keyword in node.keywords}
            return self.call(function, args, kwargs)
        if isinstance(node, ast.BinOp):
            left, right = self.expression(node.left, env), self.expression(node.right, env)
            if any(type(value) not in (Value, MatrixValue, int, float, bool, str, tuple) for value in (left, right)):
                self.fail("Binary operations require device values or literal scalar/tuple constants.")
            operation = {
                ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
                ast.LShift: operator.lshift, ast.RShift: operator.rshift,
                ast.BitAnd: operator.and_, ast.BitOr: operator.or_,
                ast.BitXor: operator.xor,
            }.get(type(node.op))
            if isinstance(node.op, ast.Pow):
                return shader_math.pow(left, right) if isinstance(left, Value) or isinstance(right, Value) else left ** right
            if operation is None:
                self.fail(f"Unsupported binary operation {type(node.op).__name__}.")
            return operation(left, right)
        if isinstance(node, ast.UnaryOp):
            value = self.expression(node.operand, env)
            if isinstance(node.op, ast.Not):
                if isinstance(value, Value):
                    if value.node.type != Type.BOOL:
                        self.fail("Device not requires a bool operand.")
                    return ~value
                return not value
            operation = {ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Invert: operator.invert}.get(type(node.op))
            if operation is None:
                self.fail("Unsupported unary operation.")
            return operation(value)
        if isinstance(node, ast.IfExp):
            return self.choose(self.expression(node.test, env),
                               lambda: self.expression(node.body, env), lambda: self.expression(node.orelse, env))
        if isinstance(node, ast.BoolOp):
            def combine(index):
                value = self.expression(node.values[index], env)
                if index == len(node.values) - 1:
                    return value
                return self.choose(value, lambda: combine(index + 1), lambda: value) if isinstance(node.op, ast.And) else \
                    self.choose(value, lambda: value, lambda: combine(index + 1))
            return combine(0)
        if isinstance(node, ast.Compare):
            operations = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
                          ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge}
            def compare(index, left):
                right = self.expression(node.comparators[index], env)
                operation = operations.get(type(node.ops[index]))
                if operation is None:
                    self.fail("Only scalar numeric/Boolean comparisons are supported.")
                value = operation(left, right)
                return value if index == len(node.ops) - 1 else \
                    self.choose(value, lambda: compare(index + 1, right), lambda: False)
            return compare(0, self.expression(node.left, env))
        self.fail(f"Unsupported expression: {type(node).__name__}.")

    def assign(self, target, value, env):
        if isinstance(target, ast.Name):
            self.check_capture(target.id, env)
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, tuple) and len(target.elts) == len(value):
            for child, item in zip(target.elts, value):
                self.assign(child, item, env)
        elif isinstance(target, ast.Subscript):
            owner = self.expression(target.value, env)
            if isinstance(owner, _BufferProxy):
                access_str = owner.access.value if hasattr(owner.access, "value") else str(owner.access)
                if access_str == "read_only":
                    self.fail(f"Cannot write to read-only buffer '{owner.name}'.")
                index = self.expression(target.slice, env)
                self._memory_store(owner, index, value)
                return
            if isinstance(owner, _SharedArrayProxy):
                index = self.expression(target.slice, env)
                self._memory_store(owner, index, value)
                return
            self.fail("Device assignments require local names; attribute/subscript mutation is unsupported.")
        else:
            self.fail("Device assignments require local names; attribute/subscript mutation is unsupported.")

    def _call_shared_array(self, args, kwargs) -> _SharedArrayProxy:
        if self.workgroup_size is None:
            self.fail("shared_array() is only allowed inside a @compute kernel.")
        if len(args) != 2 or set(kwargs) - {"name"}:
            self.fail("shared_array() expects (element_type, count_or_shape, *, name=None).")
        elem_type = _normalize_buffer_element_type(args[0])
        raw_shape = args[1]
        if type(raw_shape) is int and not isinstance(raw_shape, bool):
            if raw_shape <= 0:
                self.fail(f"shared_array count must be positive, got {raw_shape}.")
            shape = (raw_shape,)
            count = raw_shape
        elif isinstance(raw_shape, (tuple, list)) and len(raw_shape) >= 1:
            if any(type(d) is not int or isinstance(d, bool) or d <= 0 for d in raw_shape):
                self.fail(f"shared_array shape dimensions must be positive integers, got {raw_shape!r}.")
            shape = tuple(raw_shape)
            count = 1
            for d in shape:
                count *= d
        else:
            self.fail("shared_array size must be a positive compile-time integer or shape tuple.")

        sym_name = kwargs.get("name")
        if sym_name is not None and (not isinstance(sym_name, str) or not sym_name.isidentifier()):
            self.fail(f"shared_array name must be a valid identifier, got {sym_name!r}.")

        elem_schema = _mlir_element_type(elem_type)
        ptr_type = f"!gfx.ptr<{elem_schema}, workgroup>"
        attrs = {"count": {"kind": "literal", "type": "i32", "value": count}}
        if sym_name is not None:
            attrs["name"] = {"kind": "string", "value": sym_name}
        ref, = self.root.raw("gfx.shared_alloc", [], types=[ptr_type], attributes=attrs)
        self.structured = True
        return _SharedArrayProxy(
            name=sym_name or ref.name,
            ref=ref,
            element_type=elem_type,
            count=count,
            shape=shape,
        )

    def _call_barrier(self, function, args, kwargs) -> None:
        if self.workgroup_size is None:
            self.fail(f"{function.__name__}() is only allowed inside a @compute kernel.")
        if args or kwargs:
            self.fail(f"{function.__name__}() takes no arguments.")
        scope = _BARRIER_SCOPES[function]
        self.builder.raw(
            "gfx.barrier",
            [],
            [],
            attributes={"scope": {"kind": "memory_scope", "value": scope}},
        )
        self.structured = True
        return None

    def _validate_atomic_target(self, target) -> None:
        if not isinstance(target, (_BufferProxy, _SharedArrayProxy)):
            self.fail("Atomic target must be a Buffer or shared_array.")
        if isinstance(target, _BufferProxy):
            access_str = target.access.value if hasattr(target.access, "value") else str(target.access)
            if access_str != "read_write":
                self.fail(
                    f"Atomic operations require a read_write buffer, got '{access_str}' on '{target.name}'."
                )
        if target.element_type not in (Type.I32, Type.U32):
            self.fail(
                f"Atomic operations require Type.I32 or Type.U32 elements, got {target.element_type!r}."
            )

    def _coerce_atomic_int_operand(self, operand, expected_type: Type) -> _Ref:
        if isinstance(operand, Value):
            if operand.type not in (Type.I32, Type.U32):
                self.fail(f"Atomic integer operand must be i32 or u32, got {operand.type}.")
            val = (
                operand
                if operand.type == expected_type
                else (i32(operand) if expected_type == Type.I32 else u32(operand))
            )
        elif type(operand) is int and not isinstance(operand, bool):
            val = self.lift(operand, expected_type)
        else:
            self.fail(f"Atomic integer operand must be int or device i32/u32, got {type(operand).__name__}.")
        return self.builder.emit(val.node)

    def _call_atomic_rmw(self, function, args, kwargs) -> Value:
        if kwargs or len(args) != 3:
            self.fail(f"{function.__name__}() expects (target, index, value).")
        target, index, value = args
        self._validate_atomic_target(target)
        off_ref = self._element_ptr_ref(target, index)
        val_ref = self._coerce_atomic_int_operand(value, target.element_type)
        kind = _ATOMIC_RMW_KINDS[function]
        if target.element_type == Type.U32 and kind in ("min", "max"):
            kind = "u" + kind
        res_ref, = self.builder.raw(
            "gfx.atomic_rmw",
            [off_ref, val_ref],
            types=["i32"],
            attributes={"kind": {"kind": "atomic_op", "value": kind}},
        )
        self.structured = True
        return self.builder.bind(res_ref, target.element_type)

    def _call_atomic_cas(self, args, kwargs) -> Value:
        if kwargs or len(args) != 4:
            self.fail("atomic_compare_exchange() expects (target, index, expected, desired).")
        target, index, expected, desired = args
        self._validate_atomic_target(target)
        off_ref = self._element_ptr_ref(target, index)
        cmp_ref = self._coerce_atomic_int_operand(expected, target.element_type)
        val_ref = self._coerce_atomic_int_operand(desired, target.element_type)
        res_ref, = self.builder.raw(
            "gfx.atomic_cas",
            [off_ref, cmp_ref, val_ref],
            types=["i32"],
        )
        self.structured = True
        return self.builder.bind(res_ref, target.element_type)

    def _linearize_index(self, owner: _BufferProxy | _SharedArrayProxy, index: Any) -> _Ref:
        if isinstance(index, tuple):
            if owner.shape is None or len(index) != len(owner.shape):
                expected_rank = len(owner.shape) if owner.shape is not None else 1
                self.fail(
                    f"Multi-dimensional index rank {len(index)} does not match shape rank {expected_rank} "
                    f"for '{owner.name}'."
                )
            coords: list[Value] = []
            target_int_type = Type.I32
            for item in index:
                if isinstance(item, Value):
                    if item.type not in (Type.I32, Type.U32):
                        self.fail(f"Subscript coordinate must be an integer, got {item.type}")
                    if item.type == Type.U32:
                        target_int_type = Type.U32
                    coords.append(item)
                elif type(item) is int and not isinstance(item, bool):
                    coords.append(self.lift(item, Type.I32))
                else:
                    self.fail(f"Subscript coordinate must be an integer, got {type(item).__name__}")
            norm_coords = [
                c if c.type == target_int_type else (u32(c) if target_int_type == Type.U32 else i32(c))
                for c in coords
            ]
            linear = norm_coords[0]
            for coord, dim in zip(norm_coords[1:], owner.shape[1:]):
                linear = linear * dim + coord
            return self.builder.emit(linear.node)

        kind = "Shared array" if isinstance(owner, _SharedArrayProxy) else "Buffer"
        idx_val = self.lift(index, Type.I32) if not isinstance(index, Value) else index
        if idx_val.type not in (Type.I32, Type.U32):
            self.fail(f"{kind} subscript index must be an integer, got {idx_val.type}")
        return self.builder.emit(idx_val.node)

    def _element_ptr_ref(self, owner: _BufferProxy | _SharedArrayProxy, index: Any) -> _Ref:
        idx_ref = self._linearize_index(owner, index)
        elem_schema = owner.element_schema_name
        ptr_type = f"!gfx.ptr<{elem_schema}, {owner.storage_class}>"
        off_ref, = self.builder.raw(
            "gfx.ptr_offset",
            [owner.ref, idx_ref],
            types=[ptr_type],
        )
        return off_ref

    def _memory_load(self, owner: _BufferProxy | _SharedArrayProxy, index: Any) -> Value:
        off_ref = self._element_ptr_ref(owner, index)
        elem_schema = owner.element_schema_name
        load_attrs = {}
        if owner.alignment:
            load_attrs["alignment"] = {"kind": "literal", "type": "i32", "value": owner.alignment}
        load_ref, = self.builder.raw(
            "gfx.load",
            [off_ref],
            types=[elem_schema],
            attributes=load_attrs if load_attrs else None,
        )
        return self.builder.bind(load_ref, owner.element_type)

    def _memory_store(self, owner: _BufferProxy | _SharedArrayProxy, index: Any, value: Any) -> None:
        val = self.lift(value, owner.element_type) if not isinstance(value, Value) else value
        elem_type = owner.element_type
        if isinstance(elem_type, Type) and val.type != elem_type:
            val_val = val.type.value if isinstance(val.type, Type) else str(val.type)
            elem_val = elem_type.value if isinstance(elem_type, Type) else str(elem_type)
            kind_label = "shared_array" if isinstance(owner, _SharedArrayProxy) else "Buffer"
            self.fail(f"Type mismatch: cannot store {val_val} into {kind_label}[{elem_val}]")

        val_ref = self.builder.emit(val.node)
        off_ref = self._element_ptr_ref(owner, index)
        store_attrs = {}
        if owner.alignment:
            store_attrs["alignment"] = {"kind": "literal", "type": "i32", "value": owner.alignment}
        self.builder.raw(
            "gfx.store",
            [val_ref, off_ref],
            types=[],
            attributes=store_attrs if store_attrs else None,
        )

    def check_capture(self, name, env):
        if any(isinstance(value, (_Function, _Lambda)) and value.owner is self.scope
               and name in value.captured
               for value in env.values()):
            self.fail(f"Captured local {name!r} cannot be rebound; pass updated state as an explicit helper argument.")

    def execute(self, statements, env, finish, returned, broken=None, continued=None):
        return self.resolve(self.plan(statements, env, finish, returned, broken, continued))

    def plan(self, statements, env, finish, returned, broken=None, continued=None):
        return _Next(lambda: self.step(statements, env, finish, returned, broken, continued))

    def step(self, statements, env, finish, returned, broken=None, continued=None):
        if not statements:
            return finish(env)
        node, *rest = statements
        self.line, self.column = self.offset + node.lineno, self.indent + node.col_offset + 1
        return self.located(lambda: self._execute(node, rest, env, finish, returned, broken, continued))

    def _execute(self, node, rest, env, finish, returned, broken, continued):
        if isinstance(node, ast.Return):
            return returned(self.expression(node.value, env) if node.value else None, env)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:
                self.fail("Local annotations require an initializer.")
            value = self.expression(node.value, env)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                self.assign(target, value, env)
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.args.vararg or node.args.kwarg:
                self.fail("Local helpers must be static definitions without decorators or variadic arguments.")
            positional = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
            defaults = {name: self.expression(value, env) for name, value in
                        zip(positional[len(positional) - len(node.args.defaults):], node.args.defaults)}
            defaults.update({arg.arg: self.expression(value, env) for arg, value in
                             zip(node.args.kwonlyargs, node.args.kw_defaults) if value is not None})
            locals = set(positional) | {arg.arg for arg in node.args.kwonlyargs} | {
                item.id for statement in node.body for item in ast.walk(statement)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)
            }
            captured = frozenset(
                item.id for statement in node.body for item in ast.walk(statement)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
                and item.id not in locals and item.id != node.name
            )
            helper = _Function(node, dict(env), defaults, self.file, self.offset, self.indent, captured, self.scope)
            helper.env[node.name] = helper
            env[node.name] = helper
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.target, (ast.Name, ast.Subscript)):
                self.fail("Augmented assignment requires a local name or subscript target.")
            expression = ast.copy_location(ast.BinOp(node.target, node.op, node.value), node)
            self.assign(node.target, self.expression(expression, env), env)
        elif isinstance(node, ast.If):
            cond = self.expression(node.test, env)
            if (
                isinstance(cond, Value)
                and rest
                and not _has_nonlocal_jump(node.body)
                and not _has_nonlocal_jump(node.orelse)
            ):
                if cond.node.type != Type.BOOL:
                    self.fail("Device conditions require bool values, not numeric truthiness.")
                parent = self.builder
                branches = [_Builder(self, captures=parent.cache) for _ in range(2)]
                try:
                    self.builder = branches[0]
                    then_env = self.resolve(
                        self.plan(node.body, dict(env), lambda e: e, returned, broken, continued)
                    )
                    self.builder = branches[1]
                    else_env = self.resolve(
                        self.plan(node.orelse, dict(env), lambda e: e, returned, broken, continued)
                    )
                finally:
                    self.builder = parent
                unmergeable = (_Function, _Lambda, _BufferProxy, _SharedArrayProxy)
                common_keys = [k for k in then_env if k in else_env]
                changed_keys = [
                    k for k in common_keys
                    if then_env[k] is not else_env[k] or k not in env
                ]
                if not any(
                    isinstance(then_env[k], unmergeable) or isinstance(else_env[k], unmergeable)
                    for k in changed_keys
                ):
                    self.structured = True
                    cond_ref = parent.emit(cond.node)
                    left_updates = {k: then_env[k] for k in changed_keys}
                    right_updates = {k: else_env[k] for k in changed_keys}
                    leaves, rebuild = self.pair(left_updates, right_updates)
                    types = []
                    for index, branch in enumerate(branches):
                        args = [branch.emit(pair[index].node) for pair in leaves]
                        branch.raw("scf.yield", args)
                        if index == 0:
                            types = [value.node.type for value, _ in leaves]
                    refs = parent.raw(
                        "scf.if", [cond_ref], [typ.schema_name for typ in types], regions=branches
                    )
                    merged_updates = rebuild(
                        [parent.bind(ref, typ) for ref, typ in zip(refs, types)]
                    )
                    merged_env = dict(env)
                    merged_env.update(merged_updates)
                    return self.plan(rest, merged_env, finish, returned, broken, continued)
            return self.choose(cond,
                               lambda: self.plan(node.body + rest, dict(env), finish, returned, broken, continued),
                               lambda: self.plan(node.orelse + rest, dict(env), finish, returned, broken, continued))
        elif isinstance(node, (ast.For, ast.While)):
            return self.loop(node, env, lambda values: self.plan(rest, values, finish, returned, broken, continued), returned)
        elif isinstance(node, ast.Break):
            if broken is None:
                self.fail("break requires an enclosing device loop.")
            return broken(env)
        elif isinstance(node, ast.Continue):
            if continued is None:
                self.fail("continue requires an enclosing device loop.")
            return continued(env)
        elif isinstance(node, ast.Pass) or isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            pass
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self.expression(node.value, env)
        else:
            self.fail(f"Unsupported statement: {type(node).__name__}; mutation, exceptions, and dynamic Python execution are not device operations.")
        return self.plan(rest, env, finish, returned, broken, continued)

    def loop(self, node, env, continuation, returned):
        if node.orelse:
            self.fail("Loop else clauses are unsupported; express the termination result explicitly.")
        is_for = isinstance(node, ast.For)
        if is_for and not (isinstance(node.iter, ast.Call) and
                           self.expression(node.iter.func, env) is builtins.range):
            iterable = self.expression(node.iter, env)
            if not isinstance(iterable, (tuple, range)):
                self.fail("Static metaprogramming iteration requires a compile-time tuple/range, not a dynamic iterable.")
            items = tuple(iterable)
            def iteration(index, values):
                if index == len(items):
                    return continuation(values)
                self.assign(node.target, items[index], values)
                advance = lambda values: _Next(lambda: iteration(index + 1, values))
                return self.plan(node.body, values, advance, returned, continuation, advance)
            return _Next(lambda: iteration(0, dict(env)))
        self.structured = True
        parent = self.builder
        index = limit = step = None
        if is_for:
            if not isinstance(node.target, ast.Name) or not isinstance(node.iter, ast.Call) or \
                    self.expression(node.iter.func, env) is not builtins.range or node.iter.keywords:
                self.fail("Device for loops require a local induction name and range().")
            args = [self.expression(arg, env) for arg in node.iter.args]
            if not 1 <= len(args) <= 3:
                self.fail("range requires one to three scalar bounds.")
            lower, upper, stride = (0, args[0], 1) if len(args) == 1 else (*args, 1) if len(args) == 2 else args
            if type(stride) is not int or not stride or not -(2**31) <= stride < 2**31:
                self.fail("range step must be a nonzero compile-time signed 32-bit integer.")
            bounds = [
                i32(value) if isinstance(value, Value) and value.node.type == Type.U32 else self.lift(value, Type.I32)
                for value in (lower, upper, stride)
            ]
            index, limit, step = [parent.raw("arith.index_cast", [parent.emit(value.node)], ["index"])[0] for value in bounds]
        assigned = {item.id for statement in node.body for item in ast.walk(statement)
                    if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store)}
        if is_for:
            self.check_capture(node.target.id, env)
            env = dict(env)
            if node.target.id not in env and type(lower) is int and type(upper) is int and \
                    (lower < upper if stride > 0 else lower > upper):
                env[node.target.id] = self.lift(lower, Type.I32)
            assigned.add(node.target.id)
        names = [name for name in env if name in assigned]
        initial = {name: self.lift(env[name]) for name in names}
        types = [value.node.type for value in initial.values()]
        initial_refs = [parent.emit(value.node) for value in initial.values()]
        if is_for:
            initial_refs.append(index)
        argument_types = [value.type for value in initial_refs]
        arguments = [{"name": self.name(), "type": typ} for typ in argument_types]
        after = _Builder(self, arguments, parent.cache)
        local = dict(env)
        for name, arg, typ in zip(names, arguments, types):
            local[name] = after.bind(_Ref(arg["name"], arg["type"]), typ)
        if is_for:
            iv = _Ref(arguments[-1]["name"], "index")
            converted, = after.raw("arith.index_cast", [iv], ["i32"])
            local[node.target.id] = after.bind(converted, Type.I32)
        def exit(values, mode=0, value=None):
            return _LoopExit({name: self.lift(values[name], typ) for name, typ in zip(names, types)}, mode, value)
        try:
            self.builder = after
            outcome = self.execute(node.body, local, exit, lambda value, values: exit(values, 2, value),
                                   lambda values: exit(values, 1), lambda values: exit(values))
            leaves, rebuild_return = self.pair(outcome.returned, outcome.returned)
            return_types = [value.node.type for value, _ in leaves]
            yielded = [after.emit(outcome.values[name].node) for name in names]
            if is_for:
                advanced, = after.raw("arith.addi", [iv, step], ["index"])
                yielded.append(advanced)
                distance, = after.raw("arith.subi", [limit, iv] if stride > 0 else [iv, limit], ["index"])
                zero_index, = after.raw("arith.constant", types=["index"],
                                       attributes={"value": {"kind": "literal", "type": "index", "value": 0}})
                magnitude = step if stride > 0 else after.raw("arith.subi", [zero_index, step], ["index"])[0]
                ref, = after.raw("arith.cmpi", [distance, magnitude], ["bool"],
                                 {"predicate": {"kind": "cmpi", "value": "ugt"}})
                can_advance = after.bind(ref, Type.BOOL)
                mode = self.lift(outcome.mode, Type.I32)
                # Stop before a signed 32-bit induction wrap on SPIR-V's
                # 32-bit index lowering; normal range completion is not break.
                outcome.mode = select((mode == 0) & ~can_advance, self.lift(4, Type.I32), mode)
            yielded.append(after.emit(self.lift(outcome.mode, Type.I32).node))
            yielded.extend(after.emit(value.node) for value, _ in leaves)
            after.raw("scf.yield", yielded)
        finally:
            self.builder = parent
        mode0 = parent.emit(self.lift(0, Type.I32).node)
        initial_refs.append(mode0)
        zero = self.zero(outcome.returned)
        zero_leaves, _ = self.pair(zero, zero)
        initial_refs.extend(parent.emit(value.node) for value, _ in zero_leaves)
        tail_types = ["i32", *(typ.schema_name for typ in return_types)]
        after.block["arguments"].extend({"name": self.name(), "type": typ} for typ in tail_types)
        before_args = [{"name": self.name(), "type": ref.type} for ref in initial_refs]
        before = _Builder(self, before_args, parent.cache)
        condition_env = dict(env)
        for name, arg, typ in zip(names, before_args, types):
            condition_env[name] = before.bind(_Ref(arg["name"], arg["type"]), typ)
        mode_index = len(names) + int(is_for)
        mode = before.bind(_Ref(before_args[mode_index]["name"], "i32"), Type.I32)
        try:
            self.builder = before
            if is_for:
                ref, = before.raw("arith.cmpi", [_Ref(before_args[len(names)]["name"], "index"), limit],
                                  ["bool"], {"predicate": {"kind": "cmpi", "value": "slt" if stride > 0 else "sgt"}})
                condition = before.bind(ref, Type.BOOL) & (mode == 0)
            else:
                condition = self.choose(mode == 0, lambda: self.expression(node.test, condition_env), lambda: False)
                condition = self.lift(condition, Type.BOOL)
            before.raw("scf.condition", [before.emit(condition.node), *[_Ref(arg["name"], arg["type"]) for arg in before_args]])
        finally:
            self.builder = parent
        refs = parent.raw("scf.while", initial_refs, [ref.type for ref in initial_refs], regions=[before, after])
        final_env = dict(env)
        for name, ref, typ in zip(names, refs, types):
            final_env[name] = parent.bind(ref, typ)
        mode = parent.bind(refs[mode_index], Type.I32)
        if outcome.returned is None:
            return continuation(final_env)
        result = rebuild_return([parent.bind(ref, typ) for ref, typ in zip(refs[mode_index + 1:], return_types)])
        return self.choose(mode == 2, lambda: returned(result, final_env), lambda: continuation(final_env))

    def module(self, name, kind, outputs, interfaces, workgroup_size=None):
        if kind == "compute":
            self.root.raw("func.return")
            wg_size = workgroup_size or self.workgroup_size
            func_obj = {
                "name": name, "stage": kind, "results": [],
                "body": {"blocks": [self.root.block]}, **self.location(),
            }
            if wg_size:
                func_obj["workgroup_size"] = list(wg_size)
            result = {"schema_version": 2, **self.location(), "functions": [func_obj]}
            if self.parameters:
                result["parameters"] = self.parameters.to_dict()
            return result
        for node, interface in zip(outputs, interfaces):
            ref = self.root.emit(node)
            if interface.builtin == "position":
                self.root.raw("gfx.position", [ref])
            else:
                self.root.raw("gfx.output", [ref], attributes={
                    "location": {"kind": "literal", "type": "i32", "value": interface.location},
                    "name": {"kind": "string", "value": interface.name},
                    "interpolation": {"kind": "interpolation", "value": interface.interpolation},
                })
        self.root.raw("func.return")
        result = {"schema_version": 2, **self.location(), "functions": [{
            "name": name, "stage": kind, "results": [],
            "body": {"blocks": [self.root.block]}, **self.location(),
        }]}
        if self.parameters:
            result["parameters"] = self.parameters.to_dict()
        return result


@dataclass(frozen=True, eq=False)
class _Lambda:
    parameters: tuple[str, ...]
    body: ast.expr
    env: dict
    captured: frozenset[str]
    owner: object
    file: str
    offset: int
    indent: int


@dataclass(frozen=True, eq=False)
class _Function:
    definition: ast.FunctionDef
    env: dict
    defaults: dict
    file: str
    offset: int
    indent: int
    captured: frozenset[str]
    owner: object
