from __future__ import annotations

import inspect
import math
import re
import struct
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence


class Type(Enum):
    F32 = "f32"
    U32 = "i32"
    I32 = "si32"
    F16 = "f16"
    U16 = "i16"
    I16 = "si16"
    F64 = "f64"
    U64 = "i64"
    I64 = "si64"
    U8 = "i8"
    I8 = "si8"
    BOOL = "i1"
    VEC2 = "vector<2xf32>"
    VEC3 = "vector<3xf32>"
    VEC4 = "vector<4xf32>"
    MAT2X2 = "mat2x2"
    MAT2X3 = "mat2x3"
    MAT2X4 = "mat2x4"
    MAT3X2 = "mat3x2"
    MAT3X3 = "mat3x3"
    MAT3X4 = "mat3x4"
    MAT4X2 = "mat4x2"
    MAT4X3 = "mat4x3"
    MAT4X4 = "mat4x4"

    @property
    def matrix_shape(self) -> tuple[int, int] | None:
        """Return (columns, rows) for matrix types, or None for non-matrix types.

        Naming follows GLSL/WGSL: ``matCxR`` has C columns of R-component vectors.
        """
        match = re.fullmatch(r"mat(\d)x(\d)", self.value)
        return (int(match.group(1)), int(match.group(2))) if match else None

    @property
    def is_matrix(self) -> bool:
        return self.matrix_shape is not None

    @property
    def matrix_column_type(self) -> Type:
        """Vector type of one column of a matrix type."""
        shape = self.matrix_shape
        if shape is None:
            raise TypeError(f"{self.name} is not a matrix type.")
        return {2: Type.VEC2, 3: Type.VEC3, 4: Type.VEC4}[shape[1]]

    @property
    def matrix_layout(self) -> tuple[int, int, int]:
        """Return (size, alignment, column_stride) in bytes using std430 rules.

        Columns are laid out like an array of vecR, so each column is padded to the
        vector alignment (vec2: 8, vec3/vec4: 16).
        """
        shape = self.matrix_shape
        if shape is None:
            raise TypeError(f"{self.name} is not a matrix type.")
        columns, rows = shape
        stride = 8 if rows == 2 else 16
        return (columns * stride, stride, stride)

    @property
    def schema_name(self) -> str:
        shape = self.matrix_shape
        if shape is not None:
            return f"mat{shape[0]}x{shape[1]}"
        return {
            Type.F32: "f32", Type.U32: "u32", Type.I32: "i32",
            Type.F16: "f16", Type.U16: "u16", Type.I16: "i16",
            Type.F64: "f64", Type.U64: "u64", Type.I64: "i64",
            Type.U8: "u8", Type.I8: "i8", Type.BOOL: "bool",
            Type.VEC2: "f32x2", Type.VEC3: "f32x3", Type.VEC4: "f32x4",
        }[self]

    @classmethod
    def from_schema(cls, name: str) -> Type:
        for typ in cls:
            if typ.schema_name == name:
                return typ
        raise ValueError(f"Unsupported shader interface type: {name!r}.")


class StorageClass(Enum):
    GLOBAL = "global"
    WORKGROUP = "workgroup"
    PRIVATE = "private"
    PUSH_CONSTANT = "push_constant"

    @classmethod
    def from_str(cls, value: str) -> StorageClass:
        for sc in cls:
            if sc.value == value:
                return sc
        raise ValueError(f"Invalid storage class: {value!r}")


def natural_alignment(type: Type | str) -> int:
    """Return natural alignment in bytes according to std430 rules."""
    if isinstance(type, Type):
        val = type.value
    elif isinstance(type, str):
        val = type
    else:
        raise TypeError(f"Expected Type or str, got {type!r}")

    if val in ("f32", "i32", "si32", "u32"):
        return 4
    if val in ("f16", "i16", "si16", "u16"):
        return 2
    if val in ("f64", "i64", "si64", "u64"):
        return 8
    if val in ("i8", "si8", "u8", "i1", "bool"):
        return 1
    matrix = re.fullmatch(r"mat(\d)x(\d)", val)
    if matrix:
        return 8 if matrix.group(2) == "2" else 16
    if val == "vector<2xf32>":
        return 8
    if val in ("vector<3xf32>", "vector<4xf32>"):
        return 16
    if val.startswith("vector<") and val.endswith(">"):
        inner = val[7:-1]
        parts = inner.split("x", 1)
        if len(parts) == 2:
            try:
                count = int(parts[0])
                elem_align = natural_alignment(parts[1])
                factor = 4 if count in (3, 4) else count
                return factor * elem_align
            except (ValueError, TypeError):
                pass
    raise ValueError(f"Unsupported type for natural alignment: {type!r}")


@dataclass(frozen=True)
class PointerType:
    element_type: Type | str
    storage_class: StorageClass
    alignment: int = field(default=0)
    _explicit: bool = field(default=False, compare=False, hash=False, repr=False)

    def __init__(
        self,
        element_type: Type | str,
        storage_class: StorageClass,
        alignment: int | None = None,
    ):
        if element_type is None:
            raise TypeError("element_type cannot be None")
        if not isinstance(element_type, (Type, str)):
            raise TypeError(f"element_type must be a Type or str, got {type(element_type).__name__}")
        if isinstance(element_type, str) and (
            element_type.startswith("!gfx.ptr") or element_type.startswith("!gfx.handle")
        ):
            raise ValueError(f"Invalid pointee type: {element_type}")

        if not isinstance(storage_class, StorageClass):
            raise TypeError(
                f"storage_class must be a StorageClass enum instance, got {type(storage_class).__name__}"
            )

        nat = natural_alignment(element_type)
        if alignment is None:
            align = nat
            explicit = False
        else:
            if not isinstance(alignment, int) or isinstance(alignment, bool):
                raise TypeError(f"alignment must be an integer, got {type(alignment).__name__}")
            if alignment <= 0 or (alignment & (alignment - 1)) != 0:
                raise ValueError(f"pointer alignment must be a power of two, got {alignment}")
            if nat > 0 and alignment < nat:
                raise ValueError(
                    f"pointer alignment ({alignment}) is smaller than the natural alignment of pointee type ({nat})"
                )
            align = alignment
            explicit = True

        object.__setattr__(self, "element_type", element_type)
        object.__setattr__(self, "storage_class", storage_class)
        object.__setattr__(self, "alignment", align)
        object.__setattr__(self, "_explicit", explicit)

    def __str__(self) -> str:
        elem_str = (
            self.element_type.value
            if isinstance(self.element_type, Type)
            else str(self.element_type)
        )
        if self._explicit:
            return f"!gfx.ptr<{elem_str}, {self.storage_class.value}, {self.alignment}>"
        return f"!gfx.ptr<{elem_str}, {self.storage_class.value}>"


class AccessMode(Enum):
    READ_ONLY = "read_only"
    WRITE_ONLY = "write_only"
    READ_WRITE = "read_write"

    @classmethod
    def from_str(cls, val: str) -> AccessMode:
        for m in cls:
            if m.value == val:
                return m
        raise ValueError(f"Invalid buffer access mode: {val!r}")


def _normalize_buffer_element_type(elem: Any) -> Type:
    if isinstance(elem, Type):
        return elem
    if elem is float or elem is Type.F32 or elem == "f32":
        return Type.F32
    if elem is int or elem is i32 or elem is Type.I32 or elem in ("i32", "si32"):
        return Type.I32
    if elem is u32 or elem is Type.U32 or elem == "u32":
        return Type.U32
    if elem is bool or elem is Type.BOOL or elem == "bool":
        return Type.BOOL
    if elem is vec2 or elem is Type.VEC2 or elem in ("vec2", "f32x2", "vector<2xf32>"):
        return Type.VEC2
    if elem is vec3 or elem is Type.VEC3 or elem in ("vec3", "f32x3", "vector<3xf32>"):
        return Type.VEC3
    if elem is vec4 or elem is Type.VEC4 or elem in ("vec4", "f32x4", "vector<4xf32>"):
        return Type.VEC4
    if isinstance(elem, str):
        try:
            return Type.from_schema(elem)
        except ValueError:
            pass
        for t in Type:
            if t.value == elem:
                return t
    raise TypeError(f"Invalid or unsupported buffer element type: {elem!r}")


@dataclass(frozen=True)
class BufferBinding:
    element_type: Type
    access: AccessMode = AccessMode.READ_WRITE
    storage_class: StorageClass = StorageClass.GLOBAL
    set: int = 0
    binding: int | None = None
    name: str = ""
    shape: tuple[int, ...] | None = None
    alignment: int | None = None

    def __post_init__(self):
        if not isinstance(self.element_type, Type):
            norm = _normalize_buffer_element_type(self.element_type)
            object.__setattr__(self, "element_type", norm)

        if isinstance(self.access, str):
            mode = AccessMode.from_str(self.access)
            object.__setattr__(self, "access", mode)
        elif not isinstance(self.access, AccessMode):
            raise TypeError(f"access mode must be AccessMode or str, got {type(self.access).__name__}")

        if not isinstance(self.storage_class, StorageClass):
            raise TypeError(f"storage_class must be StorageClass, got {type(self.storage_class).__name__}")

        if self.set < 0:
            raise ValueError(f"descriptor set index must be non-negative, got {self.set}")
        if self.binding is not None and self.binding < 0:
            raise ValueError(f"descriptor binding index must be non-negative, got {self.binding}")

    def __str__(self) -> str:
        elem_name = (
            self.element_type.value
            if isinstance(self.element_type, Type)
            else str(self.element_type)
        )
        return f"Buffer[{elem_name}, {self.access.value}]"


BufferType = BufferBinding


class Buffer:
    """Generic Buffer descriptor for compute shader kernel parameters.

    Usage:
        def kernel(a: Buffer[Type.F32, "read_only"],
                   b: Buffer[Type.F32, "write_only"]):
            ...
    """

    def __class_getitem__(cls, item: Any) -> BufferBinding:
        if isinstance(item, tuple):
            if len(item) == 2:
                elem, acc = item
                mode = AccessMode.from_str(acc) if isinstance(acc, str) else acc
                norm_elem = _normalize_buffer_element_type(elem)
                return BufferBinding(element_type=norm_elem, access=mode)
            elif len(item) == 3:
                elem, acc, extra = item
                mode = AccessMode.from_str(acc) if isinstance(acc, str) else acc
                norm_elem = _normalize_buffer_element_type(elem)
                if isinstance(extra, int):
                    return BufferBinding(element_type=norm_elem, access=mode, binding=extra)
                elif isinstance(extra, (tuple, list)):
                    return BufferBinding(element_type=norm_elem, access=mode, shape=tuple(extra))
                raise TypeError(f"Third parameter to Buffer must be int binding or tuple shape, got {type(extra).__name__}")
            raise TypeError(f"Buffer[...] expects 1 to 3 parameters, got {len(item)}")
        norm_elem = _normalize_buffer_element_type(item)
        return BufferBinding(element_type=norm_elem, access=AccessMode.READ_WRITE)


@dataclass(frozen=True)
class ReflectedBufferBinding:
    name: str
    set_index: int
    binding_index: int
    access: AccessMode
    pointer_type: PointerType


f32 = Type.F32
i32 = Type.I32
u32 = Type.U32


@dataclass(frozen=True)
class Node:
    op: str
    type: Type
    args: tuple[Node, ...] = ()
    literal: float | int | bool | str | None = None


class Value:
    """A traced device value, not an ordinary Python number."""

    __hash__ = None
    __slots__ = ("node",)

    def __init__(self, node: Node):
        self.node = node

    @property
    def type(self) -> Type:
        return self.node.type

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
        if getattr(other, "is_device_matrix", False):
            return NotImplemented  # defer to MatrixValue.__rmul__ (vector * matrix)
        return _binary("mul", self, other)

    def __rmul__(self, other: Value | float | int) -> Value:
        return _binary("mul", other, self)

    def __truediv__(self, other: Value | float | int) -> Value:
        return _binary("div", self, other)

    def __rtruediv__(self, other: Value | float | int) -> Value:
        return _binary("div", other, self)

    def __floordiv__(self, other: Value | int) -> Value:
        return _int_binary("floordiv", self, other)

    def __rfloordiv__(self, other: Value | int) -> Value:
        return _int_binary("floordiv", other, self)

    def __mod__(self, other: Value | int) -> Value:
        return _int_binary("mod", self, other)

    def __rmod__(self, other: Value | int) -> Value:
        return _int_binary("mod", other, self)

    def __lshift__(self, other: Value | int) -> Value:
        return _int_binary("shl", self, other)

    def __rlshift__(self, other: Value | int) -> Value:
        return _int_binary("shl", other, self)

    def __rshift__(self, other: Value | int) -> Value:
        return _int_binary("shr", self, other)

    def __rrshift__(self, other: Value | int) -> Value:
        return _int_binary("shr", other, self)

    def __neg__(self) -> Value:
        if self.node.type not in (Type.F32, Type.I32, *VECTOR_SIZES):
            raise TypeError("Negation requires a floating-point or signed value.")
        return Value(Node("neg", self.node.type, (self.node,)))

    def __abs__(self) -> Value:
        if self.node.type == Type.U32:
            return self
        if self.node.type not in (Type.F32, Type.I32, *VECTOR_SIZES):
            raise TypeError("Absolute value requires a numeric value.")
        return Value(Node("abs", self.node.type, (self.node,)))

    def __and__(self, other: Value | int | bool) -> Value:
        return _bitwise_or_boolean("and", self, other)

    def __rand__(self, other: Value | int | bool) -> Value:
        return _bitwise_or_boolean("and", other, self)

    def __or__(self, other: Value | int | bool) -> Value:
        return _bitwise_or_boolean("or", self, other)

    def __ror__(self, other: Value | int | bool) -> Value:
        return _bitwise_or_boolean("or", other, self)

    def __xor__(self, other: Value | int | bool) -> Value:
        return _bitwise_or_boolean("xor", self, other)

    def __rxor__(self, other: Value | int | bool) -> Value:
        return _bitwise_or_boolean("xor", other, self)

    def __invert__(self) -> Value:
        if self.node.type in (Type.I32, Type.U32):
            mask = -1 if self.node.type == Type.I32 else 0xFFFF_FFFF
            return _int_binary("xor", self, mask)
        return _boolean("xor", self, True)

    def __lt__(self, other: Value | float | int) -> Value:
        return _compare("lt", self, other)

    def __le__(self, other: Value | float | int) -> Value:
        return _compare("le", self, other)

    def __gt__(self, other: Value | float | int) -> Value:
        return _compare("gt", self, other)

    def __ge__(self, other: Value | float | int) -> Value:
        return _compare("ge", self, other)

    def __ne__(self, other: object) -> Value:
        if not isinstance(other, (Value, float, int)):
            raise TypeError("Device comparison requires a device value or number.")
        return _compare("ne", self, other)

    def __getattr__(self, name: str) -> Value:
        indices = _swizzle(self.node.type, name)
        components = tuple(self[index] for index in indices)
        return components[0] if len(indices) == 1 else _vector(len(indices), components)

    def with_components(self, **updates: Value | float) -> Value:
        size = VECTOR_SIZES.get(self.node.type)
        if size is None:
            raise TypeError("Component replacement requires a vector.")
        components = [self[index] for index in range(size)]
        replaced: set[int] = set()
        for name, value in updates.items():
            indices = _swizzle(self.node.type, name)
            if len(set(indices)) != len(indices) or replaced.intersection(indices):
                raise ValueError("Component replacements must not overlap.")
            values = _components((value,))
            if len(values) == 1:
                values *= len(indices)
            if len(values) != len(indices):
                raise TypeError("Replacement shape must match its swizzle.")
            for index, component in zip(indices, values):
                components[index] = Value(component)
            replaced.update(indices)
        return _vector(size, tuple(components))

    def __getitem__(self, index: int) -> Value:
        size = VECTOR_SIZES.get(self.node.type)
        if size is None or type(index) is not int or not 0 <= index < size:
            raise TypeError("Vector extraction requires an in-range constant integer component.")
        elem_type = Type.U32 if self.node.type == "vector<3xi32>" else Type.F32
        return Value(Node("extract", elem_type, (self.node,), literal=index))

    def __eq__(self, other: object) -> Value:
        if not isinstance(other, (Value, float, int)):
            raise TypeError("Device equality requires a device value or number.")
        lhs, rhs = _operands(self, other)
        if lhs.type not in (Type.U32, Type.I32, Type.F32, Type.BOOL):
            raise TypeError("Equality currently supports scalars only.")
        return Value(Node("eq", Type.BOOL, (lhs, rhs)))


def _literal(value: float | int | bool, expected: Type | None = None) -> Node:
    if isinstance(value, bool):
        node = Node("constant", Type.BOOL, literal=value)
    elif isinstance(value, int) and expected != Type.F32:
        signed = expected == Type.I32
        lower, upper = (-2**31, 2**31 - 1) if signed else (0, 0xFFFFFFFF)
        if not lower <= value <= upper:
            raise ValueError(f"{'Signed' if signed else 'Unsigned'} integer literals must fit in 32 bits.")
        node = Node("constant", Type.I32 if signed else Type.U32, literal=value)
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


VECTOR_SIZES = {Type.VEC2: 2, Type.VEC3: 3, Type.VEC4: 4, "vector<3xi32>": 3}


def global_invocation_id() -> Value:
    return Value(Node("builtin", "vector<3xi32>", (), literal="global_invocation_id"))


def local_invocation_id() -> Value:
    return Value(Node("builtin", "vector<3xi32>", (), literal="local_invocation_id"))


def workgroup_id() -> Value:
    return Value(Node("builtin", "vector<3xi32>", (), literal="workgroup_id"))


def shared_array(
    element_type: Any,
    shape_or_count: int | Sequence[int],
    *,
    name: str | None = None,
) -> Any:
    """Allocate a static workgroup-shared array inside a @compute kernel."""
    raise TypeError("shared_array() can only be called inside a @compute kernel.")


def workgroup_barrier() -> None:
    """Synchronize all threads in the workgroup and order workgroup shared memory accesses."""
    raise TypeError("workgroup_barrier() can only be called inside a @compute kernel.")


def storage_barrier() -> None:
    """Synchronize all threads in the workgroup and order global storage buffer accesses."""
    raise TypeError("storage_barrier() can only be called inside a @compute kernel.")


def full_barrier() -> None:
    """Synchronize all threads in the workgroup and order both workgroup and global memory accesses."""
    raise TypeError("full_barrier() can only be called inside a @compute kernel.")


def atomic_add(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically add value to target[index] and return the previous 32-bit integer value."""
    raise TypeError("atomic_add() can only be called inside a shader kernel.")


def atomic_sub(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically subtract value from target[index] and return the previous 32-bit integer value."""
    raise TypeError("atomic_sub() can only be called inside a shader kernel.")


def atomic_min(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically replace target[index] with min(target[index], value) and return the previous value."""
    raise TypeError("atomic_min() can only be called inside a shader kernel.")


def atomic_max(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically replace target[index] with max(target[index], value) and return the previous value."""
    raise TypeError("atomic_max() can only be called inside a shader kernel.")


def atomic_umin(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically replace target[index] with unsigned min(target[index], value) and return the previous value."""
    raise TypeError("atomic_umin() can only be called inside a shader kernel.")


def atomic_umax(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically replace target[index] with unsigned max(target[index], value) and return the previous value."""
    raise TypeError("atomic_umax() can only be called inside a shader kernel.")


def atomic_and(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically bitwise-AND value into target[index] and return the previous 32-bit integer value."""
    raise TypeError("atomic_and() can only be called inside a shader kernel.")


def atomic_or(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically bitwise-OR value into target[index] and return the previous 32-bit integer value."""
    raise TypeError("atomic_or() can only be called inside a shader kernel.")


def atomic_xor(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically bitwise-XOR value into target[index] and return the previous 32-bit integer value."""
    raise TypeError("atomic_xor() can only be called inside a shader kernel.")


def atomic_exchange(target: Any, index: Any, value: Value | int) -> Value:
    """Atomically store value into target[index] and return the previous 32-bit integer value."""
    raise TypeError("atomic_exchange() can only be called inside a shader kernel.")


def atomic_compare_exchange(
    target: Any, index: Any, expected: Value | int, desired: Value | int
) -> Value:
    """Atomically store desired into target[index] if equal to expected; return the previous value."""
    raise TypeError("atomic_compare_exchange() can only be called inside a shader kernel.")


atomic_cas = atomic_compare_exchange


def _operands(lhs: Value | float | int, rhs: Value | float | int,
              broadcast: bool = False) -> tuple[Node, Node]:
    expected = lhs.node.type if isinstance(lhs, Value) else (
        rhs.node.type if isinstance(rhs, Value) else None
    )
    if broadcast:
        expected = next((value.node.type for value in (lhs, rhs)
                         if isinstance(value, Value) and value.node.type in VECTOR_SIZES), expected)
    if broadcast and expected in VECTOR_SIZES:
        def operand(value):
            if isinstance(value, Value) and value.node.type in VECTOR_SIZES:
                return _node(value, expected)
            return Node("broadcast", expected, (_node(value, Type.F32),))
        return operand(lhs), operand(rhs)
    left = _node(lhs, expected)
    right = _node(rhs, left.type)
    return left, right


def _binary(op: str, lhs: Value | float | int, rhs: Value | float | int) -> Value:
    left, right = _operands(lhs, rhs, broadcast=True)
    if left.type not in (Type.F32, Type.U32, Type.I32, *VECTOR_SIZES):
        raise TypeError("Arithmetic requires matching numeric values.")
    if op == "div" and left.type not in (Type.F32, *VECTOR_SIZES):
        raise TypeError("Division currently requires floating-point values.")
    return Value(Node(op, left.type, (left, right)))


def _compare(predicate: str, lhs: Value | float | int, rhs: Value | float | int) -> Value:
    left, right = _operands(lhs, rhs)
    if left.type not in (Type.F32, Type.I32, Type.U32, Type.BOOL):
        raise TypeError("Comparisons currently require matching scalar values.")
    if left.type == Type.BOOL and predicate != "ne":
        raise TypeError("Ordering requires numeric values.")
    if left.type == Type.F32:
        predicate = "une" if predicate == "ne" else "o" + predicate
    elif predicate != "ne":
        predicate = ("s" if left.type == Type.I32 else "u") + predicate
    return Value(Node("cmp", Type.BOOL, (left, right), literal=predicate))


def _int_binary(op: str, lhs: Value | int, rhs: Value | int) -> Value:
    left, right = _operands(lhs, rhs)
    if left.type not in (Type.I32, Type.U32):
        raise TypeError("Integer operation requires matching i32 or u32 scalar values.")
    return Value(Node(op, left.type, (left, right)))


def _bitwise_or_boolean(op: str, lhs: Value | int | bool, rhs: Value | int | bool) -> Value:
    if (isinstance(lhs, Value) and lhs.node.type in (Type.I32, Type.U32)) or (
        isinstance(rhs, Value) and rhs.node.type in (Type.I32, Type.U32)
    ):
        return _int_binary(op, lhs, rhs)
    return _boolean(op, lhs, rhs)


def _boolean(op: str, lhs: Value | bool, rhs: Value | bool) -> Value:
    return Value(Node(op, Type.BOOL, (_node(lhs, Type.BOOL), _node(rhs, Type.BOOL))))


def _swizzle(typ: Type, name: str) -> tuple[int, ...]:
    size = VECTOR_SIZES.get(typ)
    alphabet = next((chars for chars in ("xyzw", "rgba")
                     if name and all(char in chars for char in name)), None)
    if size is None or alphabet is None or not 1 <= len(name) <= 4:
        raise AttributeError(f"Invalid vector swizzle: {name!r}.")
    indices = tuple(alphabet.index(char) for char in name)
    if any(index >= size for index in indices):
        raise AttributeError(f"Swizzle {name!r} exceeds the vector size.")
    return indices


def _components(values: tuple[Value | float | int, ...]) -> tuple[Node, ...]:
    result = []
    for value in values:
        if isinstance(value, Value) and value.node.type in VECTOR_SIZES:
            result.extend(value[index].node for index in range(VECTOR_SIZES[value.node.type]))
        else:
            result.append(_node(value, Type.F32))
    return tuple(result)


def _vector(size: int, values: tuple[Value | float | int, ...]) -> Value:
    components = _components(values)
    if len(components) == 1:
        return Value(Node("broadcast", next(typ for typ, count in VECTOR_SIZES.items() if count == size),
                          components))
    if len(components) != size:
        raise TypeError(f"vec{size} requires one scalar or exactly {size} components.")
    typ = next(typ for typ, count in VECTOR_SIZES.items() if count == size)
    return Value(Node(f"vec{size}", typ, components))


def vec2(x: Value | float, y: Value | float | None = None) -> Value:
    return _vector(2, (x,) if y is None else (x, y))


def vec3(x: Value | float, y: Value | float | None = None,
         z: Value | float | None = None) -> Value:
    if y is None and z is not None:
        raise TypeError("Vector components cannot have gaps.")
    return _vector(3, tuple(value for value in (x, y, z) if value is not None))


def vec4(x: Value | float, y: Value | float | None = None,
         z: Value | float | None = None, w: Value | float | None = None) -> Value:
    values = (x, y, z, w)
    count = next((index for index, value in enumerate(values) if value is None), 4)
    if any(value is not None for value in values[count:]):
        raise TypeError("Vector components cannot have gaps.")
    return _vector(4, tuple(value for value in values[:count] if value is not None))


def _integer(value: Value | int, typ: Type) -> Value:
    if isinstance(value, Value):
        if value.node.type not in (Type.I32, Type.U32):
            raise TypeError("Integer reinterpretation requires an i32/u32 scalar.")
        return value if value.node.type == typ else Value(Node("cast", typ, (value.node,)))
    return Value(_node(value, typ))


def i32(value: Value | int) -> Value:
    return _integer(value, Type.I32)


def u32(value: Value | int) -> Value:
    return _integer(value, Type.U32)


def minimum(lhs: Value | float | int, rhs: Value | float | int) -> Value:
    return _binary("min", lhs, rhs)


def maximum(lhs: Value | float | int, rhs: Value | float | int) -> Value:
    return _binary("max", lhs, rhs)


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
    topology: str = "triangle-list"
    vertex_count: int | None = 3
    indexed: bool = False
    index_count: int | None = None
    index_type: str | None = None
    first_index: int = 0
    vertex_offset: int = 0
    instance_count: int = 1
    first_vertex: int = 0
    first_instance: int = 0

    def __post_init__(self):
        if self.topology != "triangle-list":
            raise ValueError("The preview currently supports triangle-list topology only.")
        if type(self.instance_count) is not int or not 1 <= self.instance_count <= 0xFFFFFFFF:
            raise ValueError("instance_count must be a positive integer fitting in u32.")
        if type(self.first_instance) is not int or not 0 <= self.first_instance <= 0xFFFFFFFF:
            raise ValueError("first_instance must be a nonnegative integer fitting in u32.")
        if self.indexed:
            if type(self.index_count) is not int or not 0 <= self.index_count <= 0xFFFFFFFF or self.index_count % 3:
                raise ValueError("Indexed triangle-list index_count must be a nonnegative multiple of three fitting in u32.")
            if self.index_type not in ("uint16", "uint32", "u16", "u32"):
                raise ValueError("index_type must be 'uint16' or 'uint32'.")
            if type(self.first_index) is not int or not 0 <= self.first_index <= 0xFFFFFFFF:
                raise ValueError("first_index must be a nonnegative integer fitting in u32.")
            if type(self.vertex_offset) is not int or not -0x80000000 <= self.vertex_offset <= 0x7FFFFFFF:
                raise ValueError("vertex_offset must fit in i32.")
        else:
            if type(self.vertex_count) is not int or not 0 <= self.vertex_count <= 0xFFFFFFFF or self.vertex_count % 3:
                raise ValueError("Triangle-list vertex_count must be a nonnegative multiple of three fitting in u32.")
            if type(self.first_vertex) is not int or not 0 <= self.first_vertex <= 0xFFFFFFFF:
                raise ValueError("first_vertex must be a nonnegative integer fitting in u32.")

    def to_dict(self) -> dict:
        if self.indexed:
            normalized_type = "uint16" if self.index_type in ("uint16", "u16") else "uint32"
            return {
                "topology": self.topology,
                "indexed": True,
                "index_count": self.index_count,
                "index_type": normalized_type,
                "first_index": self.first_index,
                "vertex_offset": self.vertex_offset,
                "instance_count": self.instance_count,
                "first_instance": self.first_instance,
            }
        result = {"topology": self.topology, "vertex_count": self.vertex_count}
        if self.first_vertex != 0:
            result["first_vertex"] = self.first_vertex
        if self.instance_count != 1:
            result["instance_count"] = self.instance_count
        if self.first_instance != 0:
            result["first_instance"] = self.first_instance
        return result


@dataclass(frozen=True)
class DepthStencil:
    format: str = "d32_sfloat"
    depth_test_enable: bool = True
    depth_write_enable: bool = True
    depth_compare_op: str = "less_equal"
    clear_depth: float = 1.0

    def __post_init__(self):
        from .interfaces import verify_depth_stencil
        verify_depth_stencil(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "depth_test_enable": self.depth_test_enable,
            "depth_write_enable": self.depth_write_enable,
            "depth_compare_op": self.depth_compare_op,
            "clear_depth": self.clear_depth,
        }


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
    structured_input: dict | None = None
    input_layout: dict | None = None

    @property
    def structured(self) -> dict | None:
        return self.structured_input

    def to_dict(self) -> dict:
        if self.structured_input is not None:
            return self.structured_input
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
                elif operation in ("constant", "extract", "cmp"):
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


def resolve_annotation(fn: Callable, annotation: Any) -> Any:
    if not isinstance(annotation, str):
        return annotation
    eval_globals = {
        "Buffer": Buffer,
        "BufferBinding": BufferBinding,
        "BufferType": BufferType,
        "AccessMode": AccessMode,
        "Type": Type,
        "PointerType": PointerType,
        "StorageClass": StorageClass,
        "FragmentCoord": FragmentCoord,
        "GlobalInvocationId": GlobalInvocationId,
        "LocalInvocationId": LocalInvocationId,
        "WorkgroupId": WorkgroupId,
        "f32": f32,
        "i32": i32,
        "u32": u32,
    }
    if hasattr(fn, "__globals__"):
        eval_globals.update(fn.__globals__)
    eval_locals = {}
    if hasattr(fn, "__closure__") and fn.__closure__ and hasattr(fn, "__code__"):
        for var, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                eval_locals[var] = cell.cell_contents
            except (ValueError, AttributeError):
                pass
    try:
        return eval(annotation, eval_globals, eval_locals)
    except Exception:
        if hasattr(fn, "__globals__"):
            return fn.__globals__.get(annotation, annotation)
        return annotation


@dataclass(frozen=True)
class Entry:
    function: Callable
    kind: str
    workgroup_size: tuple[int, int, int] = (1, 1, 1)

    @property
    def reflected_buffer_bindings(self) -> list[ReflectedBufferBinding]:
        results = []
        binding_counter = 0
        for param in inspect.signature(self.function).parameters.values():
            ann = resolve_annotation(self.function, param.annotation)
            if isinstance(ann, BufferBinding):
                set_idx = ann.set
                binding_idx = ann.binding if ann.binding is not None else binding_counter
                binding_counter = max(binding_counter + 1, binding_idx + 1)
                mode = ann.access if isinstance(ann.access, AccessMode) else AccessMode.from_str(ann.access)
                elem_type = ann.element_type if isinstance(ann.element_type, Type) else Type.from_schema(str(ann.element_type))
                results.append(ReflectedBufferBinding(
                    name=param.name,
                    set_index=set_idx,
                    binding_index=binding_idx,
                    access=mode,
                    pointer_type=PointerType(elem_type, ann.storage_class, ann.alignment),
                ))
        return results

    def trace(self, varying_interface: tuple[ShaderIO, ...] | None = None) -> Stage:
        name = self.function.__name__
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid entry-point name: {name!r}.")
        parameters = list(inspect.signature(self.function).parameters.values())
        from .parameters import ParameterBlock, is_struct, _resolve_field_type
        parameter_blocks = {}
        vertex_layout_struct = None
        for parameter in parameters:
            annotation = resolve_annotation(self.function, parameter.annotation)
            if isinstance(annotation, ParameterBlock):
                parameter_blocks[parameter.name] = annotation
            elif is_struct(annotation):
                if getattr(annotation, "_is_vertex_struct", False) or getattr(annotation, "__struct_layout__", "") == "vertex":
                    if self.kind != "vertex":
                        raise TypeError("Vertex layout structs can only be used with @vertex entry points.")
                    if vertex_layout_struct is not None:
                        raise TypeError("Only one vertex struct input is supported per vertex shader.")
                    vertex_layout_struct = (parameter.name, annotation)
                else:
                    parameter_blocks[parameter.name] = annotation.as_parameter_block(name=parameter.name)

        if self.kind == "compute":
            interfaces = ()
            input_nodes = ()
            from .ast_frontend import ASTLowering
            lowering = ASTLowering(self.function, input_nodes, interfaces, parameter_blocks, workgroup_size=self.workgroup_size)
            structured_data = lowering.module(name, "compute", (), (), workgroup_size=self.workgroup_size)
            return Stage(
                name=name,
                kind="compute",
                inputs=(),
                outputs=(),
                input_interface=(),
                output_interface=(),
                source_file=inspect.getsourcefile(self.function) or "",
                source_line=inspect.getsourcelines(self.function)[1] if inspect.getsourcefile(self.function) else 0,
                structured_input=structured_data,
            )

        raster_parameters = [parameter for parameter in parameters if parameter.name not in parameter_blocks]
        if any(parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
               for parameter in parameters):
            raise TypeError("Entry points require positional device arguments.")
        for parameter in parameters:
            resolved_ann = resolve_annotation(self.function, parameter.annotation)
            if resolved_ann in (GlobalInvocationId, LocalInvocationId, WorkgroupId):
                raise TypeError("Compute invocation/workgroup ID parameters require a compute entry point.")
        coordinate_parameters = {
            parameter.name for parameter in parameters
            if resolve_annotation(self.function, parameter.annotation) is FragmentCoord or parameter.annotation == "FragmentCoord"
        }
        input_layout = None
        if self.kind == "vertex":
            if coordinate_parameters:
                raise TypeError("FragmentCoord parameters require a fragment entry point.")
            if vertex_layout_struct is not None:
                param_name, struct_cls = vertex_layout_struct
                interfaces = tuple(
                    ShaderIO(f.name, f.type if isinstance(f.type, Type) else Type.VEC4, location=i, interpolation="perspective")
                    for i, f in enumerate(struct_cls.__struct_fields__)
                )
                input_layout = struct_cls.as_input_layout()
            elif len(raster_parameters) == 1 and (
                raster_parameters[0].annotation is inspect.Parameter.empty
                or raster_parameters[0].name in ("index", "vertex_index")
                or resolve_annotation(self.function, raster_parameters[0].annotation) in (Type.U32, Type.I32, u32, i32, "u32", "i32", int)
            ):
                interfaces = (ShaderIO("vertex_index", Type.U32, builtin="vertex_index"),)
            elif raster_parameters:
                interfaces = tuple(
                    ShaderIO(p.name, _resolve_field_type(resolve_annotation(self.function, p.annotation)), location=i, interpolation="perspective")
                    for i, p in enumerate(raster_parameters)
                )
            else:
                raise TypeError("A vertex entry point requires vertex inputs or a vertex-index argument.")
        elif varying_interface is None:
            if sum(parameter.name not in coordinate_parameters for parameter in raster_parameters) > 1:
                raise TypeError("Trace multi-input fragment shaders through Pipeline to resolve their interfaces.")
            interfaces = tuple(
                ShaderIO(parameter.name, Type.VEC2, builtin="fragment_coord")
                if parameter.name in coordinate_parameters else
                ShaderIO(parameter.name, Type.VEC4, location=0)
                for parameter in raster_parameters
            )
        else:
            available = {interface.name: interface for interface in varying_interface}
            resolved = []
            for parameter in raster_parameters:
                if parameter.name in coordinate_parameters:
                    resolved.append(ShaderIO(parameter.name, Type.VEC2, builtin="fragment_coord"))
                    continue
                interface = available.get(parameter.name)
                if interface is None and len(raster_parameters) == 1 and len(available) == 1:
                    interface = next(iter(available.values()))
                if interface is None:
                    raise ValueError(f"Fragment input {parameter.name!r} has no matching vertex varying.")
                resolved.append(interface)
            interfaces = tuple(resolved)
        if len(coordinate_parameters) > 1:
            raise TypeError("A fragment entry point may declare FragmentCoord only once.")
        input_nodes = tuple(Node("input", interface.type, literal=interface.name) for interface in interfaces)
        from .ast_frontend import ASTLowering
        lowering = ASTLowering(self.function, input_nodes, interfaces, parameter_blocks)
        result = lowering.result
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
            if node.op not in {
                "input", "constant", "vec2", "vec3", "vec4", "extract", "select",
                "eq", "cmp", "add", "sub", "mul", "div", "neg", "abs", "min", "max",
                "and", "or", "xor", "cast", "broadcast",
                "sin", "cos", "sqrt", "rsqrt", "exp", "exp2", "log", "log2", "floor", "pow",
                "ssa",
            }:
                raise ValueError(f"Unsupported device operation: {node.op}.")
            if node.op == "ssa" and id(node) not in lowering.owned:
                raise ValueError("A stage cannot capture another stage's device SSA value.")
            for arg in node.args:
                validate(arg)

        for output in outputs:
            validate(output)
        structured = lowering.module(name, self.kind, outputs, output_interface) if lowering.structured else None
        return Stage(name, self.kind, input_nodes, outputs, interfaces, output_interface,
                     self.function.__code__.co_filename, self.function.__code__.co_firstlineno, structured, input_layout=input_layout)


def vertex(function: Callable) -> Entry:
    return Entry(function, "vertex")


class FragmentCoord:
    """Annotate a fragment parameter to read top-left-origin pixel-center XY."""


class GlobalInvocationId:
    """Annotate a compute parameter to read the 3D global invocation coordinate."""


class LocalInvocationId:
    """Annotate a compute parameter to read the 3D workgroup-local invocation coordinate."""


class WorkgroupId:
    """Annotate a compute parameter to read the 3D workgroup coordinate."""


def fragment(function: Callable) -> Entry:
    return Entry(function, "fragment")


def compute(
    function_or_workgroup_size: Any = None,
    *,
    workgroup_size: tuple[int, int, int] = (1, 1, 1),
) -> Any:
    """Decorator declaring a compute shader entry point with workgroup dimensions."""
    if callable(function_or_workgroup_size):
        return Entry(function_or_workgroup_size, "compute", workgroup_size=workgroup_size)

    size = (
        function_or_workgroup_size
        if function_or_workgroup_size is not None
        else workgroup_size
    )
    if isinstance(size, int) and not isinstance(size, bool):
        size = (size, 1, 1)
    elif isinstance(size, (tuple, list)) and len(size) in (1, 2, 3):
        if any(type(d) is not int or isinstance(d, bool) for d in size):
            raise TypeError("workgroup_size dimensions must be integers.")
        size = (*size, *(1 for _ in range(3 - len(size))))
    else:
        raise TypeError("workgroup_size must be a 1D, 2D, or 3D integer tuple.")
    if any(d <= 0 for d in size):
        raise ValueError("workgroup_size dimensions must be positive integers.")

    def decorator(fn: Callable) -> Entry:
        return Entry(fn, "compute", workgroup_size=tuple(size))

    return decorator


@dataclass(frozen=True)
class Dispatch:
    grid: tuple[int, int, int] = (1, 1, 1)

    def __init__(self, grid: int | Sequence[int] = (1, 1, 1)):
        if isinstance(grid, int) and not isinstance(grid, bool):
            norm = (grid, 1, 1)
        elif isinstance(grid, (tuple, list)) and len(grid) in (1, 2, 3):
            if any(type(d) is not int or isinstance(d, bool) for d in grid):
                raise TypeError("Dispatch grid dimensions must be integers.")
            norm = (*grid, *(1 for _ in range(3 - len(grid))))
        else:
            raise TypeError("Dispatch grid must be a 1D, 2D, or 3D integer tuple.")
        if any(d <= 0 or d > 65535 for d in norm):
            raise ValueError(f"Dispatch grid dimensions must be in [1, 65535], got {norm}.")
        object.__setattr__(self, "grid", tuple(norm))

    def to_dict(self) -> dict[str, list[int]]:
        return {"grid": list(self.grid)}


@dataclass(frozen=True)
class ComputePipeline:
    compute: Entry
    dispatch: Dispatch = field(default_factory=Dispatch)

    def __post_init__(self) -> None:
        if not isinstance(self.compute, Entry) or self.compute.kind != "compute":
            raise TypeError("ComputePipeline requires a @compute entry point.")
        if not isinstance(self.dispatch, Dispatch):
            raise TypeError("ComputePipeline dispatch configuration must be a Dispatch object.")

    def trace(self) -> tuple[Stage]:
        return (self.compute.trace(),)


@dataclass(frozen=True)
class Pipeline:
    vertex: Entry
    fragment: Entry
    draw: Draw = field(default_factory=Draw)
    depth_stencil: DepthStencil | None = None
    input_layout: object = None

    def __post_init__(self):
        if self.depth_stencil is not None:
            if isinstance(self.depth_stencil, dict):
                object.__setattr__(self, "depth_stencil", DepthStencil(**self.depth_stencil))
            elif not isinstance(self.depth_stencil, DepthStencil):
                raise TypeError("depth_stencil must be a DepthStencil instance or dict.")

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
