"""Explicit elementwise expression fusion for gfxpy using GfxIR @compute kernels.

Fusion policy:
- Explicit boundary: only calls wrapped with ``gp.fuse(fn)`` / ``@gp.fuse`` fuse;
  ordinary ``GfxArray`` operators remain eager.
- Supported subset: contiguous 1D or 2D ``float32`` arrays with ``+`` and ``*``
  (array/array and array/scalar, including reflected scalar ops).
- Contraction and reassociation policy: ``no_fast_math`` — the generated GfxIR
  kernel preserves the exact parenthesized expression tree from Python without
  fast-math or algebraic reassociation. Target shader drivers may still apply
  hardware FMA contraction unless disabled by the target compiler.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import functools
import hashlib
from pathlib import Path
from typing import Any

from .array import GfxArray, _allocate_output, _is_scalar_number
from .dtypes import float32
from .targets import (
    BackendNotImplementedError,
    CrossDeviceError,
    ShapeMismatchError,
    UnsupportedDTypeError,
    UnsupportedFusionError,
)

CONTRACTION_POLICY = "no_fast_math"


@dataclass(frozen=True)
class _ArrayInput:
    index: int


@dataclass(frozen=True)
class _ScalarInput:
    index: int


@dataclass(frozen=True)
class _BinaryExpr:
    op: str
    left: _ExprNode
    right: _ExprNode


_ExprNode = _ArrayInput | _ScalarInput | _BinaryExpr


@dataclass(frozen=True)
class FusedExpressionSpec:
    """Canonical structural description of a fused float32 elementwise kernel."""

    signature: str
    num_inputs: int
    num_scalars: int
    expr_code: str
    dtype: str = "float32"
    layout: str = "contiguous"
    contraction_policy: str = CONTRACTION_POLICY


class _TraceContext:
    def __init__(self) -> None:
        self.scalar_values: list[float] = []

    def add_scalar(self, value: float) -> _ScalarInput:
        idx = len(self.scalar_values)
        self.scalar_values.append(float(value))
        return _ScalarInput(idx)


class _ExprProxy:
    """Symbolic array proxy traced inside gp.fuse()."""

    def __init__(self, trace: _TraceContext, node: _ExprNode) -> None:
        self._trace = trace
        self._node = node

    def _combine(self, op: str, other: object, *, reverse: bool = False) -> _ExprProxy:
        if isinstance(other, _ExprProxy):
            if other._trace is not self._trace:
                raise UnsupportedFusionError(
                    "Cannot mix traced proxies from different gp.fuse() calls."
                )
            other_node: _ExprNode = other._node
        elif _is_scalar_number(other):
            other_node = self._trace.add_scalar(float(other))
        else:
            raise UnsupportedFusionError(
                f"Unsupported operand type {type(other).__name__!r} for '{op}' inside gp.fuse()."
            )
        left, right = (other_node, self._node) if reverse else (self._node, other_node)
        return _ExprProxy(self._trace, _BinaryExpr(op, left, right))

    def __add__(self, other: object) -> _ExprProxy:
        return self._combine("add", other, reverse=False)

    def __radd__(self, other: object) -> _ExprProxy:
        return self._combine("add", other, reverse=True)

    def __mul__(self, other: object) -> _ExprProxy:
        return self._combine("mul", other, reverse=False)

    def __rmul__(self, other: object) -> _ExprProxy:
        return self._combine("mul", other, reverse=True)

    def _unsupported(self, op_name: str) -> Any:
        raise UnsupportedFusionError(
            f"Operation '{op_name}' is not supported inside gp.fuse(); "
            "only float32 '+' and '*' are supported."
        )

    def __sub__(self, other: object) -> Any:
        return self._unsupported("-")

    def __rsub__(self, other: object) -> Any:
        return self._unsupported("-")

    def __truediv__(self, other: object) -> Any:
        return self._unsupported("/")

    def __rtruediv__(self, other: object) -> Any:
        return self._unsupported("/")

    def __floordiv__(self, other: object) -> Any:
        return self._unsupported("//")

    def __pow__(self, other: object) -> Any:
        return self._unsupported("**")

    def __matmul__(self, other: object) -> Any:
        return self._unsupported("@")

    def __neg__(self) -> Any:
        return self._unsupported("unary -")

    def __pos__(self) -> Any:
        return self._unsupported("unary +")

    def __bool__(self) -> bool:
        raise UnsupportedFusionError("Truth-value testing is not supported inside gp.fuse().")

    def __len__(self) -> int:
        raise UnsupportedFusionError("len() is not supported inside gp.fuse().")

    def __getitem__(self, key: object) -> Any:
        return self._unsupported("indexing")

    def sum(self) -> Any:
        return self._unsupported("sum()")

    def tolist(self) -> Any:
        return self._unsupported("tolist()")


def _format_signature(node: _ExprNode) -> str:
    if isinstance(node, _ArrayInput):
        return f"in{node.index}"
    if isinstance(node, _ScalarInput):
        return f"s{node.index}"
    op_sym = "+" if node.op == "add" else "*"
    return f"({_format_signature(node.left)}{op_sym}{_format_signature(node.right)})"


def _format_expr_code(node: _ExprNode) -> str:
    if isinstance(node, _ArrayInput):
        return f"in{node.index}[idx]"
    if isinstance(node, _ScalarInput):
        return f"params.s{node.index}"
    op_sym = "+" if node.op == "add" else "*"
    return f"({_format_expr_code(node.left)} {op_sym} {_format_expr_code(node.right)})"


def build_fused_compute_kernel(spec: FusedExpressionSpec, out_dir: Path) -> Any:
    """Generate and load a GfxIR @compute kernel for ``spec``."""
    from mlir_gpu import Buffer, ParameterBlock, Type, compute, global_invocation_id, i32

    from .kernels import WORKGROUP_SIZE_1D

    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(spec.signature.encode("utf-8")).hexdigest()[:16]
    kernel_path = out_dir / f"fused_{digest}.py"

    param_fields = [("n", Type.I32)] + [(f"s{i}", Type.F32) for i in range(spec.num_scalars)]
    params_block = ParameterBlock(f"gfxpy_fused_params_{digest}", tuple(param_fields))

    input_params = ",\n    ".join(
        f'in{i}: Buffer[Type.F32, "read_only"]' for i in range(spec.num_inputs)
    )
    source = (
        f"@compute(workgroup_size={WORKGROUP_SIZE_1D!r})\n"
        f"def fused_elementwise_f32_kernel(\n"
        f"    {input_params},\n"
        f'    out: Buffer[Type.F32, "write_only"],\n'
        f"    params: FusedParams,\n"
        f"):\n"
        f"    idx = i32(global_invocation_id().x)\n"
        f"    if idx < params.n:\n"
        f"        out[idx] = {spec.expr_code}\n"
    )
    kernel_path.write_text(source, encoding="utf-8")
    ns: dict[str, Any] = {
        "Buffer": Buffer,
        "FusedParams": params_block,
        "Type": Type,
        "compute": compute,
        "global_invocation_id": global_invocation_id,
        "i32": i32,
    }
    exec(compile(source, str(kernel_path), "exec"), ns)
    return ns["fused_elementwise_f32_kernel"]


def fuse(fn: Callable[..., Any]) -> Callable[..., GfxArray]:
    """Wrap a float32 elementwise function into a single fused GPU dispatch."""
    if not callable(fn):
        raise TypeError(f"fuse() expects a callable, got {type(fn).__name__}.")

    @functools.wraps(fn)
    def fused_wrapper(*args: Any) -> GfxArray:
        if not args:
            raise TypeError("fuse() wrapper requires at least one GfxArray argument.")

        trace = _TraceContext()
        input_arrays: list[GfxArray] = []
        array_index_by_id: dict[int, int] = {}
        traced_args: list[Any] = []

        first_array: GfxArray | None = None
        for arg in args:
            if isinstance(arg, GfxArray):
                arg._ensure_open()
                if arg.dtype is not float32:
                    raise UnsupportedDTypeError(
                        f"fuse() only supports float32 arrays, got {arg.dtype.name!r}."
                    )
                if first_array is None:
                    first_array = arg
                else:
                    if arg.device is not first_array.device:
                        raise CrossDeviceError(
                            f"Cannot fuse arrays on different devices: "
                            f"{first_array.device!r} vs {arg.device!r}."
                        )
                    if arg.shape != first_array.shape:
                        raise ShapeMismatchError(
                            f"Incompatible shapes in gp.fuse(): {first_array.shape} vs {arg.shape}."
                        )
                arr_id = id(arg)
                idx = array_index_by_id.get(arr_id)
                if idx is None:
                    idx = len(input_arrays)
                    array_index_by_id[arr_id] = idx
                    input_arrays.append(arg)
                traced_args.append(_ExprProxy(trace, _ArrayInput(idx)))
            elif _is_scalar_number(arg):
                traced_args.append(_ExprProxy(trace, trace.add_scalar(float(arg))))
            else:
                raise UnsupportedFusionError(
                    f"fuse() arguments must be GfxArray or float/int scalars, got {type(arg).__name__!r}."
                )

        if first_array is None:
            raise TypeError("fuse() requires at least one GfxArray argument.")

        result = fn(*traced_args)
        if not isinstance(result, _ExprProxy) or result._trace is not trace:
            raise UnsupportedFusionError(
                "gp.fuse() callable must return a single elementwise expression derived from its inputs."
            )

        if first_array.size == 0:
            return GfxArray(
                shape=first_array.shape,
                dtype=first_array.dtype,
                device=first_array.device,
                buffer=None,
            )

        spec = FusedExpressionSpec(
            signature=_format_signature(result._node),
            num_inputs=len(input_arrays),
            num_scalars=len(trace.scalar_values),
            expr_code=_format_expr_code(result._node),
        )
        backend = first_array.device.backend
        if not hasattr(backend, "fused_op"):
            raise BackendNotImplementedError(first_array.device.target, "fused_op")

        out_buf = _allocate_output(backend, first_array.nbytes)
        try:
            in_bufs = [arr._buffer for arr in input_arrays]
            backend.fused_op(spec, in_bufs, trace.scalar_values, out_buf, first_array.size)
        except Exception:
            out_buf.close()
            raise
        return GfxArray(
            shape=first_array.shape,
            dtype=first_array.dtype,
            device=first_array.device,
            buffer=out_buf,
        )

    return fused_wrapper
