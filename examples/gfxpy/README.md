# gfxpy (NumPy-Free Cross-Backend GPU Array Example)

`gfxpy` is a NumPy-free, CuPy-inspired example array library built on GfxIR.
The same Python source executes on **Linux Vulkan**, **Windows DirectX 12**,
and **macOS Metal** using persistent device-resident GPU buffers without
generating GLSL, HLSL, or MSL source.

## Quick Example

```python
import gfxpy as gp

dev = gp.device("auto")  # macOS -> metal, Windows -> directx, Linux -> vulkan
try:
    with dev:
        a = gp.array([1.0, 2.0, 3.0], dtype=gp.float32)
        b = gp.array([4.0, 5.0, 6.0], dtype=gp.float32)
        c = gp.array([0.5, 1.0, 1.5], dtype=gp.float32)

        # Intermediate results stay device-resident; no host readback per operator.
        y = (a * b + 1.5) * 2.0 + c
        result = y.tolist()
finally:
    dev.close()
```

A complete runnable script comparing GPU output against a standard-library
Python reference is provided in [chain_example.py](chain_example.py).

## Supported Platforms & Target Selection

| Platform (`sys.platform`) | Default Target | Explicit Override |
| --- | --- | --- |
| macOS (`darwin`) | `"metal"` | `gp.device("metal")` |
| Windows (`win32`) | `"directx"` | `gp.device("directx")` |
| Linux (`linux*`) | `"vulkan"` | `gp.device("vulkan")` |

Pass `target="auto"` (or `None`) for platform-default selection, or pass
`"vulkan"`, `"directx"`, or `"metal"` explicitly. Unsupported platforms or
unknown target names raise `UnsupportedTargetError`.

## Setup & Backend Prerequisites

- **Python**: `>= 3.11` with only the standard library (NumPy is **not** required).
  Install `mlir_gpu` in editable mode (`pip install -e .`) or run with
  `PYTHONPATH=src:examples`.
- **GfxIR Compiler Toolchain**: `gfx-compile` and `gfx-opt` (built via
  `tools/build_toolchain.py` into `.toolchains/llvm-build/bin`, or set `LLVM_BIN`).
- **Linux (`vulkan`)**:
  - Vulkan loader (`libvulkan1`) and a hardware Vulkan ICD or SwiftShader ICD
    (`VK_ICD_FILENAMES` / `VK_DRIVER_FILES` for software execution).
  - `spirv-val` (optional validator in `PATH` or `GPU_TOOLS_BIN`).
- **Windows (`directx`)**:
  - Windows 10/11 with DirectX 12 hardware adapter or WARP software adapter
    (`Microsoft Basic Render Driver`).
  - Prebuilt `mlir-gpu-directx-compute.dll` in `.tools/bin` / `GPU_TOOLS_BIN`,
    or MSVC `cl.exe` / `clang++` to build `native/directx_compute_bridge.cpp`
    on demand, plus `dxv` / `dxcompiler.dll` / `dxil.dll`.
- **macOS (`metal`)**:
  - Metal device supporting Argument Buffers Tier 2.
  - Apple Metal Shader Converter (`metal-shaderconverter` and
    `libmetalirconverter`), plus `clang++` or prebuilt
    `libmlir_gpu_metal_compute.dylib` for `native/metal_compute_bridge.mm`.

## API Scope, Limits, and Error Semantics

- **Supported types & shapes**: Contiguous 1D and 2D `float32` arrays
  (`gp.array(seq, dtype=gp.float32)`, `gp.empty(shape_or_len, dtype=gp.float32)`).
  Ragged or >2D sequences raise `ShapeMismatchError`; other dtypes raise
  `UnsupportedDTypeError`.
- **Supported operations**:
  - Elementwise `+` and `*` (array/array and array/scalar, including reflected
    scalar ops `2.0 * a` and `10.0 + b`).
  - Bounded sum reduction (`gp.sum(a)` / `a.sum()`) over all elements of a 1D or
    2D array, returning a device-resident `(1,)` `GfxArray`.
  - 2D matrix multiplication (`gp.matmul(a, b)` / `a @ b`) for `(M, K) @ (K, N) -> (M, N)`.
    Non-2D or mismatched inner dimensions raise `ShapeMismatchError`.
  - Explicit host readback via `tolist()`, `tobytes()`, `toarray()`,
    `tomemoryview()`, and `download_into()`.
- **Explicit elementwise fusion (`@gp.fuse` / `gp.fuse(fn)`)**:
  Ordinary `GfxArray` operators remain eager. Wrapping a function in `gp.fuse`
  lowers a contiguous `float32` `+`/`*` expression tree into a single
  bounds-checked GfxIR `@compute` dispatch with one output allocation and no
  materialized intermediates or host readback.
  - **Evaluation & caching**: Evaluated eagerly when the fused callable is
    invoked. Compiled kernels are cached by expression structure, dtype/layout,
    target, validation/provider, contraction policy, and toolchain; runtime
    scalars are passed via `ParameterBlock` parameters without recompilation.
  - **Restrictions & errors**: All array inputs must be open, `float32`,
    identical in shape (`ShapeMismatchError`), and on the same `Device`
    (`CrossDeviceError`). Unsupported operations inside `gp.fuse` (such as
    `-`, `/`, `@`, `.sum()`, indexing, or Python control flow over proxies)
    raise `UnsupportedFusionError`. Empty arrays return an empty `GfxArray`
    with zero allocations or dispatches.
  - **Contraction & reassociation policy**: `no_fast_math` — the generated
    GfxIR kernel preserves the exact parenthesized Python expression order
    without fast-math or algebraic reassociation (target shader compilers may
    still perform hardware FMA contraction unless disabled by the driver).
- **Empty arrays & `sum` / `matmul` shape semantics**:
  - Empty 1D `(0,)` and 2D `(0, N)` / `(M, 0)` arrays avoid zero-byte GPU
    allocations and zero-grid dispatches.
  - `gp.sum(x)` always returns a `(1,)` `GfxArray`; for any empty `x`, it
    returns `[0.0]` without dispatching a compute kernel.
  - `gp.matmul(a, b)` with `M == 0` or `N == 0` returns an empty `(M, N)`
    `GfxArray` without allocation/dispatch; `(M, 0) @ (0, N)` returns a
    zero-filled `(M, N)` `GfxArray`.

## Numerical Accuracy & Indexing Limits Contract

- **Scalar/array conversion & overflow**:
  - Python `int` and `float` inputs are converted to IEEE 754 `float32`
    (`FLOAT32_MAX ≈ 3.4028235e+38`, `FLOAT32_EPSILON = 2^-23 ≈ 1.1920929e-07`).
  - Finite host array elements or scalar operands outside `[-FLOAT32_MAX, FLOAT32_MAX]`
    raise `OverflowError` before GPU buffer allocation or dispatch.
  - Sub-minimum magnitudes (e.g. `1e-50`) underflow toward `0.0` according to
    IEEE 754 `float32` rounding.
  - Arithmetic overflow during GPU execution (e.g. `3e38 + 3e38` or
    `1e20 * 1e20`) produces `±inf`.
- **Float32 accumulation & reduction ordering**:
  - All elementwise, fused, `sum`, and `matmul` kernels accumulate in `float32`
    (not `float64`). Target shader compilers may contract `a * b + c` into
    hardware FMA unless disabled by the driver.
  - `gp.sum(x)` uses a multi-pass 64-lane parallel tree reduction
    (`reduce_sum_f32_kernel`, padding out-of-bounds lanes with `0.0`), and
    `gp.matmul` accumulates in 16-element `K` tiles. Because `float32` addition
    is non-associative, results are not bitwise identical to sequential CPU
    `sum()` or NumPy pairwise reductions; error bounds must account for
    `float32` rounding and conditioning (cancellation / mixed magnitudes).
- **Non-finite behavior (`inf`, `-inf`, `nan`)**:
  - `inf`, `-inf`, and `nan` are accepted in `gp.array()`, scalar ops,
    `gp.fuse()`, `gp.sum()`, and `gp.matmul()`, and propagate according to
    IEEE 754 `float32` rules (`inf + (-inf) -> nan`, `0.0 * inf -> nan`, `nan`
    propagates; out-of-bounds reduction/tile padding uses `0.0` so it never
    injects `nan` or `inf`).
- **Checked indexing, offset & backend dispatch limits (`IndexingLimitError`)**:
  - Kernels index buffers and parameters using signed 32-bit integers
    (`I32_MAX = 2_147_483_647`). Before allocation, compilation, or dispatch,
    `gfxpy` validates:
    - Each dimension `d <= I32_MAX` and total element count `size <= I32_MAX`.
    - Rounded-up 1D invocation index `grid_x * 64 - 1 <= I32_MAX`.
    - MatMul dimensions `m, n, k <= I32_MAX`, operand/output sizes
      `m * k, k * n, m * n <= I32_MAX`, max linear offsets `< I32_MAX`, and
      rounded-up tile coordinates `grid_x * 16 - 1 <= I32_MAX`,
      `grid_y * 16 - 1 <= I32_MAX`.
    - Backend `ComputeLimits` (`max_workgroup_count` per axis, e.g. `65535` on
      DirectX 12 and typical Vulkan devices, and `max_buffer_bytes` when
      reported).
  - Violations raise `IndexingLimitError` (subclass of `GfxPyError` and
    `ValueError`) before any GPU buffer allocation or native dispatch.
- **Explicit ownership & lifecycle**: `Device` and `GfxArray` support `close()`
  and `with ...:` context managers (`Device.__exit__` restores the active
  context; call `dev.close()` to release GPU resources). Use-after-close raises
  `DeviceClosedError` or `ArrayClosedError`; combining arrays from different
  devices raises `CrossDeviceError`.
- **Implicit default-device ownership, shutdown, and recreation**:
  Calling `gp.array(...)` or `gp.empty(...)` without `device=` and outside a
  `with gp.device(...)` block lazily creates and reuses a process-wide default
  `Device` for the platform-default target (`"auto"`), so implicit arrays share
  the same compute context and compiled pipeline cache. Nested `with gp.device(...)`
  scopes override `current_device()` and restore the previous scope on exit, while
  explicit `Device` instances remain isolated (`CrossDeviceError`). Call
  `gp.close_default_device()` (also registered with `atexit`) or
  `gp.current_device().close()` to deterministically shut down the default
  context. Closing a `Device` never silently reopens it (using that `Device`,
  its arrays, or an active `with dev:` scope after `dev.close()` raises
  `DeviceClosedError`); subsequent implicit calls outside a closed scope create
  a fresh default `Device`.
- **No CPU or cross-backend fallback**: Missing tools, missing adapters, or
  backend initialization failures raise `BackendInitializationError`,
  `PersistentComputeError`, or `BackendNotImplementedError` with the target and
  cause. `gfxpy` never silently falls back to CPU execution.
- **Optional NumPy & out of scope**: NumPy is **not** required by `gfxpy` and
  is only an optional benchmark/reference dependency if used externally.
  `float64`, arbitrary NumPy broadcasting, slicing/strides, full IEEE
  reproducibility across differently ordered reductions, and SciPy coverage are
  out of scope.

## Running the Example & Checks

### 1. Host-only contract & caching checks (no GPU required)
```bash
python -m unittest discover -s tests -p "test_gfxpy_*.py" -v
```

### 2. Runnable chained example
```bash
# Platform default target
python examples/gfxpy/chain_example.py --target auto

# Software providers (CI-safe)
python examples/gfxpy/chain_example.py --target directx --provider warp --validation
python examples/gfxpy/chain_example.py --target vulkan --provider swiftshader --validation

# Hardware providers
python examples/gfxpy/chain_example.py --target directx --provider hardware --validation
python examples/gfxpy/chain_example.py --target vulkan --provider hardware --validation
python examples/gfxpy/chain_example.py --target metal --validation
```

If tools or a compatible GPU device are unavailable, `chain_example.py` prints
`SKIPPED/UNAVAILABLE [...]` and exits with status `2` instead of falling back
to CPU.

### 3. Opt-in live execution tests (separates software vs hardware, skips when unavailable)
```bash
# Windows DirectX 12 (WARP software)
MLIR_GPU_TEST_DIRECTX=1 MLIR_GPU_TEST_WARP=1 MLIR_GPU_TEST_VALIDATION=1 \
  python -m unittest discover -s tests -p "test_gfxpy_execution.py" -v

# Windows DirectX 12 (Hardware)
MLIR_GPU_TEST_DIRECTX=1 MLIR_GPU_TEST_VALIDATION=1 \
  python -m unittest discover -s tests -p "test_gfxpy_execution.py" -v

# Linux Vulkan (SwiftShader software)
MLIR_GPU_TEST_VULKAN=1 MLIR_GPU_TEST_SWIFTSHADER=1 MLIR_GPU_TEST_VALIDATION=1 \
  python -m unittest discover -s tests -p "test_gfxpy_execution.py" -v

# macOS Metal
MLIR_GPU_TEST_METAL=1 MLIR_GPU_TEST_VALIDATION=1 \
  python -m unittest discover -s tests -p "test_gfxpy_execution.py" -v
```
