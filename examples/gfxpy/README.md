# gfxpy (Example GPU Array API on GfxIR)

`gfxpy` is a NumPy-free, CuPy-inspired example array library built on GfxIR's `@compute` compiler and persistent GPU runtime (`open_compute_context`).

## Supported Operations
- **1D and 2D contiguous `float32` arrays**: `array(seq, dtype=float32)` and `empty(shape, dtype=float32)` supporting 1D `(n,)` and 2D `(rows, cols)` shapes
- **Device-resident elementwise arithmetic**: `+` and `*` (array/array and array/scalar) executed via bounds-checked GfxIR `@compute` kernels without intermediate host transfers
- **Bounded `sum` reduction**: `a.sum()` and `gfxpy.sum(a)` reduce all elements of a 1D or 2D array to a 1-element `GfxArray` of shape `(1,)` using a multi-dispatch shared-memory tree reduction (`workgroup_size=(64, 1, 1)`) where intermediate buffers stay on the device and are synchronized between passes. Empty inputs (`size == 0`) return a `(1,)` array with `[0.0]` without launching kernels.
- **2D `float32` matrix multiplication**: `a @ b` and `gfxpy.matmul(a, b)` compute `(M, K) @ (K, N) -> (M, N)` using a 16x16 tiled shared-memory `@compute` kernel (`workgroup_size=(16, 16, 1)`). Handles non-tile-aligned dimensions with bounds checks; if `K == 0` with `M > 0, N > 0`, returns a zero-filled `(M, N)` array; if `M == 0` or `N == 0`, returns an empty `(M, N)` array without dispatching.
- **Explicit readback**: `tolist()` synchronizes and downloads device data to a nested Python `list` (`list[float]` for 1D, `list[list[float]]` for 2D)
- **Explicit ownership**: `Device` and `GfxArray` support `close()` and `with ...:` context managers
- **Platform target selection**:
  - macOS (`darwin`) -> `"metal"`
  - Windows (`win32`) -> `"directx"`
  - Linux (`linux`) -> `"vulkan"`
  - Explicit override via `gfxpy.device("vulkan")` (or `"metal"`, `"directx"`)

## Numerical Semantics & Failure Modes
- **Accumulation & tolerances**: All device arithmetic, reductions, and matrix multiplications accumulate in IEEE-754 `float32`. Because parallel shared-memory tree reductions and tiled matmul reorder additions relative to sequential Python `float64` reference sums, results are verified within `float32` tolerances (for example `rel_tol=1e-4, abs_tol=1e-4` for bounded `sum` and `rel_tol=1e-4, abs_tol=1e-3` for `matmul`).
- **Validation & failure modes**: 3D+ or ragged sequences, non-`float32` dtypes, shape mismatches (`ShapeMismatchError`), cross-device operations (`CrossDeviceError`), and use after `close()` (`ClosedResourceError`) raise explicit errors immediately.

## Unsupported Semantics & Non-Claims
- No broadcasting beyond scalar operands
- No 3D+ arrays, slicing, strides, or non-`float32` dtypes
- No operator kernel fusion or arbitrary-axis reductions
- No NumPy compatibility claims
- No BLAS or CuPy performance claims (kernels are straightforward educational GfxIR `@compute` implementations)

## No NumPy Dependency & No CPU Fallback
- Importing `gfxpy` and running host-side contract tests requires only the Python standard library (>= 3.11) and `mlir_gpu`.
- NumPy is **not** required. If you want to compare against NumPy in benchmarks, install the optional `bench` extra (`pip install -e '.[bench]'`).
- Backend initialization or compilation errors always surface the chosen target and cause; `gfxpy` never silently falls back to CPU execution or another GPU backend.

## Backend Prerequisites
- **Linux (`vulkan`)**: Vulkan ICD/loader + `spirv-val` + GfxIR LLVM/MLIR toolchain
- **macOS (`metal`)**: Metal Argument Buffers Tier 2 device + `metal-shaderconverter` + GfxIR toolchain
- **Windows (`directx`)**: DirectX 12 device or WARP + `dxil` tools + GfxIR toolchain
