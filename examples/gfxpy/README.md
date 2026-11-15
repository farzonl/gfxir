# gfxpy (Example GPU Array API on GfxIR)

`gfxpy` is a NumPy-free, CuPy-inspired example array library built on GfxIR's `@compute` compiler and persistent GPU runtime (`open_compute_context`).

## Supported Operations (First Milestone)
- **1D contiguous `float32` arrays**: `array(seq, dtype=float32)`, `empty(length, dtype=float32)`
- **Device-resident elementwise arithmetic**: `+` and `*` (array/array and array/scalar) executed via bounds-checked GfxIR `@compute` kernels without intermediate host transfers
- **Explicit readback**: `tolist()` synchronizes and downloads device data to a Python `list[float]`
- **Explicit ownership**: `Device` and `GfxArray` support `close()` and `with ...:` context managers
- **Platform target selection**:
  - macOS (`darwin`) -> `"metal"`
  - Windows (`win32`) -> `"directx"`
  - Linux (`linux`) -> `"vulkan"`
  - Explicit override via `gfxpy.device("vulkan")` (or `"metal"`, `"directx"`)

## Unsupported Semantics
- No broadcasting beyond scalar operands
- No multidimensional arrays, slicing, strides, or non-`float32` dtypes
- No reductions, matrix multiplication, or operator kernel fusion in this increment
- No NumPy compatibility claims

## No NumPy Dependency & No CPU Fallback
- Importing `gfxpy` and running host-side contract tests requires only the Python standard library (>= 3.11) and `mlir_gpu`.
- NumPy is **not** required. If you want to compare against NumPy in benchmarks, install the optional `bench` extra (`pip install -e '.[bench]'`).
- Backend initialization or compilation errors always surface the chosen target and cause; `gfxpy` never silently falls back to CPU execution or another GPU backend.

## Backend Prerequisites
- **Linux (`vulkan`)**: Vulkan ICD/loader + `spirv-val` + GfxIR LLVM/MLIR toolchain
- **macOS (`metal`)**: Metal Argument Buffers Tier 2 device + `metal-shaderconverter` + GfxIR toolchain
- **Windows (`directx`)**: DirectX 12 device or WARP + `dxil` tools + GfxIR toolchain
