# gfxpy (Example GPU Array API on GfxIR)

`gfxpy` is a NumPy-free, CuPy-inspired example array library built on GfxIR.

## First-Milestone Scope
- **1D contiguous `float32` arrays**: `array(seq, dtype=float32)`, `empty(length, dtype=float32)`
- **Elementwise arithmetic**: `+` and `*` (array/array and array/scalar)
- **Explicit readback**: `tolist()`
- **Explicit ownership**: `Device` and `GfxArray` support `close()` and `with ...:` context managers
- **Platform target selection**:
  - macOS (`darwin`) -> `"metal"`
  - Windows (`win32`) -> `"directx"`
  - Linux (`linux`) -> `"vulkan"`
  - Explicit override via `gfxpy.device("vulkan")` (or `"metal"`, `"directx"`)

## No NumPy Dependency & No CPU Fallback
- Importing `gfxpy` and running host-side contract tests requires only the Python standard library (>= 3.11).
- NumPy is **not** required. If you want to compare against NumPy in benchmarks, install the optional `bench` extra (`pip install -e '.[bench]'`).
- Backend initialization or runtime errors always surface the chosen target and cause; `gfxpy` never silently falls back to CPU execution or another GPU backend.

## Backend Prerequisites
GPU execution is provided by the persistent compute runtime backends (tracked in `#178`, `#179`, `#181`):
- **Linux (`vulkan`)**: Vulkan ICD/loader + `spirv-val` + GfxIR LLVM/MLIR toolchain
- **macOS (`metal`)**: Metal Argument Buffers Tier 2 device + `metal-shaderconverter` + GfxIR toolchain
- **Windows (`directx`)**: DirectX 12 device or WARP + `dxil` tools + GfxIR toolchain
