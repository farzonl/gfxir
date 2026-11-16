# GfxIR

**GfxIR** is an out-of-tree **MLIR `gfx` dialect compiler** and **Python-embedded GPU programming framework**. Write graphics and compute kernels directly in Python and lower them through MLIR to **Vulkan (SPIR-V)**, **DirectX 12 (DXIL)**, and **Metal (`.metallib`)**—without authoring or generating GLSL, HLSL, or MSL.

The repository includes:
- **`mlir_gpu`**: The Python AST-to-MLIR authoring frontend, compiler driver, and cross-backend viewer/compute runtime.
- **`native/compiler`**: The C++ MLIR `gfx` dialect (`gfx-opt`, `gfx-compile`) lowering through upstream `arith`/`vector`/`scf`/`spirv` and `llvm.dx.*` intrinsics.
- **`examples/minigfx`**: A lightweight, single-source Python 3D rendering/game engine proof-of-concept.
- **`examples/gfxpy`**: A NumPy-free, CuPy-inspired GPU array library with persistent device buffers and operator fusion.

---

## Setup

Python **3.11+** is required. The core authoring package has no mandatory runtime dependencies.

```sh
# 1. Build the pinned LLVM/MLIR + native gfx compiler toolchain
python3 tools/build_toolchain.py --jobs 8
export LLVM_BIN="$PWD/.toolchains/llvm-build/bin"
export PATH="$LLVM_BIN:$PWD/.tools/bin:$PATH"

# 2. Install the Python package and viewer dependencies
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[viewer]'
```

> **Note (macOS Vulkan)**: If using the Vulkan viewer via MoltenVK, set `export DYLD_LIBRARY_PATH=/usr/local/lib` (or your Vulkan SDK lib path) before running Python.
> **Note (DirectX on Linux/macOS)**: CPU DXIL execution is supported via [Nacelle](https://github.com/farzonl/Nacelle/releases/tag/v0.0.1) (`python3 tools/setup_nacelle.py`).

---

## Examples

### 1. Compute Shader

Compute kernels use `@compute` with typed `Buffer` bindings, `ParameterBlock` uniforms, and `GlobalInvocationId` / `LocalInvocationId` / `WorkgroupId`.

Run the 2D Mandelbrot compute shader ([`examples/compute/mandelbrot_compute.py`](examples/compute/mandelbrot_compute.py)) to compile and render an RGBA8 PNG on the GPU:

```sh
PYTHONPATH=src .venv/bin/python examples/compute/mandelbrot_compute.py \
  --target vulkan --output build/mandelbrot.png --width 512 --height 512
```

Switch `--target` to `directx` or `metal` (or add `--provider nacelle` / `warp` / `swiftshader`). See [`examples/compute/matmul.py`](examples/compute/matmul.py) for a tiled shared-memory matrix multiplication benchmark using `shared_array` and `workgroup_barrier()`.

---

### 2. Raymarcher / Pixel Shader

Full-screen raymarching shaders pair `fullscreen_vertex` with a `@fragment` shader using `FrameParameters`, `FragmentCoord`, `sdf` helpers, and bounded GPU loops.

Compile and render the animated ray-marched beacon gallery ([`examples/raymarching/beacon_gallery.py`](examples/raymarching/beacon_gallery.py)):

```sh
# Compile to Vulkan SPIR-V (or use --target directx / --target metal)
.venv/bin/python -m mlir_gpu examples/raymarching/beacon_gallery.py \
  --target vulkan --output build/beacon-gallery-vulkan

# Open a live, resize-aware window
.venv/bin/python -m mlir_gpu.viewer build/beacon-gallery-vulkan/pipeline.json \
  --width 640 --height 480

# Or save a deterministic snapshot / MP4 video
.venv/bin/python -m mlir_gpu.viewer build/beacon-gallery-vulkan/pipeline.json \
  --width 320 --height 240 --time 0.25 --output build/gallery.png
```

Also see [`examples/raymarching/light_cycle.py`](examples/raymarching/light_cycle.py) for a 110-step volumetric ray-marched tunnel scene.

---

### 3. Combination Vertex & Pixel Shader

Graphics pipelines combine custom `@vertex` and `@fragment` entry points in a `Pipeline`. Vertex outputs (like `position` and interpolated `color`) flow directly into the fragment stage with no separate shader files or duplicated interface declarations.

Compile and view the vertex+fragment triangle ([`examples/basics/triangle.py`](examples/basics/triangle.py)):

```sh
# Compile
.venv/bin/python -m mlir_gpu examples/basics/triangle.py \
  --target vulkan --output build/triangle-vulkan

# Live window (or add --output build/triangle.png for offscreen rendering)
.venv/bin/python -m mlir_gpu.viewer build/triangle-vulkan/pipeline.json \
  --width 800 --height 600
```

For indexed/multi-vertex quads and HDR fullscreen post-processing, see [`examples/basics/quad.py`](examples/basics/quad.py) and [`examples/basics/post_process.py`](examples/basics/post_process.py).

---

## MiniGfx (Game & Rendering Engine)

[**MiniGfx**](examples/minigfx/README.md) is a lightweight real-time 3D rendering engine proof-of-concept inspired by Microsoft MiniEngine. It eliminates traditional CPU/GPU language bifurcation by defining shared `@struct` layouts, math, materials, and shaders in one Python codebase:

- **Single-source host/device types**: `@struct` classes pack bytes for the CPU and bind directly into `@vertex`/`@fragment` shaders without `cbuffer` or `std140` duplication.
- **Engine features**: Scene graph, OBJ mesh loading, orbit/fly camera, depth buffering, textured materials, HDR render targets with tone mapping, debug overlays, and an interactive HUD.
- **Cross-backend**: Runs the same Python engine code on Vulkan, DirectX 12, Metal, and Nacelle.

Launch the MiniGfx `ModelViewer` sample:

```sh
PYTHONPATH=src .venv/bin/python -m examples.minigfx --width 800 --height 600
```

See [`examples/minigfx/README.md`](examples/minigfx/README.md) for controls, architecture details, and backend options.

---

## gfxpy (NumPy-Free GPU Array Alternative)

[**gfxpy**](examples/gfxpy/README.md) is a NumPy-free, CuPy-inspired array library built on GfxIR's persistent compute runtime. Expressions execute on **Linux Vulkan**, **Windows DirectX 12**, and **macOS Metal** (plus **Nacelle** CPU DXIL) with intermediate results staying device-resident and chained elementwise operations fused into single kernels:

```python
import gfxpy as gp

with gp.device("auto"):  # macOS -> metal, Windows -> directx, Linux -> vulkan
    a = gp.array([1.0, 2.0, 3.0], dtype=gp.float32)
    b = gp.array([4.0, 5.0, 6.0], dtype=gp.float32)
    c = gp.array([0.5, 1.0, 1.5], dtype=gp.float32)

    # Stays on the GPU; no host readback between operators
    y = (a * b + 1.5) * 2.0 + c
    print(y.tolist())
```

Run the chained GPU array example:

```sh
.venv/bin/python examples/gfxpy/chain_example.py --target auto
```

See [`examples/gfxpy/README.md`](examples/gfxpy/README.md) for supported dtypes, reductions, linear algebra, and operator fusion details.

---

## Running Tests

```sh
# Compiler IR / lit tests
cmake --build .toolchains/llvm-build --target check-gfx

# Python unit tests
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
