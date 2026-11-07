# GfxIR

Author GPU programs as a **Python embedded API**, lower them through **MLIR**,
and target Vulkan, DirectX 12, and Metal without authoring or generating GLSL,
HLSL, or MSL.

This is a graphics-first **out-of-tree `gfx` MLIR compiler foundation** with
Vulkan and DXIL-to-Metal viewers. Python is a bootstrap authoring frontend, not
the compiler's internal representation or a permanent restriction on frontends.
The examples define vertex and fragment stages for a triangle and a UV-driven quad.
Compile them into artifacts, then open a live window or render a PNG through
Vulkan (MoltenVK on macOS) or native Metal using converted DXIL. The viewer loads
the actual compiled shaders; it does not redraw the triangle with CPU code or
generate shading-language source.

## Status

| Path | Implementation | Verification |
| --- | --- | --- |
| Python → structured graph → native `gfx` MLIR | Implemented | TableGen/ODS operations, types, attributes, native builders and verifiers |
| `gfx` → upstream MLIR SPIR-V → `.spv` | Implemented | Upstream arithmetic/vector/math/function/SCF conversion and binary serializer; Vulkan 1.0 validation |
| `gfx` → MLIR LLVM → LLVM DirectX backend | Implemented | LLVM main/24.0.0git, modern DirectX intrinsics, upstream signature helpers and DXIL/container emission |
| DXIL validation/signing | Microsoft native validator API | The compiler validates/signs LLVM-produced containers; no HLSL parser or textual IR patching |
| DXIL → `.metallib` | Metal Shader Converter adapter | Tested with converter 3.0, preserving entry-point names |
| Vulkan graphics rendering | GLFW window and offscreen PNG readback | Triangle and UV quad tested on macOS/MoltenVK with Vulkan validation, pixel checks, presentation, and window resizing |
| Native Metal rendering of DXIL | MetalKit window and offscreen readback using Apple's converter runtime | Tested on Apple M3 Max with API/GPU validation, pixel checks, and persistent-window tests |
| Native DirectX 12 runtime rendering | Not implemented | DirectX artifacts are previewed through Metal on macOS, not a DirectX device |
| Compute entry points | IR foundation only | Typed stage, workgroup-size and invocation-ID verification; compute lowering/execution is explicitly rejected for now |
| Typed pointers / bindless handles | IR foundation only | ODS types parse and round-trip; memory operations, ABI/layout legalization and resource heaps are not implemented yet |

LLVM IR verification is **not** DXIL validation: the native DXC validator also
checks and signs the actual containers. Real rendering tests cover Vulkan and
DXIL-to-Metal in addition to mocked command-wiring tests. Native Windows
DirectX runtime interoperability remains unverified. No path emits fake binaries
or falls back to shading-language source.

## Compiler architecture

The compiler lives in [native/compiler](native/compiler). Its
[TableGen definitions](native/compiler/include/Gfx) define `gfx` stage and
interpolation attributes, typed pointers, resource handles, builtin inputs,
varying inputs/outputs, and clip-position writes. Entry points are standard
`func.func` operations annotated with `gfx.stage`; they are not tied to a draw
count or the triangle example. Arithmetic and vector construction/extraction
use upstream `arith` and `vector` operations.

The Python frontend sends a typed, indexed JSON graph to `gfx-compile`, which
constructs real MLIR operations through C++ builders. It does not generate
textual MLIR, LLVM assembly, or SSA variable names. Textual IR is an inspectable
compiler output and a native-tool input format, not a string-based lowering
implementation.

- `gfx-opt`: parse, verify and round-trip IR; run `--gfx-verify`,
  `--convert-gfx-to-spirv`, or `--gfx-to-directx`.
- `gfx-compile`: import structured JSON or `--input-format=mlir`, construct and
  verify IR, optimize it, emit per-stage reflection, and lower to SPIR-V or
  modern LLVM IR.
- Vulkan uses upstream MLIR SPIR-V conversion patterns and serialization.
- DirectX uses upstream `llvm.dx.*` intrinsics and LLVM's official DirectX
  backend. LLVM handles instruction scalarization, `dx.op` lowering, old-bitcode
  compatibility, signature translation and DXContainer emission. The project
  uses LLVM's typed semantic-signature API, not regexes or hand-numbered metadata.
  Interface information retains MLIR types and enums through lowering; JSON
  reflection is serialized only at the artifact boundary. Reflection is captured
  after optimization so removed inputs cannot corrupt signature numbering.
- Metal still uses the supported DXIL-to-metallib converter boundary. This is
  not a direct AIR backend, and no MSL fallback is generated.

This is the graphics-first starting point of the unified-dialect roadmap, not a
claim that its later milestones are complete. Compute has a distinct entry-point
contract from day one. Pointer-first/bindless memory lowering, shared host/device
struct layouts, Python structured control flow, native host-language frontends, mesh/task
execution, ray queries and differentiation remain future work with explicit
capability boundaries.

## Try it

Python 3.11+ is sufficient for the authoring API and unit tests; the core
Python package has no mandatory Python dependencies. Compilation requires the
native compiler and a matching LLVM/MLIR toolchain. The installed Homebrew LLVM
21 does not contain the DirectX backend. Build a separate pinned main checkout:

```sh
python3 tools/build_toolchain.py --jobs 8
export LLVM_BIN="$PWD/.toolchains/llvm-build/bin"
export PATH="$LLVM_BIN:$PWD/.tools/bin:$PATH"
```

The source revision is pinned in [tools/llvm-revision.txt](tools/llvm-revision.txt);
LLVM and MLIR are built together with the native CPU and DirectX targets. The
script checks the checkout revision and refuses local tracked modifications.
Build products remain under `.toolchains`; Homebrew LLVM is not modified.
CMake, Ninja, Git and a C++17 compiler are required. On macOS, Xcode supplies the
matching SDK selected through `xcrun`. The first source build is substantial;
later invocations are incremental.

Run from this directory:

```sh
export PYTHONPATH="$PWD/src"

python3 -m unittest discover -s tests -v
python3 -m mlir_gpu examples/triangle.py \
  --target vulkan --output build/triangle-vulkan
```

Install SPIRV-Tools separately. Vulkan compilation requires the built
`gfx-compile` and `spirv-val`; DirectX additionally requires the matching `llc`
and the native `dxv` helper below. Integration tests explicitly
skip when their tools are unavailable; inspect the test report for skips.
The output directory must not already exist. Compilation is staged in a
temporary sibling directory and published only after both stages succeed.

The output includes the structured graph, original and lowered MLIR, native
binaries, per-stage compiler-generated reflection, and
`pipeline.json` with stage names, binary paths, linked interfaces, a separate draw description, and
`runtime_verified: false`. Compilation alone cannot verify execution; the viewer
does not mutate this compiler-produced flag. Python input files are executed
normally: only compile code you trust.

### See the triangle

Install the viewer into a project-local virtual environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[viewer]'
```

You also need a system Vulkan loader and driver. On macOS, install the Vulkan
SDK with MoltenVK. If the Python binding cannot find its library, set the loader
search path **before starting Python** (the SDK on this machine installs it in
`/usr/local/lib`; adjust for your installation):

```sh
export DYLD_LIBRARY_PATH=/usr/local/lib
```

Use the already-compiled manifest from the previous step:

```sh
.venv/bin/python -m mlir_gpu.viewer build/triangle-vulkan/pipeline.json
```

Close the window or press **Esc** to exit. The window supports resizing and
minimizing. It uses a Vulkan swapchain with a single frame in flight, no vertex
buffers, and the manifest's vertex/fragment entry-point names. A desktop session
is required. The installed command `mlir-gpu-view` provides the same interface.

To save a GPU-rendered image without opening a window:

```sh
.venv/bin/python -m mlir_gpu.viewer build/triangle-vulkan/pipeline.json \
  --output build/triangle.png --width 800 --height 600 --validation
open build/triangle.png  # macOS
```

The PNG path must be new; existing files are not overwritten. Requested preview
dimensions must be between 1 and 4096. `--validation` requires the Vulkan SDK's
Khronos validation layer and fails on validation errors, including errors during
resource teardown. Omit it if the layer is not installed. Offscreen rendering
still requires a Vulkan driver, but does not require GLFW or a desktop session.
On macOS this uses Vulkan through MoltenVK, **not** the native DXIL-to-Metal
compiler adapter.

For a bounded window smoke test, use `--frames 3` instead of `--output`.
`-frames` is also accepted: `-frames 0` and `--frames 0` both mean render
continuously until Esc or window close, as does omitting the option entirely.
Do **not** pass `--output` for a live preview: it explicitly selects PNG-only
mode and cannot be combined with either spelling of the frames option.
The preview accepts reflected f32 scalar/vector varyings and configurable
triangle-list vertex counts, with one `f32x4` color attachment. Vertex buffers,
resource bindings, indexed/instanced draws, other topologies and multiple render
targets are not implemented in the preview; incompatible interfaces are rejected.

### A different interface and draw count

[The quad example](examples/quad.py) emits six vertices, passes a named `vec2` UV
varying instead of a fixed `vec4` color, and computes nonlinear color in the
fragment shader:

```sh
python3 -m mlir_gpu examples/quad.py --target metal --output build/quad-metal
.venv/bin/mlir-gpu-view build/quad-metal/pipeline.json -frames 0
```

`Pipeline(..., draw=Draw(vertex_count=6))` controls the preview command separately
from shader IR. Both live and offscreen Vulkan/Metal paths use that draw count.

### One coordinate convention for every target

The authoring API defines homogeneous clip coordinates with **X right, Y up,
and depth from 0 to W**. After perspective division, visible Z is in `[0, 1]`.
Readback images have a top-left origin and half-integer pixel centers.
All viewers use the full attachment viewport, one sample, no face culling, and
linear RGBA/BGRA8 UNORM attachments. Live windows declare sRGB presentation;
the Vulkan viewer rejects incompatible surface formats rather than silently
switching to an sRGB-encoding attachment and changing the colors.

Vulkan's positive-height viewport maps clip Y in the opposite direction from
DirectX/Metal. The SPIR-V lowering therefore negates **only clip Y**, preserving
X, Z, and W. This correction is in the compiled vertex shader, so it applies to
both live rendering and PNG export; no target-specific author code or PNG flip
is involved. The example's blue corner points up and its red/green corners
remain bottom-left/bottom-right on every implemented runtime.

New manifests use **schema version 3**, with compiler-generated stage reflection,
a separate draw description and `raster_convention`. Corrected version-2
artifacts remain viewable through a legacy compatibility check. Version-1
binaries predate normalization and require recompilation into a new directory.
Existing artifact directories and PNGs are not rewritten.

### Rendering tests

Regular unit tests do not require a GPU or open windows. To additionally compile
the triangle and check actual Vulkan readback pixels:

```sh
LLVM_BIN=/path/to/llvm/bin MLIR_GPU_TEST_VULKAN=1 MLIR_GPU_TEST_VALIDATION=1 \
  .venv/bin/python -m unittest discover -s tests -v
```

On a desktop session, also set `MLIR_GPU_TEST_WINDOW=1` to exercise real window
presentation and resizing. Keep `DYLD_LIBRARY_PATH` set as above on macOS.
The pixel test checks background pixels, interpolated vertex colors, and
triangle coverage (17.5%–18.5% of the image), not just successful API calls.
Opting into rendering tests makes missing runtime/compiler dependencies a
failure rather than a silent skip. DirectX/Metal native compilation tests remain
separate and skip when their toolchains are unavailable. Set
`MLIR_GPU_TEST_METAL=1` to also test actual DXIL-to-Metal pixels; with
`MLIR_GPU_TEST_WINDOW=1`, this includes a bounded native window and a test that
the default native window stays open. Metal API/GPU validation uses the same
`MLIR_GPU_TEST_VALIDATION=1` switch. Include `.tools/bin` on `PATH` as below.

Enable both GPU paths to run cross-target parity tests as well:

```sh
PATH="$PWD/.tools/bin:$PATH" LLVM_BIN=/path/to/llvm/bin \
  MLIR_GPU_TEST_VULKAN=1 MLIR_GPU_TEST_METAL=1 MLIR_GPU_TEST_VALIDATION=1 \
  MLIR_GPU_TEST_WINDOW=1 .venv/bin/python -m unittest discover -s tests -v
```

These compare Vulkan, DXIL converted at preview time, and precompiled metallibs.
They check identical coverage masks and at most **2/255** channel differences,
plus projected bounds within two pixels of the mathematical bounds.
Fixtures include the RGB triangle, asymmetric geometry with varying homogeneous
W, near/far clipping, and the six-vertex UV quad at square, non-square, and odd resolutions.
This tests shared geometry, scale, perspective interpolation, and color placement,
not just independently plausible images. It does not promise bit-identical
floating-point results on arbitrary hardware or verify a Windows DirectX runtime.

### Other native targets

```sh
python3 -m mlir_gpu examples/triangle.py \
  --target directx --output build/triangle-directx
python3 -m mlir_gpu examples/triangle.py \
  --target metal --output build/triangle-metal
```

Both use native `gfx` lowering into MLIR's LLVM dialect, LLVM IR translation,
and the pinned LLVM DirectX backend targeting Shader Model 6.6. LLVM's DXIL
writer handles compatibility with the older bitcode format. The compiler does
not invoke `dxil-assemble`, manually scalarize expression graphs, or patch
LLVM IR text.

The native prerequisites on macOS are:

- Xcode C++ tools, CMake, and Ninja.
- Microsoft's `libdxcompiler` and `dxc/dxcapi.h` (available in the Vulkan SDK's
  DXC component; already installed under `/usr/local` on this machine).
- Apple's **Metal Shader Converter** and its
  `metal_irconverter_runtime/metal_irconverter_runtime.h` headers.
  Xcode's `metal`/`metallib` commands alone do not provide this converter.

Build and install the project-local native tools:

```sh
cmake -S . -B build/native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PWD/.tools" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0 \
  -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
cmake --build build/native
cmake --install build/native
export PATH="$PWD/.tools/bin:$PATH"
```

Adjust the SDK path for your Xcode installation; an explicit matching SDK avoids
mixing an older Xcode linker with a newer Command Line Tools SDK. For SDKs
installed outside standard prefixes, use `CMAKE_PREFIX_PATH`.

The installed `dxv` and `dxil-assemble` are **this project's small public-API
frontends** to Microsoft's native `IDxcValidator` and `IDxcAssembler` in
`libdxcompiler`, not downloaded upstream standalone executables. They contain
no replacement validation logic. `dxil-assemble input.ll -o unsigned.dxil`
assembles compatible legacy IR as an optional utility; the compiler instead uses
LLVM `llc` for container generation. `dxv unsigned.dxil -o signed.dxil` validates
and signs it. Diagnostics and nonzero exits propagate to the compiler, with no partial
published pipeline. Native tools retain the DXC library directory in their
runtime search path.

Metal conversion explicitly selects and preserves the manifest's entry-point
names. The renderer uses Apple's `IRRuntimeDrawPrimitives`, including its draw
argument/uniform bindings, rather than assuming a converted shader has the same
ABI as a hand-written Metal shader. Argument Buffers Tier 2 and macOS 14+ are
required.

### Preview DirectX shaders through Metal

After building the tools and compiling a **new** DirectX output directory:

```sh
LLVM_BIN=/path/to/llvm/bin .venv/bin/python -m mlir_gpu examples/triangle.py \
  --target directx --output build/triangle-directx
.venv/bin/mlir-gpu-view build/triangle-directx/pipeline.json
# Equivalently, explicitly request an unbounded live window:
.venv/bin/mlir-gpu-view build/triangle-directx/pipeline.json -frames 0
```

The preview revalidates the DXIL and converts it into temporary metallibs before
opening a native **MetalKit** window. It does **not** use Vulkan or GLFW on this
path. It stays open until Esc or window close (`--frames 0`, the default).
The same viewer also accepts a `--target metal` manifest without reconversion.
Both formats support PNG export and Metal API/GPU validation:

```sh
.venv/bin/mlir-gpu-view build/triangle-directx/pipeline.json \
  --output build/triangle-directx.png --validation
```

This proves DXIL-to-Metal execution on macOS, **not** native DirectX 12 runtime
execution on Windows. The original compiler manifest remains unchanged with
`runtime_verified: false`; rendering tests separately establish execution.

Use `--llvm-bin` / `LLVM_BIN` for LLVM tools, and `--tools-bin` / `GPU_TOOLS_BIN`
for `spirv-val`, `dxv`, `metal-shaderconverter`, and the native
viewer; otherwise tools use `PATH`. Prefer adding `.tools/bin` to `PATH` when
using the local helpers alongside SDK-installed tools.
An explicit directory is authoritative, so a missing executable there is an
error rather than a silent fallback to another installed version.

## Programming model

See [the triangle](examples/triangle.py). Ordinary Python helpers compose traced
device expressions, with no custom parser, source rewriting, or shader strings.
`@vertex` receives a `u32` vertex index and returns `Vertex(position, color=...)`
or `Vertex(position, varyings={"name": value, ...})`.
`@fragment` parameters resolve against those varying names and types when traced
through `Pipeline`; a fragment shader may also take no inputs. Fragment outputs
are device values or a mapping of named render targets. The compiler links their
interfaces from native reflection; the preview currently supports one `f32x4` target.

### Native MLIR and device loops

`MLIRPipeline` accepts vertex/fragment MLIR files and their declared entry names,
using the same validated artifact, reflection, and viewer pipeline as Python
authoring. Entry names/stages must match native reflection. Missing files,
malformed IR, unsupported target operations, and interface mismatches fail
before publishing the output directory.

Native graphics lowering supports `scf.for`, `scf.while`, `scf.if`, and
direct calls to defined, nonrecursive helpers with at most one result.
Helpers may return vectors; loops carry multiple scalars/vectors. Use structured
regions, not raw multi-block CFGs or indirect/recursive calls. Shader IO belongs
to the marked entry; helpers are ordinary functions. The compiler converts only
the reachable call graph and resolves reflection by the marked entry, not
function order. Upstream SPIR-V conversion performs function/signature and
structured merge/loop legalization for Vulkan. DirectX reuses upstream
SCF-to-CF and CF/function-to-LLVM conversion. Its profile-6.6 legalization
inlines ordinary helpers with upstream MLIR, then removes unused helpers,
before LLVM handles instruction scalarization. This avoids unsupported vector
values crossing helper-call boundaries in DXIL; it does not unroll loops.
Only the marked entry receives shader attributes/signature metadata.

[The sphere march example](examples/sphere_march.py) loads a native
[sphere marcher](examples/ir/sphere_march.mlir). It executes a bounded GPU while
loop with early hit/miss termination, a distance helper, a nested conditional,
and carried iteration/distance/hit/glow state:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/sphere_march.py \
  --target vulkan --output build/sphere-march-vulkan
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/mlir-gpu-view \
  build/sphere-march-vulkan/pipeline.json --width 256 --height 256 -frames 0 --validation
```

The example intentionally uses a fixed pixel camera centered at `(128, 128)`;
it is not resize-aware or animated until frame resources land. It performs at
most 64 iterations per pixel. Increasing the cap changes the loop bound, not
shader-body duplication. The bound is authored in IR; the compiler does not
guarantee termination for arbitrary native programs.

Both examples support Vulkan, DirectX, and converted Metal; the `--target`
option selects the backend, not the example's filename. The
[loop bands example](examples/loop_bands.py) additionally shows color
bands where each pixel selects the number of GPU loop iterations:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/loop_bands.py \
  --target directx --output build/loop-bands-directx
PATH="$PWD/.tools/bin:$PATH" .venv/bin/mlir-gpu-view \
  build/loop-bands-directx/pipeline.json -frames 0 --validation
```

Use `--target metal` and a fresh directory for preconverted metallibs. On macOS
the DirectX artifact is rendered through Metal, not a native DirectX device.
Python `if`/loops on device values and the extended JSON region/function
importer remain separate pending work.

### Arithmetic and vectors

Device values support `+`, `-`, `*`, floating-point `/`, unary negation, `abs`,
scalar comparisons, and eager Boolean `&`, `|`, `^`, `~`. Python `if`, `and`,
`or`, and loops still cannot branch on device values; use `select` for now.
Vector arithmetic broadcasts f32 scalars, without implicitly converting device
integers to floats or broadcasting the arms of `select`.

```python
from mlir_gpu import maximum, minimum, vec2, vec3

p = vec3(vec2(0.25, 0.75), 0.0)
q = abs(-p).with_components(xy=p.yx)
distance = maximum(q.xy - 0.5, 0.0) / 2.0
bounded = minimum(distance, 1.0)
```

Vector constructors accept a scalar splat or exactly the requested number of
components, including composed vectors. Read swizzles use `xyzw` or `rgba`;
`with_components` returns a new vector and rejects overlapping replacements.
Direct component assignment is not supported.

`i32` and `u32` create explicit local integer scalars or reinterpret each other's
32-bit bit patterns; they do not convert floats. Integer add/subtract/multiply
wrap modulo 2^32, with signed comparisons for `i32` and unsigned comparisons for
`u32`. Signed negation and `abs(INT32_MIN)` wrap to `INT32_MIN`. Integer division
and integer rasterization interfaces are not exposed by this frontend yet.

Floating `minimum`/`maximum` return the numeric operand if just one operand is
NaN, and NaN if both are NaN. For a zero tie, the result is numerically zero but
its sign is unspecified; NaN payloads are unspecified. Floating `abs` maps
negative zero to positive zero. No fast-math flags are enabled. Float literals
must be finite f32 values, although arithmetic can produce infinity/NaN.

Numeric literals are coerced to `f32` when combined with a traced float or used
as vector components. Device input values cannot leak between stages. `select`
is an expression: both arms are traced, not lazy control flow.

The [arithmetic example](examples/arithmetic.py) draws a full-screen gradient
with a square outline using the new operations:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/arithmetic.py \
  --target metal --output build/arithmetic-metal
PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/mlir-gpu-view build/arithmetic-metal/pipeline.json -frames 0
```

Use `--target vulkan` or `--target directx` with a fresh output directory to
compile the same example for the other targets.

### Device math

Use `from mlir_gpu import shader_math as sm`, not Python's host `math` module,
for `sin`, `cos`, `sqrt`, `exp`, `exp2`, `log`, `log2`, `floor`, and `pow`.
These functions build upstream MLIR `math` operations for f32 scalars/vectors.
`pow` allows scalar/vector broadcasting.

The portable contract requires finite inputs, nonnegative `sqrt` inputs,
positive `log`/`log2` inputs, and positive `pow` bases (also zero with a positive
exponent). Negative bases, zero to a nonpositive exponent, and out-of-domain
dynamic inputs have no portable result guarantee. Known literal-domain errors
are rejected at authoring time; dynamic domain validity is the program's
responsibility. Positive powers of zero return zero using a guarded native
lowering, because upstream logarithmic power expansion excludes zero.

Transcendentals are approximate f32 operations, not cross-device bit-exact
Python/libm results. No fast-math flags, contraction permission, or host libm
calls are added. Overflow, underflow/subnormal handling, infinity/NaN payloads,
and the sign of zero are target-dependent outside the stated finite-result
contract. Tests check scalar/vector identities with absolute error <= 2e-5
over the example's bounded range and rendered colors against CPU references
and other targets within 2/255; these are tested ranges, not a universal ULP
guarantee. The existing arithmetic-only exact-parity tests remain unchanged.

[The math example](examples/math.py) adds procedural rings, stripes, glow, and
grid cells without frame resources or loops:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/math.py \
  --target metal --output build/math-metal
PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/mlir-gpu-view build/math-metal/pipeline.json -frames 0
```

### Full-screen drawing and fragment coordinates

`fullscreen_vertex` covers the framebuffer with a single oversized triangle and
no vertex buffers. Annotate a fragment parameter with `FragmentCoord` to read
device-provided f32 pixel-center XY, independently of user varying names:

```python
from mlir_gpu import FragmentCoord, Pipeline, fragment, fullscreen_vertex, vec4

@fragment
def pixel_coordinates(pixel: FragmentCoord):
    return vec4(pixel / 256, 0, 1)

pipeline = Pipeline(fullscreen_vertex, pixel_coordinates)
```

Coordinates have a top-left framebuffer origin, X right, Y down, and pixel
centers at `(x + 0.5, y + 0.5)`. Units are framebuffer pixels, not logical
window points. Only XY is exposed; depth and reciprocal clip-W are intentionally
not part of this API. Native lowering reads SPIR-V `FragCoord` / DXIL
`SV_Position`, projects XY, and records a distinct builtin in reflection.
Existing user varying locations and builtin signature IDs stay independent.

For a Y-up camera, explicitly transform pixels using a positive framebuffer
resolution: `camera = (pixel - resolution * 0.5) / resolution.y`, then
`camera = camera.with_components(y=-camera.y)`. The resolution can currently
be a compile-time `vec2(width, height)`; resize-aware reflected frame parameters
are a separate pending feature, not inferred or baked into this builtin.

[The coordinates example](examples/coordinates.py) shows 32-pixel color cells
and 128-pixel bands. Their pixel sizes stay constant when the window resizes:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/coordinates.py \
  --target metal --output build/coordinates-metal
PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/mlir-gpu-view build/coordinates-metal/pipeline.json -frames 0
```

### Reusable shader and distance-field helpers

`shader_math` also provides `dot`, `cross`, `length`, `normalize`, `rotate2`,
`clamp`, `mix`, `smoothstep`, `fract`, and floor-based `mod`.
These are small pure device-expression helpers composed from the existing
arithmetic/vector/math primitives, not new `gfx` operations. `rotate2` is a
counterclockwise mathematical rotation; use an explicit Y-up coordinate
conversion if starting from framebuffer pixels.

`from mlir_gpu import sdf` exposes `sphere`, `box`, `box2`, centered `repeat`,
`fold_square`, and tetrahedral finite-difference `normal`. Box sizes are
half-extents. For example:

```python
point = vec3(0.5, 0.25, 0.75)
distance = sdf.box(point, vec3(0.4))
normal = sdf.normal(lambda p: sdf.sphere(p, 1.0), point)
```

`normalize` returns zero for a zero vector. Flat distance gradients likewise
produce a zero normal, rather than silently choosing a direction. Nonzero
lengths must be representable in the bounded f32 domain; singular SDF points,
underflow/overflow, and nondifferentiable corners do not guarantee a unit
geometric normal. Normal epsilon and modulo/repetition periods must be
positive, sphere radii and box half-extents nonnegative, and smoothstep edges
strictly increasing. Known literal-domain errors are rejected; dynamic
domain validity remains the shader author's responsibility.

`mod(x, period)` uses `x - period * floor(x / period)`, including for negative
coordinates. It is not truncating remainder: `mod(-0.25, 2) = 1.75` and centered
`repeat(-0.25, 2) = -0.25`. `mix` permits extrapolation; clamp bounds must be
ordered. These helpers preserve the numerical/NaN policy of their primitives.

[Distance fields](examples/distance_fields.py) shows rotated, repeated box
slices with smooth edges and glow. [Surface normals](examples/surface_normals.py)
shows sphere normals as RGB. Both are visual helper demonstrations, not a
ray marcher yet:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/distance_fields.py \
  --target metal --output build/distance-fields-metal
PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/mlir-gpu-view build/distance-fields-metal/pipeline.json -frames 0
```

Compile `examples/surface_normals.py` into another fresh directory for the
second example. The current frontend deliberately inlines these small pure
helpers, including four fixed normal samples, during tracing. It does not
unroll a ray-marching loop; GPU loops/functions remain separate pending work.

Python executes once during tracing. Python `if`/loops cannot branch on device
values: use `select` for now. Host-side loops and metaprogramming can construct
expressions. This is not a compiler for arbitrary Python, dynamic GPU function
pointers, recursion, or a claim that those features are portable.

## Design decisions

Inspired by [Death to Shading Languages](https://xol.io/blah/death-to-shading-languages/):

- **Embed rather than invent a language.** Reuse Python's functions, modules,
  tooling, and host-side metaprogramming. Do not copy GLSL's source-level model.
- **Keep one program and one interface contract.** Stage interfaces live in the
  Python API and artifact manifest, not independently maintained shader files.
  Rich shared host/device data layouts and buffer arguments are future work;
  shader interfaces and runtime draw commands are separate concerns.
- **Use MLIR as compiler infrastructure.** Traced typed expressions lower to
  a TableGen-defined `gfx` dialect composed with upstream operations, then pass
  through verification, canonicalization, common-subexpression elimination,
  reflection and native conversion passes. Target-specific operations stay at
  the lowering boundary rather than leaking into the authoring API.
- **Keep runtime formats at the boundary.** Vulkan consumes SPIR-V; DirectX
  consumes validated DXIL containers; Metal consumes metallibs through Apple's
  supported DXIL conversion route. No undocumented hand-authored AIR.
- **Make capability limits explicit.** This initial portable subset does not
  impose a long-term ban on pointers or higher-order abstractions, but neither
  does MLIR make unsupported runtime features magically available. Future
  lowering must specialize or reject features with explicit target diagnostics.
- **Separate compilation from execution.** Successful binary validation does
  not assert successful pipeline creation, ABI interoperability, or rendering.

Next steps are pointer/resource operations with verified host/device layouts,
capability-driven legalization and meaningful compute execution, followed by
broader graphics features and a native Windows DirectX 12 runtime adapter.
The shared dialect is designed for both graphics and compute, while this first
executable milestone remains graphics-first.

### Backend references

- [MLIR SPIR-V dialect and serialization](https://mlir.llvm.org/docs/Dialects/SPIR-V/)
- [LLVM DirectX target](https://llvm.org/docs/DirectXUsage.html)
- [Stable DXIL specification and signature metadata](https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/DXIL.rst)
- [DXIL validator/signing utility](https://github.com/microsoft/DirectXShaderCompiler/blob/main/tools/clang/tools/dxv/dxv.cpp)
- [Apple Metal Shader Converter](https://developer.apple.com/metal/shader-converter/)
