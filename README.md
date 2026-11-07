# GfxIR

Author GPU programs as a **Python embedded API**, lower them through **MLIR**,
and target Vulkan, DirectX 12, and Metal without authoring or generating GLSL,
HLSL, or MSL.

This is a graphics-first **out-of-tree `gfx` MLIR compiler** with
Vulkan and DXIL-to-Metal viewers. Python is a bootstrap authoring frontend, not
the compiler's internal representation or a permanent restriction on frontends.
Examples range from triangle/quad basics to an original animated ray-marched
[beacon gallery](examples/beacon_gallery.py). Compile them into artifacts,
then open a live window or render a PNG through
Vulkan (MoltenVK on macOS) or native Metal using converted DXIL. The viewer loads
the actual compiled shaders; it does not redraw the triangle with CPU code or
generate shading-language source.
All three preview paths can export deterministic MP4 animation.

## Status

| Path | Implementation | Verification |
| --- | --- | --- |
| Python graph / typed structured JSON → native `gfx` MLIR | Implemented | Compatible flat graphs plus versioned function/block/region/SSA input; upstream ODS verification |
| `gfx` → upstream MLIR SPIR-V → `.spv` | Implemented | Upstream arithmetic/vector/math/function/SCF conversion and binary serializer; Vulkan 1.0 validation |
| `gfx` → MLIR LLVM → LLVM DirectX backend | Implemented | LLVM main/24.0.0git, modern DirectX intrinsics, upstream signature helpers and DXIL/container emission |
| DXIL validation/signing | Microsoft native validator API | The compiler validates/signs LLVM-produced containers; no HLSL parser or textual IR patching |
| DXIL → `.metallib` | Metal Shader Converter adapter | Tested with converter 3.0, preserving entry-point names |
| Vulkan graphics rendering | GLFW window, offscreen PNG and MP4 | Simple graphics and original ray marcher tested with validation, deterministic time, presentation, and resizing |
| Native Metal rendering of DXIL | MetalKit window and offscreen readback using Apple's converter runtime | Tested on Apple M3 Max with API/GPU validation, pixel checks, and persistent-window tests |
| Python device control flow / SDF composition | Default AST frontend and upstream SCF | Real bounded GPU loops, branches, early exits, carried scalar/vector state, and finite-difference normals |
| Read-only frame resources | Reflected f32 root ABI, Vulkan push constants, DXIL root constants, converted Metal argument buffer | Exact time/pixel dimensions in both stages; live clock, frozen snapshots, resize and Retina drawable readback |
| Native DirectX 12 runtime rendering | Not implemented | DirectX artifacts are previewed through Metal on macOS, not a DirectX device |
| Compute entry points | IR foundation only | Typed stage, workgroup-size and invocation-ID verification; compute lowering/execution is explicitly rejected for now |
| Typed pointers / bindless handles | IR foundation only | ODS types parse and round-trip; general pointer/load-store lowering and resource heaps are not implemented yet |

LLVM IR verification is **not** DXIL validation: the native DXC validator also
checks and signs the actual containers. Real rendering tests cover Vulkan and
DXIL-to-Metal in addition to mocked command-wiring tests. Native Windows
DirectX runtime interoperability remains unverified. No path emits fake binaries
or falls back to shading-language source.
The completed ray-march stack passed **177 tests** with both GPU paths,
API validation, and real desktop windows enabled. The measured acceptance
section records reproducible workload, binary sizes, and numerical thresholds.

## Compiler architecture

The compiler lives in [native/compiler](native/compiler). Its
[TableGen definitions](native/compiler/include/Gfx) define `gfx` stage and
interpolation attributes, typed pointers, resource handles, builtin inputs,
varying inputs/outputs, and clip-position writes. Entry points are standard
`func.func` operations annotated with `gfx.stage`; they are not tied to a draw
count or the triangle example. Arithmetic and vector construction/extraction
use upstream `arith` and `vector` operations.

The default Python AST frontend sends typed JSON to `gfx-compile`, which
constructs real MLIR operations through C++ builders. Straight-line expressions
use indexed graphs; control flow uses structured operations and SSA references.
Neither path generates textual MLIR or LLVM assembly. Textual IR is an inspectable
compiler output and a native-tool input format, not a string-based lowering
implementation. The compiler also accepts version-2 structured JSON: typed
functions, blocks, operations, regions, and SSA references. It constructs
registered upstream operations directly and uses their ODS/SSA verifiers;
there is no custom `gfx` arithmetic or loop dialect.

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
contract from day one. Beyond the minimal read-only frame ABI, pointer-first/bindless
memory lowering, general shared host/device struct layouts, native host-language frontends, mesh/task
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

Resource-free manifests use **schema version 3**, with compiler-generated stage reflection,
a separate draw description and `raster_convention`. Corrected version-2
artifacts remain viewable through a legacy compatibility check. Version-1
binaries predate normalization and require recompilation into a new directory.
Existing artifact directories and PNGs are not rewritten.
Frame-resource pipelines use **schema version 4** with complete reflected
layout/bindings; earlier resource schemas are rejected rather than guessed.

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
The full ray-march acceptance also covers frame resources, device control flow,
the original scene, MP4 encoding, and live drawable readback. Install the
`video` extra or supply FFmpeg for movie tests, and provide converter runtime
headers plus `clang++` for the opted-in Metal window probe. Run window suites
serially. See the measured ray-march section below for its separate numerical
thresholds; those do not weaken the exact simple-graphics baselines.

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

See [the triangle](examples/triangle.py). Ordinary Python helpers compose device
expressions through AST interpretation, without source rewriting or shader strings.
`@vertex` receives a `u32` vertex index and returns `Vertex(position, color=...)`
or `Vertex(position, varyings={"name": value, ...})`.
`@fragment` parameters resolve against those varying names and types when traced
through `Pipeline`; a fragment shader may also take no inputs. Fragment outputs
are device values or a mapping of named render targets. The compiler links their
interfaces from native reflection; the preview currently supports one `f32x4` target.

### Native MLIR and device loops

`MLIRPipeline` accepts vertex/fragment textual MLIR or structured `.json` files
and their declared entry names,
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
Only the marked entry receives shader attributes/signature metadata. Builtins,
varying reads, and stage writes must be in its entry block, outside nested
regions. Compute values conditionally, merge them with SSA, then write the
stage interface unconditionally. Repeated identical input reads are allowed
and merge under CSE; conflicting declarations at a location are rejected.

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
Python authoring supports the same device control flow through the default AST frontend.

### Default Python AST frontend

The existing decorators and `Pipeline` API now interpret shader/helper source
instead of calling shader functions with symbolic arguments. Source must be
inspectable; unavailable source or unsupported syntax produces a file/line/column
error, never a fallback to executing the shader as host Python.

Supported device syntax includes local assignment and augmented assignment,
`if`/`else`, conditional expressions, `while`, direct `for ... in range(...)`,
`break`, `continue`, helper returns, and scalar/vector loop-carried state.
`and`, `or`, `not`, and chained comparisons short-circuit with structured
selections; `select` and Boolean bitwise operators remain eager.
`int(device_float)` and `float(device_integer)` are explicit numeric conversions.
Device floating `**` uses the portable shader-power operation.

Direct range loops become GPU loops, even with literal bounds: changing the
bound does not duplicate the body. Bounds must fit signed 32-bit integers;
the step must be a nonzero compile-time signed 32-bit integer. Loop `else`
and dynamically selected steps are unsupported. GPU `while` termination is
the author's responsibility.

Literal tuple/reversed-range iteration, list/generator comprehensions, dictionary
expansion, and constant arithmetic are compile-time metaprogramming. They preserve
existing vector/table construction; device-dependent iterables or comprehension
filters are not supported. Ordinary helpers, local helper definitions, and lambdas
are AST-interpreted and inlined. Captured locals are snapshots: rebinding a captured
local is rejected, so pass changing state as an explicit helper argument instead.
Dynamic calls, recursion, arbitrary object mutation/introspection, and exception
handling are outside this subset. SDK expression factories remain explicitly
registered builders; Python builtins and math modules are not monkeypatched.

[The capsule marcher](examples/capsule_march.py) demonstrates a Python-authored
64-step GPU loop, nested hit/miss breaks, continue, carried scalar/vector state,
an ordinary distance helper, and the reusable tetrahedral normal helper:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/capsule_march.py \
  --target metal --output build/capsule-march-metal
PATH="$PWD/.tools/bin:$PATH" .venv/bin/mlir-gpu-view \
  build/capsule-march-metal/pipeline.json --width 128 --height 128 -frames 0 --validation
```

This target-neutral example also supports `--target vulkan` and `--target directx`.
Its camera is fixed at 128×128; use the frame-aware lit sphere or gallery below
for animation and resize-aware rendering.

### Typed structured input

[The Julia example](examples/julia.py) loads [version-2 JSON](examples/ir/julia.json).
It renders Julia-set escape iterations with a vector-return helper, a nested
selection, and a GPU while loop carrying a counter, two coordinates, escape
state, and accumulated color:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/julia.py \
  --target metal --output build/julia-metal
PATH="$PWD/.tools/bin:$PATH" .venv/bin/mlir-gpu-view \
  build/julia-metal/pipeline.json --width 128 --height 128 -frames 0 --validation
```

The same source supports `--target vulkan` and `--target directx`. This example
uses a fixed 128-pixel camera and at most 32 iterations; unlike the gallery,
it intentionally does not read the available frame-resource ABI.

Shader input versions are independent of the output pipeline-manifest schema:

- Unversioned input and `schema_version: 1` retain the existing flat graph.
- `schema_version: 2` contains `functions`. Each function has a unique `name`,
  result type names in `results`, and a `body` region. Its first block's typed
  arguments define the function parameters. The unique entry has `stage`;
  compute entries additionally require `workgroup_size`. Other functions are
  private helpers; definitions may precede or follow the entry.
- Each region contains `blocks`. A block has a unique `name`, typed
  `arguments` (`name`/`type`), and ordered `operations`.
- Operations use registered MLIR names such as `arith.addf`, `func.call`,
  `scf.if`, or `scf.while`; `func.func` is represented by the function envelope,
  not a nested operation. Optional `args` reference SSA names, `results`
  declare typed SSA names, `regions` describe nested bodies, and
  `successors` reference blocks in the same region.
- SSA names must be unique within a region and cannot shadow captured names.
  All block arguments are declared before operations; operation results must
  be defined before use in block-list order. Nested regions capture visible
  enclosing values but cannot leak internal results. A region-bearing
  operation exposes only its declared results after its regions are built.
  Upstream MLIR additionally verifies dominance, call signatures, region
  yields, terminators, and loop-carried types.
- Types are `f32`, `i32`/`u32` (signless 32-bit storage), `bool`,
  `f32x2`/`f32x3`/`f32x4`, and `index` for loop induction.
- `attributes` maps registered property names to typed JSON descriptors:
  `literal` (`type` and `value`), `dense` (vector `type` and component array,
  optionally one splat), `string`, `symbol`, `i32_array`, `i64_array`,
  or enum kinds `builtin`, `interpolation`, `cmpi`, and `cmpf`.
  Enum values use upstream names, such as `ogt` for a float comparison.
  Integer literals describe representable 32-bit patterns or signed index
  values; operation predicates determine signed arithmetic semantics.
  Unknown properties, invalid types/enums, and malformed shapes fail explicitly.
- Optional `source_file` and positive `source_line` pairs on the module,
  function, block, region, or operation give source-located diagnostics;
  omitted locations inherit from the enclosing object.
- Unknown versions fail with migration guidance. `cf` constructs can be built
  and verified, but current graphics target lowering still requires structured
  single-block function bodies. Recursion, external/indirect calls, and
  multiple helper results remain explicit target limitations.

#### Structured termination contract

The input represents normalized SSA, not source-language `break`, `continue`,
or early-return syntax. Frontends must preserve statement order when producing it:

- For `break`, carry a running flag into `scf.while`, clear it at the break,
  and guard the rest of that iteration with `scf.if`. The condition combines
  the flag with loop bounds; yield state updates made before the break.
- For `continue`, merge the updated state and guard only the skipped suffix.
  A `scf.for` still advances its induction variable; a while loop must retain
  exactly the source-authored updates, including any intentionally skipped
  update.
- For early return, carry the return value and function-active flag through
  enclosing regions/loops, skip later statements, and emit one `func.return`
  at the function boundary. Graphics entries similarly merge interface values
  before their unconditional stage writes.

Use `scf.condition` for while termination and `scf.yield` for region/backedge
state. Do not place function returns or stage writes inside SCF regions.
The Julia example demonstrates an escape check that freezes coordinates while
still retaining the counter/glow updates performed in the terminating iteration.

### Read-only frame parameters

[Frame bands](examples/frame_bands.py) reads elapsed time and framebuffer-pixel
resolution through a typed, immutable parameter block, not interpolated inputs:

```python
from mlir_gpu import FrameParameters, FragmentCoord, fragment, vec4

@fragment
def diagnostic(pixel: FragmentCoord, frame: FrameParameters):
    uv = pixel / frame.resolution
    return vec4(uv, frame.time, 1)
```

`ParameterBlock(name, ((field_name, Type), ...))` also declares reusable blocks.
The current portable slice supports one matching block per pipeline, with f32
scalars and f32x2/3/4 fields, at most 128 bytes. There are no authored register,
descriptor-set, or Metal-buffer numbers. Unsupported types, duplicate fields,
oversized layouts, and mismatched stage declarations fail explicitly.
Read fields through parameters and ordinary helpers; writes, field pointers,
general heaps, and writable/array resources are not implemented.

The compiler owns layout: scalar/vector sizes are 4/8/12/16 bytes; alignments
are 4/8/16/16 bytes. Each field starts at the next aligned offset; the block
is rounded to 16 bytes. This places a scalar after a vec3 at byte 12, while a
following vec2 starts at an 8-byte boundary. Host packing is little-endian f32,
including zero-filled internal/trailing padding. The frame layout is:

| Field | Type | Offset | Size | Alignment |
|-------|------|--------|------|-----------|
| time | f32 | 0 | 4 | 4 |
| padding | - | 4 | 4 | - |
| resolution | f32x2 | 8 | 8 | 8 |

Total size/alignment: **16 bytes**. `pack_parameters(reflected_block, values)`
validates the complete reflected layout/bindings and exact supplied fields;
it returns owned immutable bytes and rejects non-finite/out-of-range f32 values.
Packing does not establish resource lifetimes or make zero framebuffer dimensions
valid; frame validation/update policies belong to the runtime adapters.

Native authoring uses TableGen `#gfx.parameter_field`, `#gfx.parameter_block`,
and `gfx.parameter`. Structured JSON adds a module-level `parameters` object
containing only `name` and typed `fields`. Backend bindings and offsets are
synthesized, not accepted from authored JSON.

Vulkan emits a PushConstant struct with explicit member offsets. DirectX uses
upstream LLVM `dx.CBuffer` handles and resource-pointer loads with an explicitly
padded packed struct; LLVM performs resource access legalization. A version-1.1
root signature with root constants is built using LLVM's typed metadata API and
embedded in DXIL. Metal Shader Converter consumes that embedded signature; its
runtime argument-buffer binding and byte layout are persisted in reflection.
Parameters are host-owned, read-only during invocation, and not addressable.

Resource pipelines use **manifest schema 4** with complete fields, byte layout,
stage visibility, and Vulkan/DirectX/Metal bindings. Resource-free pipelines
continue emitting schema 3; existing schema-2/3 previews remain supported.
Stale schemas, incomplete layouts, and mismatched bindings are rejected.

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/frame_bands.py \
  --target metal --output build/frame-bands-metal
```

Also compile with `--target vulkan` or `--target directx` into fresh directories.
Vulkan, Metal, and DXIL-to-Metal previews now bind these resources; they never
silently substitute zero resolution. Custom parameter blocks compile, but the preview supplies
only the standard `FrameParameters` profile.

### Animated previews and fixed-time snapshots

[Orbiting sphere](examples/orbiting_sphere.py) is a portable Python ray marcher
with an animated sphere center and a resize-aware camera. It uses the same
frame-parameter ABI as frame bands. Vulkan live previews supply monotonic elapsed
seconds since renderer initialization and the actual swapchain framebuffer-pixel
dimensions, not logical window points. Minimized/zero-sized windows pause drawing;
restoration and resize rebuild the swapchain and use its new extent.

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/orbiting_sphere.py \
  --target vulkan --output build/orbiting-sphere-vulkan
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/mlir-gpu-view \
  build/orbiting-sphere-vulkan/pipeline.json --width 640 --height 480 -frames 0 --validation
```

For a deterministic PNG, use a fresh output file:

```sh
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/mlir-gpu-view \
  build/orbiting-sphere-vulkan/pipeline.json --width 640 --height 480 \
  --time 1.25 --output build/orbiting-sphere-1.25.png --validation
```

Offscreen time defaults to **0 seconds**. `--time` (or Python `time=...`) supplies
fixed, finite, nonnegative f32 seconds. In live mode it freezes shader time while
still updating framebuffer dimensions on resize; omitting it advances the clock.
The Python APIs are `render_offscreen(program, ..., time=1.25)` and
`show_window(program, ..., time=None)`. `render_metal_offscreen` and
`show_metal_window` accept the same time policy, after their existing `toolchain`
argument. Metal uses actual attachment texture dimensions, including Retina
drawable pixels, and a monotonic live clock. Reflected offsets and stage visibility
control copied encoder bytes at the converter SDK's argument-buffer bind point.
Each submission completes before the next frame; no buffer is reused in flight.

[Pulse rings](examples/pulse_rings.py) demonstrates the Metal frame adapter:

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/pulse_rings.py \
  --target metal --output build/pulse-rings-metal
PATH="$PWD/.tools/bin:$PATH" .venv/bin/mlir-gpu-view \
  build/pulse-rings-metal/pipeline.json --time 0.4 --output build/pulse-rings.png --validation
```

Omit `--time`/`--output` for animation; compile with `--target directx` to
exercise launch-time DXIL conversion instead of loading preconverted libraries.

The reflected push-constant range and stage visibility determine the Vulkan
pipeline layout. Packing uses reflected offsets and zero padding. Each acquired
image's command buffer is reset/re-recorded with fresh frame bytes, after the
previous submission has completed. Push-constant data is copied into the command
buffer; the single-frame-in-flight queue-idle policy prevents resetting commands
still used by the GPU. Missing/mismatched layouts, unbound custom blocks, invalid
time, and device-limit failures are explicit errors. No shader recompilation,
descriptor heap, or writable memory is needed for this read-only ABI.

### Export animation to MP4

MP4 export renders actual offscreen GPU frames at fixed shader times on all three preview paths.
It does not record the desktop, recompile shaders, or create a PNG sequence.
Install the optional encoder dependency in the project environment:

```sh
.venv/bin/python -m pip install -e '.[video]'
```

An existing FFmpeg executable on `PATH` takes precedence; it must support
`libx264`. Otherwise the video extra supplies a platform-specific FFmpeg binary.
Missing encoders and encoding errors are reported explicitly.

After compiling the orbiting sphere as above:

```sh
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/mlir-gpu-view \
  build/orbiting-sphere-vulkan/pipeline.json \
  --video build/orbiting-sphere.mp4 --duration 6 --fps 30 \
  --width 640 --height 480 --validation
```

`--video`, PNG `--output`, and live `--frames` are mutually exclusive.
Video duration defaults to 4 seconds and FPS to 30; `--duration`/`--fps` are
video-only controls. `--time` selects the shader's starting time (default 0).
Frame `i` samples `start_time + i / fps`, independently of GPU/encoding speed.
The frame count is `ceil(duration * fps)`, so playback duration rounds upward
to a whole frame; playback timestamps start at zero.

The Python API is
`export_mp4(program, path, width=640, height=480, duration=6, fps=30, start_time=0, validation=True)`
from `mlir_gpu.viewer`; it returns the exported frame count. The Vulkan device,
pipeline, attachments, command buffer, fence, and readback buffer are reused
across frames, with completion before each reset/readback.

Output is H.264/yuv420p MP4 with fast-start metadata for ordinary video players.
It is lossy and has no alpha channel; decoded pixel values can differ from PNG
readback. Dimensions must be even, within the existing 1–4096 preview limits,
FPS an integer from 1–240, and duration positive. Times must be finite,
nonnegative f32 seconds. Output paths must end in `.mp4`, have an existing parent,
and not already exist. The final file is published only after encoding succeeds;
failures clean up temporary output and stop the encoder without overwriting files.

Metal export converts DXIL libraries once per sequence (preconverted Metal
needs no converter). It launches a native offscreen renderer per frame;
unlike Vulkan, Metal device/pipeline reuse across a movie is not yet implemented.
For example, export `build/pulse-rings-metal/pipeline.json` with
`--video build/pulse-rings.mp4 --duration 3 --fps 24 --width 320 --height 240`.
Existing PNG and persistent-window behavior is unchanged.

### Original ray-marched beacon gallery

[Beacon gallery](examples/beacon_gallery.py) combines repeated sphere/box
geometry inside a room, a moving camera, a **72-step GPU loop**, distance/hit
termination, loop-carried glow, four-sample SDF normals, lighting, analytically
filtered procedural lines, fog, and tone mapping. It is an original scene,
not a translation of an attributed third-party shader. Geometry and effects are
ordinary Python helpers built from the shared device library.

```sh
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" PATH="$PWD/.tools/bin:$PATH" \
  .venv/bin/python -m mlir_gpu examples/beacon_gallery.py \
  --target vulkan --output build/beacon-gallery-vulkan
DYLD_LIBRARY_PATH=/usr/local/lib .venv/bin/mlir-gpu-view \
  build/beacon-gallery-vulkan/pipeline.json --width 320 --height 240 \
  --video build/beacon-gallery.mp4 --duration 4 --fps 24 --validation
```

Compile the same source with `--target directx` or `--target metal` into fresh
directories. All previews support fixed-time PNG/MP4 and live animation.
Start at **320x240** for interactive use. `MAX_STEPS` and `MAX_DISTANCE` are
compile-time workload controls; exhausting either bound is a miss, not a surface.
Increasing the step cap changes the loop bound, not the number of cloned bodies.
Normals are evaluated only on a confirmed hit; glow integrates along the ray.
Use `--time 0.25 --output build/gallery.png` for a deterministic snapshot,
or omit output/time for a persistent, resize-aware window.

### Device Python authoring walkthrough

[Lit sphere](examples/lit_sphere.py) is a compact starting point before the
gallery: a fixed sphere with animated directional lighting, SDF normals, rim
lighting, fog, and display encoding. The same program runs on every target.

1. `@fragment` declares an entry. `pixel: FragmentCoord` is a two-component
   raster builtin with top-left origin and pixel centers. `frame: FrameParameters`
   is host-owned, immutable invocation data, **not** an interpolated varying.
2. Normalize coordinates by `frame.resolution.y` to retain aspect ratio.
   Negating camera Y makes scene-space up agree with top-left framebuffer Y.
   `fullscreen_vertex` supplies the reusable three-vertex full-screen draw.
3. Define scalar distance and Boolean hit state before the loop. Even
   `range(64)` is a GPU loop. The hit check happens before advancing; the
   distance check happens afterward. A cap/distance miss keeps the background.
   Changing a bound requires compilation; advancing time does not.
4. `sdf.sphere` and `sdf.normal` compose supported device operations. The
   normal's lambda is a deliberately inlined helper, not a GPU function pointer.
   The four normal samples are fixed work, not an unrolled march.
5. Use `shader_math` for device trigonometry, normalization, dot products,
   interpolation, and power. Reconstruct vectors with `with_components` rather
   than mutating swizzles. Return one `vec4` fragment color.

| Value/work | When it runs |
|------------|--------------|
| Module constants and pipeline construction | Host Python before compilation |
| Host-computed constants such as `math.pi` | Captured by compilation; not per-frame device reads |
| SDK arithmetic/math on pixels or frame values | Native shader instructions |
| Direct range loops and device-dependent branches | GPU control flow |
| Literal tuple iteration for four normal samples | Compile-time helper expansion |
| Seconds/resolution supplied by the preview | Per-frame resource update; no shader recompilation |

Do not use Python `math.sin` on a device value, mutate captured state, or place
arbitrary Python object operations in a shader. For changing helper state, pass
it as an explicit argument. Unsupported source/syntax/types report the source
file, line, and column; source inspection must work, so put shaders in ordinary
files rather than dynamically generated/unavailable source. There is no host
execution or old-tracer fallback.

With the pinned compiler, native helper tools, validator, converter prerequisites,
and optional viewer/video extras installed:

```sh
export LLVM_BIN="$PWD/.toolchains/llvm-build/bin"
export PATH="$PWD/.tools/bin:$LLVM_BIN:$PATH"
export DYLD_LIBRARY_PATH=/usr/local/lib  # macOS Vulkan loader

# Use fresh directories/files; run from the repository root.
for target in vulkan directx metal; do
  .venv/bin/python -m mlir_gpu examples/beacon_gallery.py \
    --target "$target" --output "build/gallery-guide-$target"
  .venv/bin/mlir-gpu-view "build/gallery-guide-$target/pipeline.json" \
    --time 0.25 --width 320 --height 240 \
    --output "build/gallery-guide-$target.png" --validation
done

# Preconverted Metal needs no converter during preview.
.venv/bin/mlir-gpu-view build/gallery-guide-metal/pipeline.json \
  --width 320 --height 240 --validation

# Deterministic animation, independent of export speed.
.venv/bin/mlir-gpu-view build/gallery-guide-vulkan/pipeline.json \
  --video build/gallery-guide.mp4 --duration 4 --fps 24 \
  --width 320 --height 240 --validation
```

For the tutorial, substitute `examples/lit_sphere.py` and fresh output names.
For a bounded smoke preview add `--frames 4`; persistent windows otherwise stay
open until closed or Esc. DXIL preview on macOS performs **launch-time conversion
and Metal rendering**, not DirectX device execution. `--target metal` performs
conversion while compiling and previews its existing metallibs directly.

Troubleshooting: install the documented Vulkan loader/layers for Vulkan errors,
rebuild/install the native Metal viewer after adapter changes, check converter
runtime tier-2 support for Metal pipeline errors, and use fresh artifact paths.
Schema-4 parameter reflection must match compiler-owned fields/bindings; do not
edit byte offsets or invent runtime slots. A custom block requires host values
not provided by this preview. Non-finite time, zero-sized offscreen output, stale
resource schemas, and unsupported capabilities fail explicitly. Live minimized
Vulkan windows pause and resume; Metal draws only when a drawable/pass exists.
Reduce resolution before increasing `MAX_STEPS`; a larger cap improves grazing
hits at a cost, but does not add geometry or remove numerical discontinuities.

### Measured ray-march acceptance

[March budget](examples/march_budget.py) visualizes a per-pixel device loop
bound: four iterations on the left and 64 on the right. Red encodes iterations
used, green a confirmed hit, and bright blue a distance-limit miss; dark blue
indicates budget exhaustion. It animates using the standard frame block.

The focused suite is `tests/test_ray_march_acceptance.py`. On Apple M3 Max,
with the pinned LLVM main build and Vulkan/Metal API validation enabled:

| Gallery target | Compile wall time (warm tools) | Fragment binary |
|----------------|--------------------------------|-----------------|
| Vulkan | ~0.09 s | 19,508 bytes |
| DirectX | ~0.17 s | 12,572 bytes |
| Converted Metal | ~0.22 s | 11,932 bytes |

Changing the gallery cap from 72 to 288 leaves the native MLIR operation counts
and SPIR-V/DXIL byte sizes unchanged. SPIR-V retains a loop; LLVM retains
loop-carried scalar/vector phi state. Compile timings are observations, not
portable performance promises; the suite prints fresh timings and sizes.
The documented 320x240 workload measured approximately **31 ms Vulkan,
164 ms DXIL-to-Metal, and 109 ms preconverted Metal** for a single offscreen
call including setup/readback. These are not GPU-only timings or live frame
rates; launch-time conversion/process creation accounts for part of the Metal
cost. Four-frame live runs were also verified on all three paths.

At **64x64, 96x54, 97x73**, and times **0, 0.25, 1.75**, launch-converted
DXIL and preconverted Metal are byte-identical. Measured Vulkan/Metal RGB
differences peak at **1/255**, with mean below **0.0002/255** and zero 99th
percentile pixel error. Regression limits allow mean channel error <=1/255,
99th percentile maximum-channel error <=4/255, and at most 0.2% of pixels
above 16/255 to distinguish rare hit-edge discontinuities from broad drift.
Alpha remains exact; unchanged triangle/quad coordinate/coverage tests retain
their existing stricter contracts.

The budget shader checks **every pixel** against an independent double-precision
CPU sphere-tracing oracle at square, rectangular, and odd dimensions. Hit,
distance-miss, budget-miss, and alpha classifications are exact. Iteration-color
tolerance is 1/255, or 5/255 (one extra iteration plus quantization) only where
the oracle comes within 1e-5 world units of the 0.002 hit threshold. No pixels
are excluded. CPU code is a test oracle, never a preview fallback.

A test-only Cocoa/Metal probe reuses the production encoder and reads actual
live drawable pixels before presentation. It verifies advancing time, frozen
time, resize, and framebuffer-pixel rather than window-point dimensions on both
Metal paths. It requires `clang++`, a desktop session, and the converter runtime
headers; set `METAL_CONVERTER_RUNTIME_INCLUDE_DIR` if they are not in
`/usr/local/include`.

```sh
PATH="$PWD/.tools/bin:$PATH" DYLD_LIBRARY_PATH=/usr/local/lib \
LLVM_BIN="$PWD/.toolchains/llvm-build/bin" \
MLIR_GPU_TEST_VULKAN=1 MLIR_GPU_TEST_METAL=1 \
MLIR_GPU_TEST_VALIDATION=1 MLIR_GPU_TEST_WINDOW=1 \
  .venv/bin/python -m unittest discover -s tests -p test_ray_march_acceptance.py -v
```

GPU/window tests are explicitly opt-in. Opted-in missing tools, runtime errors,
and invalid resource layouts fail; they do not select a substitute renderer.
Run desktop suites serially, not alongside another window/video workload.

### Arithmetic and vectors

Device values support `+`, `-`, `*`, floating-point `/`, unary negation, `abs`,
scalar comparisons, and eager Boolean `&`, `|`, `^`, `~`. The AST frontend lowers
Python branches, short-circuit expressions, and loops to device control flow.
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

Numeric literals are coerced to `f32` when combined with a device float or used
as vector components. Device input values cannot leak between stages. `select`
is an expression: both arms are built, not lazy control flow.

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
second example. The AST frontend deliberately inlines these small pure helpers,
including four fixed normal samples. Direct range/while loops remain GPU loops;
the capsule marcher above uses these helpers inside device control flow.
This is not a compiler for arbitrary Python, dynamic GPU function pointers,
or recursion.

## Design decisions

Inspired by [Death to Shading Languages](https://xol.io/blah/death-to-shading-languages/):

- **Embed rather than invent a language.** Reuse Python's functions, modules,
  tooling, and host-side metaprogramming. Do not copy GLSL's source-level model.
- **Keep one program and one interface contract.** Stage interfaces live in the
  Python API and artifact manifest, not independently maintained shader files.
  General shared host/device layouts and arbitrary buffer arguments are future work
  beyond the implemented reflected f32 frame block;
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
