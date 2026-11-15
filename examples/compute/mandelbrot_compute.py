"""2D Mandelbrot escape-time and RGBA8 image compute shaders authored in Python."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct

from mlir_gpu import (
    Buffer,
    ComputePipeline,
    Dispatch,
    GlobalInvocationId,
    ParameterBlock,
    RenderedImage,
    Target,
    Type,
    Value,
    compile_pipeline,
    compute,
    render_compute_rgba,
    shader_math as sm,
    u32,
    write_png,
)

DEFAULT_MANDELBROT_PARAMS: dict[str, int | float] = {
    "width": 256,
    "height": 256,
    "max_iterations": 100,
    "xmin": -2.0,
    "xmax": 1.0,
    "ymin": -1.5,
    "ymax": 1.5,
    "pitch": 256,
}

MandelbrotParams = ParameterBlock(
    "mandelbrot_params",
    (
        ("width", Type.U32),
        ("height", Type.U32),
        ("max_iterations", Type.U32),
        ("xmin", Type.F32),
        ("xmax", Type.F32),
        ("ymin", Type.F32),
        ("ymax", Type.F32),
        ("pitch", Type.U32),
    ),
)


def _pack_rgba8(r: Value, g: Value, b: Value) -> Value:
    ru = u32(int(sm.clamp(r * 255.0 + 0.5, 0.0, 255.0)))
    gu = u32(int(sm.clamp(g * 255.0 + 0.5, 0.0, 255.0)))
    bu = u32(int(sm.clamp(b * 255.0 + 0.5, 0.0, 255.0)))
    return ru + gu * u32(256) + bu * u32(65536) + u32(255) * u32(16777216)


@compute(workgroup_size=(16, 16, 1))
def mandelbrot_kernel(
    gid: GlobalInvocationId,
    output_buffer: Buffer[Type.U32, "write_only"],
    params: MandelbrotParams,
):
    px = gid.x
    py = gid.y
    if (px < params.width) and (py < params.height):
        u = (float(px) + 0.5) / float(params.width)
        v = (float(py) + 0.5) / float(params.height)
        cr = params.xmin + u * (params.xmax - params.xmin)
        ci = params.ymin + v * (params.ymax - params.ymin)

        k = u32(0)
        x = 0.0
        y = 0.0
        while (k < params.max_iterations) and (x * x + y * y <= 4.0):
            next_x = x * x - y * y + cr
            next_y = 2.0 * x * y + ci
            x = next_x
            y = next_y
            k = k + 1

        output_buffer[py * params.pitch + px] = k


@compute(workgroup_size=(16, 16, 1))
def mandelbrot_rgba_kernel(
    gid: GlobalInvocationId,
    output_buffer: Buffer[Type.U32, "write_only"],
    params: MandelbrotParams,
):
    px = gid.x
    py = gid.y
    if (px < params.width) and (py < params.height):
        u = (float(px) + 0.5) / float(params.width)
        v = (float(py) + 0.5) / float(params.height)
        cr = params.xmin + u * (params.xmax - params.xmin)
        ci = params.ymin + v * (params.ymax - params.ymin)

        k = u32(0)
        x = 0.0
        y = 0.0
        while (k < params.max_iterations) and (x * x + y * y <= 4.0):
            next_x = x * x - y * y + cr
            next_y = 2.0 * x * y + ci
            x = next_x
            y = next_y
            k = k + 1

        if k == params.max_iterations:
            pixel = u32(255) * u32(16777216)
        else:
            t = float(k) / float(params.max_iterations)
            inv_t = 1.0 - t
            r = 9.0 * inv_t * t * t * t
            g = 15.0 * inv_t * inv_t * t * t
            b = 8.5 * inv_t * inv_t * inv_t * t
            pixel = _pack_rgba8(r, g, b)

        output_buffer[py * params.pitch + px] = pixel


pipeline = ComputePipeline(mandelbrot_kernel, dispatch=Dispatch(grid=(16, 16, 1)))
rgba_pipeline = ComputePipeline(mandelbrot_rgba_kernel, dispatch=Dispatch(grid=(16, 16, 1)))


def _f32(val: float) -> float:
    return struct.unpack("<f", struct.pack("<f", val))[0]


def cpu_mandelbrot_pixel(
    px: int,
    py: int,
    *,
    width: int = 256,
    height: int = 256,
    max_iterations: int = 100,
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
) -> int:
    """Evaluate single-precision Mandelbrot escape count for pixel (px, py)."""
    u = _f32(_f32(float(px) + 0.5) / _f32(float(width)))
    v = _f32(_f32(float(py) + 0.5) / _f32(float(height)))
    cr = _f32(_f32(xmin) + _f32(u * _f32(_f32(xmax) - _f32(xmin))))
    ci = _f32(_f32(ymin) + _f32(v * _f32(_f32(ymax) - _f32(ymin))))

    x = 0.0
    y = 0.0
    k = 0
    while k < max_iterations and _f32(_f32(x * x) + _f32(y * y)) <= 4.0:
        next_x = _f32(_f32(_f32(x * x) - _f32(y * y)) + cr)
        next_y = _f32(_f32(2.0 * _f32(x * y)) + ci)
        x = next_x
        y = next_y
        k += 1
    return k


def cpu_mandelbrot_rgba(
    *,
    width: int = 256,
    height: int = 256,
    max_iterations: int = 100,
    xmin: float = -2.0,
    xmax: float = 1.0,
    ymin: float = -1.5,
    ymax: float = 1.5,
) -> RenderedImage:
    """Render reference RGBA8 Mandelbrot image on the CPU matching mandelbrot_rgba_kernel."""
    rgba = bytearray(width * height * 4)
    offset = 0
    inv_max = _f32(1.0 / float(max_iterations))
    for py in range(height):
        for px in range(width):
            k = cpu_mandelbrot_pixel(
                px,
                py,
                width=width,
                height=height,
                max_iterations=max_iterations,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )
            if k == max_iterations:
                rgba[offset] = 0
                rgba[offset + 1] = 0
                rgba[offset + 2] = 0
                rgba[offset + 3] = 255
            else:
                t = _f32(float(k) * inv_max)
                inv_t = _f32(1.0 - t)
                t2 = _f32(t * t)
                t3 = _f32(t2 * t)
                inv_t2 = _f32(inv_t * inv_t)
                inv_t3 = _f32(inv_t2 * inv_t)
                r = max(0.0, min(255.0, _f32(_f32(9.0 * _f32(inv_t * t3)) * 255.0 + 0.5)))
                g = max(0.0, min(255.0, _f32(_f32(15.0 * _f32(inv_t2 * t2)) * 255.0 + 0.5)))
                b = max(0.0, min(255.0, _f32(_f32(8.5 * _f32(inv_t3 * t)) * 255.0 + 0.5)))
                rgba[offset] = int(r)
                rgba[offset + 1] = int(g)
                rgba[offset + 2] = int(b)
                rgba[offset + 3] = 255
            offset += 4
    return RenderedImage(width=width, height=height, rgba=bytes(rgba), device="cpu-reference")


def main() -> int:
    import tempfile

    parser = argparse.ArgumentParser(description="Render Mandelbrot compute shader output to PNG.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--target", choices=[t.value for t in Target], default="vulkan")
    parser.add_argument("--provider", default=None, help="Optional runtime provider (warp, lavapipe, swiftshader, hardware)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-iterations", type=int, default=100)
    args = parser.parse_args()

    params = {
        **DEFAULT_MANDELBROT_PARAMS,
        "width": args.width,
        "height": args.height,
        "pitch": args.width,
        "max_iterations": args.max_iterations,
    }
    with tempfile.TemporaryDirectory() as td:
        manifest = compile_pipeline(rgba_pipeline, Target(args.target), Path(td) / "compiled")
        image = render_compute_rgba(
            manifest,
            args.width,
            args.height,
            parameters=params,
            provider=args.provider,
        )
    args.output.unlink(missing_ok=True)
    write_png(args.output, image)
    print(f"Rendered Mandelbrot {image.width}x{image.height} on {image.device}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
