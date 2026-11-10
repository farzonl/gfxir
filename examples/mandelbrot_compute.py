"""2D Mandelbrot escape-time compute shader authored in Python."""

from mlir_gpu import (
    Buffer,
    ComputePipeline,
    Dispatch,
    GlobalInvocationId,
    ParameterBlock,
    Type,
    compute,
    u32,
)

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


pipeline = ComputePipeline(mandelbrot_kernel, dispatch=Dispatch(grid=(16, 16, 1)))
