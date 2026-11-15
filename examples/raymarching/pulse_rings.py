from mlir_gpu import FrameParameters, FragmentCoord, Pipeline, fragment, fullscreen_vertex, shader_math as sm, vec3, vec4


@fragment
def pulse_rings(pixel: FragmentCoord, frame: FrameParameters):
    uv = (pixel - frame.resolution * 0.5) / frame.resolution.y
    phase = sm.fract(sm.length(uv) * 8.0 - frame.time)
    footprint = 8.0 / frame.resolution.y
    ring = 1.0 - sm.smoothstep(0.04 - footprint, 0.04 + footprint, abs(phase - 0.5))
    color = sm.mix(vec3(0.02, 0.04, 0.08), vec3(0.1, 0.8, 1.0), ring)
    return vec4(color, 1)


pipeline = Pipeline(fullscreen_vertex, pulse_rings)
