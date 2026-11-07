from mlir_gpu import FrameParameters, FragmentCoord, Pipeline, fragment, fullscreen_vertex, shader_math as sm, vec4


@fragment
def frame_bands(pixel: FragmentCoord, frame: FrameParameters):
    uv = pixel / frame.resolution
    wave = sm.sin((uv.x + uv.y) * 12.0 - frame.time * 2.0) * 0.5 + 0.5
    return vec4(uv.x, uv.y, wave, 1)


pipeline = Pipeline(fullscreen_vertex, frame_bands)
