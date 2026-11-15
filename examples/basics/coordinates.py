from mlir_gpu import FragmentCoord, Pipeline, fragment, fullscreen_vertex, shader_math as sm, vec4


@fragment
def coordinates_fragment(coord: FragmentCoord):
    cell = coord / 32 - sm.floor(coord / 32)
    band = coord.y / 128 - sm.floor(coord.y / 128)
    return vec4(cell, band, 1)


pipeline = Pipeline(fullscreen_vertex, coordinates_fragment)
