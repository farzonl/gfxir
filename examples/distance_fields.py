from mlir_gpu import FragmentCoord, Pipeline, fragment, fullscreen_vertex, sdf, shader_math as sm, vec2, vec3, vec4


@fragment
def distance_fields_fragment(pixel: FragmentCoord):
    point = pixel / 48
    cell = sdf.repeat(point, 2.0)
    rotated = sm.rotate2(cell, 0.35)
    distance = sdf.box2(rotated, vec2(0.55, 0.35))
    mask = 1.0 - sm.smoothstep(-0.025, 0.025, distance)
    rings = sm.exp(-abs(distance) * 5)
    color = sm.mix(vec3(0.02, 0.04, 0.08), vec3(0.0, 0.7, 1.0), mask)
    color = color + vec3(1.0, 0.25, 0.05) * rings * 0.35
    return vec4(sm.clamp(color, 0.0, 1.0), 1)


pipeline = Pipeline(fullscreen_vertex, distance_fields_fragment)
