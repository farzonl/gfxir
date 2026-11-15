from mlir_gpu import FragmentCoord, Pipeline, fragment, fullscreen_vertex, maximum, sdf, select, shader_math as sm, vec3, vec4


@fragment
def surface_normals_fragment(pixel: FragmentCoord):
    cell = sdf.repeat(pixel / 64, 2.0)
    radius_squared = sm.dot(cell, cell)
    z = sm.sqrt(maximum(0.64 - radius_squared, 0.0))
    point = vec3(cell, z)
    normal = sdf.normal(lambda p: sdf.sphere(p, 0.8), point)
    color = normal * 0.5 + 0.5
    return vec4(select(radius_squared <= 0.64, color, vec3(0.02, 0.04, 0.08)), 1)


pipeline = Pipeline(fullscreen_vertex, surface_normals_fragment)
