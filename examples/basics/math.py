from mlir_gpu import Pipeline, Vertex, fragment, maximum, select, shader_math as sm, vec3, vec4, vertex


@vertex
def math_vertex(index):
    position = vec4(select(index == 1, 3.0, -1.0), select(index == 2, 3.0, -1.0), 0, 1)
    return Vertex(position, varyings={"uv": (position.xy + 1) / 2})


@fragment
def math_fragment(uv):
    point = uv * 2 - 1
    radius = sm.sqrt(point.x * point.x + point.y * point.y)
    rings = sm.sin(radius * 18) * 0.5 + 0.5
    stripes = sm.cos(point.x * 12) * 0.5 + 0.5
    glow = sm.exp(-radius * 3)
    cells = uv * 8 - sm.floor(uv * 8)
    color = vec3(sm.pow(maximum(rings, 0.0), 1.5) * glow, stripes * glow, cells.x * 0.35)
    return vec4(color, 1)


pipeline = Pipeline(math_vertex, math_fragment)
