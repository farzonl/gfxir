from mlir_gpu import Pipeline, Vertex, fragment, i32, maximum, minimum, select, vec3, vec4, vertex


@vertex
def arithmetic_vertex(index):
    x = select(index == 1, 3.0, -1.0)
    y = select(index == 2, 3.0, -1.0)
    position = vec4(x, y, 0, 1)
    return Vertex(position, varyings={
        "uv": (position.xy + 1) / 2,
        "valid": select(i32(index) >= 0, 1.0, 0.0),
    })


@fragment
def arithmetic_fragment(uv, valid):
    point = uv * 2 - 1
    distance = maximum(abs(point.x), abs(point.y)) - 0.45
    outline = (distance >= -0.025) & (distance <= 0.025)
    color = vec3(uv.yx, 0.25).with_components(z=abs(point.x) / 2)
    color = minimum(color + select(outline, 0.6, 0.0), 1.0) * valid
    return vec4(color, 1)


pipeline = Pipeline(arithmetic_vertex, arithmetic_fragment)
