from mlir_gpu import Pipeline, Value, Vertex, fragment, select, vec4, vertex


def corner(index: Value, first: float, second: float, third: float) -> Value:
    return select(index == 0, first, select(index == 1, second, third))


@vertex
def triangle_vertex(index: Value) -> Vertex:
    x = corner(index, -0.6, 0.6, 0.0)
    y = corner(index, -0.6, -0.6, 0.6)
    return Vertex(
        position=vec4(x, y, 0.0, 1.0),
        color=vec4(
            corner(index, 1.0, 0.0, 0.0),
            corner(index, 0.0, 1.0, 0.0),
            corner(index, 0.0, 0.0, 1.0),
            1.0,
        ),
    )


@fragment
def triangle_fragment(color: Value) -> Value:
    return color


pipeline = Pipeline(triangle_vertex, triangle_fragment)
