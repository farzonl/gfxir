from mlir_gpu import Draw, Pipeline, Vertex, fragment, select, vec2, vec4, vertex


def choose(index, values):
    result = float(values[-1])
    for candidate in reversed(range(len(values) - 1)):
        result = select(index == candidate, float(values[candidate]), result)
    return result


@vertex
def quad_vertex(index):
    u = choose(index, (0, 1, 0, 0, 1, 1))
    v = choose(index, (0, 0, 1, 1, 0, 1))
    return Vertex(
        position=vec4(u * 1.6 - 0.8, v * 1.6 - 0.8, 0, 1),
        varyings={"uv": vec2(u, v)},
    )


@fragment
def quad_fragment(uv):
    return vec4(uv[0], uv[1], uv[0] * uv[1], 1)


pipeline = Pipeline(quad_vertex, quad_fragment, draw=Draw(vertex_count=6))
