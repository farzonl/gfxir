from .frontend import Vertex, select, vec2, vec4, vertex


@vertex
def fullscreen_vertex(index):
    """Cover the framebuffer using one oversized triangle and no buffers."""
    return Vertex(vec4(select(index == 1, 3.0, -1.0),
                       select(index == 2, 3.0, -1.0), 0, 1))


@vertex
def fullscreen_uv_vertex(index):
    """Fullscreen triangle with top-left-origin UVs for sampled post passes."""
    x = select(index == 1, 3.0, -1.0)
    y = select(index == 2, 3.0, -1.0)
    return Vertex(vec4(x, y, 0, 1), varyings={"uv": vec2((x + 1.0) * 0.5, (1.0 - y) * 0.5)})
