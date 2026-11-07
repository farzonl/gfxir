from .frontend import Vertex, select, vec4, vertex


@vertex
def fullscreen_vertex(index):
    """Cover the framebuffer using one oversized triangle and no buffers."""
    return Vertex(vec4(select(index == 1, 3.0, -1.0),
                       select(index == 2, 3.0, -1.0), 0, 1))
