import runpy

from mlir_gpu import (
    FrameParameters, ParameterBlock, Pipeline, Type, Vertex,
    fragment, fullscreen_vertex, vec3, vec4, vertex,
)
from test_support import ROOT


Material = ParameterBlock("material", (
    ("normal", Type.VEC3), ("roughness", Type.F32),
    ("pair", Type.VEC2), ("weight", Type.F32),
))


@fragment
def material_fragment(material: Material):
    return vec4(material.normal * material.roughness + vec3(material.pair, material.weight), 1)


@vertex
def frame_vertex(index, frame: FrameParameters):
    return Vertex(fullscreen_vertex.function(index).position.with_components(z=frame.time * 0.01))


OtherFrame = ParameterBlock("frame", (("time", Type.VEC2), ("resolution", Type.VEC2)))


@fragment
def incompatible(frame: OtherFrame):
    return vec4(frame.time, frame.resolution)


Limit = ParameterBlock("limit", tuple((f"v{i}", Type.VEC4) for i in range(8)))
Limit256 = ParameterBlock("limit256", tuple((f"v{i}", Type.VEC4) for i in range(16)))


@fragment
def limit(parameters: Limit):
    return parameters.v0


@fragment
def limit256_fragment(parameters: Limit256):
    return parameters.v0


def pipeline(name):
    if name == "frame":
        return runpy.run_path(str(ROOT / "examples/frame_bands.py"))["pipeline"]
    if name == "material":
        return Pipeline(fullscreen_vertex, material_fragment)
    if name == "limit":
        return Pipeline(fullscreen_vertex, limit)
    if name == "limit256":
        return Pipeline(fullscreen_vertex, limit256_fragment)
    if name == "visibility":
        return Pipeline(frame_vertex, pipeline("frame").fragment)
    if name == "mismatch":
        return Pipeline(frame_vertex, incompatible)
    raise ValueError(name)
