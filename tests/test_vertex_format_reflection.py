import json
import os
import tempfile
import unittest
from pathlib import Path

from mlir_gpu.compiler import CompileError, Target, Toolchain, compile_pipeline
from mlir_gpu.frontend import Pipeline, Value, Vertex, fragment, vec4, vertex
from mlir_gpu.parameters import struct, Type
from mlir_gpu.viewer import _VK_FORMAT_NAMES, load_program

LLVM_BIN = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
GPU_TOOLS_BIN = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
TOOLS = Toolchain(LLVM_BIN, GPU_TOOLS_BIN)


@struct(layout="vertex")
class PositionOnlyVertex:
    position: Type.VEC3


@struct(layout="vertex")
class PositionUVVertex:
    position: Type.VEC3
    uv: Type.VEC2


@struct(layout="vertex")
class StandardPbrVertex:
    position: Type.VEC3
    normal: Type.VEC3
    uv: Type.VEC2


@struct(layout="vertex")
class FullGeometryVertex:
    position: Type.VEC3
    normal: Type.VEC3
    tangent: Type.VEC4
    uv: Type.VEC2


@struct(layout="vertex")
class TwoDColoredVertex:
    position: Type.VEC2
    color: Type.VEC4


@struct(layout="vertex")
class ScaledVertex:
    position: Type.VEC3
    scale: Type.F32


class TestVertexFormatReflection(unittest.TestCase):
    def test_diverse_vertex_configurations_layout_reflection(self):
        cases = [
            (PositionOnlyVertex, 12, [
                {"name": "position", "location": 0, "format": "f32x3", "offset": 0, "size": 12, "semantic_name": "POSITION", "semantic_index": 0},
            ]),
            (PositionUVVertex, 20, [
                {"name": "position", "location": 0, "format": "f32x3", "offset": 0, "size": 12, "semantic_name": "POSITION", "semantic_index": 0},
                {"name": "uv", "location": 1, "format": "f32x2", "offset": 12, "size": 8, "semantic_name": "TEXCOORD", "semantic_index": 0},
            ]),
            (StandardPbrVertex, 32, [
                {"name": "position", "location": 0, "format": "f32x3", "offset": 0, "size": 12, "semantic_name": "POSITION", "semantic_index": 0},
                {"name": "normal", "location": 1, "format": "f32x3", "offset": 12, "size": 12, "semantic_name": "NORMAL", "semantic_index": 0},
                {"name": "uv", "location": 2, "format": "f32x2", "offset": 24, "size": 8, "semantic_name": "TEXCOORD", "semantic_index": 0},
            ]),
            (FullGeometryVertex, 48, [
                {"name": "position", "location": 0, "format": "f32x3", "offset": 0, "size": 12, "semantic_name": "POSITION", "semantic_index": 0},
                {"name": "normal", "location": 1, "format": "f32x3", "offset": 12, "size": 12, "semantic_name": "NORMAL", "semantic_index": 0},
                {"name": "tangent", "location": 2, "format": "f32x4", "offset": 24, "size": 16, "semantic_name": "TANGENT", "semantic_index": 0},
                {"name": "uv", "location": 3, "format": "f32x2", "offset": 40, "size": 8, "semantic_name": "TEXCOORD", "semantic_index": 0},
            ]),
            (TwoDColoredVertex, 24, [
                {"name": "position", "location": 0, "format": "f32x2", "offset": 0, "size": 8, "semantic_name": "POSITION", "semantic_index": 0},
                {"name": "color", "location": 1, "format": "f32x4", "offset": 8, "size": 16, "semantic_name": "COLOR", "semantic_index": 0},
            ]),
            (ScaledVertex, 16, [
                {"name": "position", "location": 0, "format": "f32x3", "offset": 0, "size": 12, "semantic_name": "POSITION", "semantic_index": 0},
                {"name": "scale", "location": 1, "format": "f32", "offset": 12, "size": 4, "semantic_name": "SCALE", "semantic_index": 0},
            ]),
        ]

        for struct_cls, expected_stride, expected_attrs in cases:
            with self.subTest(cls=struct_cls.__name__):
                self.assertEqual(struct_cls.__struct_size__, expected_stride)
                layout = struct_cls.as_input_layout()
                self.assertEqual(len(layout["bindings"]), 1)
                binding = layout["bindings"][0]
                self.assertEqual(binding["binding"], 0)
                self.assertEqual(binding["stride"], expected_stride)
                self.assertEqual(binding["input_rate"], "per_vertex")

                self.assertEqual(len(layout["attributes"]), len(expected_attrs))
                for actual, expected in zip(layout["attributes"], expected_attrs):
                    for key, val in expected.items():
                        self.assertEqual(actual[key], val, f"Mismatch in {key} for {actual['name']}")

    def test_canonical_format_strings_in_vulkan_format_table(self):
        canonical_formats = {
            "f32": ("VK_FORMAT_R32_SFLOAT", 100),
            "f32x2": ("VK_FORMAT_R32G32_SFLOAT", 103),
            "f32x3": ("VK_FORMAT_R32G32B32_SFLOAT", 106),
            "f32x4": ("VK_FORMAT_R32G32B32A32_SFLOAT", 109),
            "u32": ("VK_FORMAT_R32_UINT", 98),
            "i32": ("VK_FORMAT_R32_SINT", 99),
            "u16": ("VK_FORMAT_R16_UINT", 74),
            "i16": ("VK_FORMAT_R16_SINT", 75),
        }
        for fmt, (vk_name, vk_code) in canonical_formats.items():
            self.assertIn(fmt, _VK_FORMAT_NAMES)
            name, code = _VK_FORMAT_NAMES[fmt]
            self.assertEqual(name, vk_name)
            self.assertEqual(code, vk_code)

    def test_shared_vertex_layout_across_multiple_pipelines(self):
        @vertex
        def vs_depth(vert: StandardPbrVertex) -> Vertex:
            return Vertex(vec4(vert.position, 1.0))

        @fragment
        def fs_depth() -> Value:
            return vec4(1.0, 1.0, 1.0, 1.0)

        @vertex
        def vs_forward(vert: StandardPbrVertex) -> Vertex:
            pos = vert.position + vert.normal * 0.05
            return Vertex(vec4(pos, 1.0), varyings={"norm": vert.normal, "uv": vert.uv})

        @fragment
        def fs_forward(norm: Value, uv: Value) -> Value:
            return vec4(norm.x, norm.y, uv.x, 1.0)

        depth_pipe = Pipeline(vs_depth, fs_depth)
        fwd_pipe = Pipeline(vs_forward, fs_forward)

        try:
            TOOLS.preflight(Target.VULKAN)
        except CompileError as error:
            self.skipTest(str(error))

        with tempfile.TemporaryDirectory() as tmp:
            depth_manifest_path = compile_pipeline(depth_pipe, Target.VULKAN, Path(tmp) / "depth", TOOLS)
            fwd_manifest_path = compile_pipeline(fwd_pipe, Target.VULKAN, Path(tmp) / "forward", TOOLS)

            depth_manifest = json.loads(depth_manifest_path.read_text())
            fwd_manifest = json.loads(fwd_manifest_path.read_text())

            self.assertEqual(depth_manifest["input_layout"], fwd_manifest["input_layout"])
            self.assertEqual(depth_manifest["input_layout"]["bindings"][0]["stride"], 32)
            self.assertEqual(len(depth_manifest["input_layout"]["attributes"]), 3)

            depth_prog = load_program(depth_manifest_path)
            fwd_prog = load_program(fwd_manifest_path)
            self.assertIsNotNone(depth_prog.input_layout)
            self.assertEqual(depth_prog.input_layout, fwd_prog.input_layout)


if __name__ == "__main__":
    unittest.main()
