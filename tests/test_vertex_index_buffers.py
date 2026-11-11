import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlir_gpu.compiler import CompileError, RASTER_CONVENTION, Target, Toolchain, compile_pipeline
from mlir_gpu.frontend import Draw, Pipeline, ShaderIO, Type, Value, Vertex, fragment, vec4, vertex
from mlir_gpu.parameters import struct, is_struct
from mlir_gpu.viewer import GraphicsProgram, Shader, ViewerError, load_program, _Vulkan, _metal_libraries

LLVM_BIN = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
GPU_TOOLS_BIN = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
TOOLS = Toolchain(LLVM_BIN, GPU_TOOLS_BIN)


@struct(layout="vertex")
class StandardVertex:
    position: Type.VEC3
    normal: Type.VEC3
    uv: Type.VEC2


@struct(layout="vertex")
class ColorVertex:
    position: Type.VEC3
    color: Type.VEC4


@struct(layout="std430")
class UniformBlock:
    model: Type.MAT4X4


class TestVertexStructLayout(unittest.TestCase):
    def test_packed_vertex_stride_and_offsets(self):
        self.assertEqual(StandardVertex.__struct_size__, 32)
        self.assertEqual(StandardVertex.__struct_alignment__, 4)
        self.assertTrue(StandardVertex._is_vertex_struct)
        self.assertEqual(StandardVertex.__struct_layout__, "vertex")

        fields = {f.name: f for f in StandardVertex.__struct_fields__}
        self.assertEqual(fields["position"].offset, 0)
        self.assertEqual(fields["position"].size, 12)
        self.assertEqual(fields["position"].alignment, 4)

        self.assertEqual(fields["normal"].offset, 12)
        self.assertEqual(fields["normal"].size, 12)
        self.assertEqual(fields["normal"].alignment, 4)

        self.assertEqual(fields["uv"].offset, 24)
        self.assertEqual(fields["uv"].size, 8)
        self.assertEqual(fields["uv"].alignment, 4)

    def test_std430_has_sixteen_byte_alignment_for_vec3(self):
        @struct(layout="std430")
        class Std430Vertex:
            position: Type.VEC3
            normal: Type.VEC3
            uv: Type.VEC2

        self.assertEqual(Std430Vertex.__struct_size__, 48)
        fields = {f.name: f for f in Std430Vertex.__struct_fields__}
        self.assertEqual(fields["position"].offset, 0)
        self.assertEqual(fields["normal"].offset, 16)
        self.assertEqual(fields["uv"].offset, 32)

    def test_vertex_binary_packing(self):
        vert = StandardVertex(
            position=(1.0, 2.0, 3.0),
            normal=(0.0, 1.0, 0.0),
            uv=(0.5, 0.25),
        )
        packed = vert.to_bytes()
        self.assertEqual(len(packed), 32)
        unpacked = StandardVertex.from_bytes(packed)
        self.assertEqual(unpacked.position, (1.0, 2.0, 3.0))
        self.assertEqual(unpacked.normal, (0.0, 1.0, 0.0))
        self.assertEqual(unpacked.uv, (0.5, 0.25))

    def test_as_input_layout_generation(self):
        layout = StandardVertex.as_input_layout(binding=0)
        self.assertEqual(layout["bindings"], [{"binding": 0, "stride": 32, "input_rate": "per_vertex"}])
        attrs = layout["attributes"]
        self.assertEqual(len(attrs), 3)

        self.assertEqual(attrs[0]["name"], "position")
        self.assertEqual(attrs[0]["location"], 0)
        self.assertEqual(attrs[0]["format"], "f32x3")
        self.assertEqual(attrs[0]["offset"], 0)
        self.assertEqual(attrs[0]["semantic_name"], "POSITION")
        self.assertEqual(attrs[0]["semantic_index"], 0)

        self.assertEqual(attrs[1]["name"], "normal")
        self.assertEqual(attrs[1]["location"], 1)
        self.assertEqual(attrs[1]["format"], "f32x3")
        self.assertEqual(attrs[1]["offset"], 12)
        self.assertEqual(attrs[1]["semantic_name"], "NORMAL")
        self.assertEqual(attrs[1]["semantic_index"], 0)

        self.assertEqual(attrs[2]["name"], "uv")
        self.assertEqual(attrs[2]["location"], 2)
        self.assertEqual(attrs[2]["format"], "f32x2")
        self.assertEqual(attrs[2]["offset"], 24)
        self.assertEqual(attrs[2]["semantic_name"], "TEXCOORD")
        self.assertEqual(attrs[2]["semantic_index"], 0)

    def test_color_vertex_input_layout_semantic(self):
        layout = ColorVertex.as_input_layout()
        attrs = layout["attributes"]
        self.assertEqual(len(attrs), 2)
        self.assertEqual(attrs[0]["name"], "position")
        self.assertEqual(attrs[0]["semantic_name"], "POSITION")
        self.assertEqual(attrs[0]["semantic_index"], 0)
        self.assertEqual(attrs[1]["name"], "color")
        self.assertEqual(attrs[1]["semantic_name"], "COLOR")
        self.assertEqual(attrs[1]["semantic_index"], 0)

    def test_vertex_struct_u16_i16_from_bytes_roundtrip(self):
        @struct(layout="vertex")
        class MixedVertex:
            pos: Type.VEC3
            id: Type.U16
            flags: Type.I16

        v = MixedVertex(pos=(1.0, 2.0, 3.0), id=65530, flags=-1234)
        b = v.to_bytes()
        v2 = MixedVertex.from_bytes(b)
        self.assertEqual(v2.pos, (1.0, 2.0, 3.0))
        self.assertEqual(v2.id, 65530)
        self.assertEqual(v2.flags, -1234)
        self.assertEqual(v, v2)

    def test_invalid_layout_rejected(self):
        with self.assertRaises(ValueError):
            @struct(layout="unknown_layout")
            class BadLayout:
                pos: Type.VEC3


class TestDrawDataclass(unittest.TestCase):
    def test_non_indexed_draw_defaults(self):
        d = Draw()
        self.assertFalse(d.indexed)
        self.assertEqual(d.vertex_count, 3)
        self.assertEqual(d.to_dict(), {"topology": "triangle-list", "vertex_count": 3})

    def test_indexed_draw_valid(self):
        d = Draw(indexed=True, index_count=36, index_type="uint16", first_index=0, vertex_offset=0)
        self.assertTrue(d.indexed)
        self.assertEqual(d.index_count, 36)
        self.assertEqual(d.index_type, "uint16")
        expected = {
            "topology": "triangle-list",
            "indexed": True,
            "index_count": 36,
            "index_type": "uint16",
            "first_index": 0,
            "vertex_offset": 0,
            "instance_count": 1,
            "first_instance": 0,
        }
        self.assertEqual(d.to_dict(), expected)

    def test_indexed_draw_uint32(self):
        d = Draw(indexed=True, index_count=6, index_type="u32")
        self.assertEqual(d.to_dict()["index_type"], "uint32")

    def test_indexed_draw_non_multiple_of_three_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=7, index_type="uint16")

    def test_indexed_draw_missing_index_count_rejected(self):
        with self.assertRaises((ValueError, TypeError)):
            Draw(indexed=True, index_type="uint16")

    def test_indexed_draw_invalid_type_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=6, index_type="uint8")


class TestVertexShaderTracing(unittest.TestCase):
    def test_vertex_struct_input_tracing(self):
        @vertex
        def vs(vert: StandardVertex) -> Vertex:
            return Vertex(
                position=vec4(vert.position[0], vert.position[1], vert.position[2], 1.0),
                varyings={"normal": vert.normal, "uv": vert.uv},
            )

        stage = vs.trace()
        self.assertEqual(stage.kind, "vertex")
        self.assertEqual(len(stage.input_interface), 3)
        self.assertEqual(stage.input_interface[0].name, "position")
        self.assertEqual(stage.input_interface[0].location, 0)
        self.assertEqual(stage.input_interface[1].name, "normal")
        self.assertEqual(stage.input_interface[1].location, 1)
        self.assertEqual(stage.input_interface[2].name, "uv")
        self.assertEqual(stage.input_interface[2].location, 2)
        self.assertIsNotNone(stage.input_layout)
        self.assertEqual(stage.input_layout["bindings"][0]["stride"], 32)

    def test_discrete_attribute_parameters_tracing(self):
        @vertex
        def vs_discrete(position: Type.VEC3, normal: Type.VEC3, uv: Type.VEC2) -> Vertex:
            return Vertex(
                position=vec4(position[0], position[1], position[2], 1.0),
                varyings={"normal": normal, "uv": uv},
            )

        stage = vs_discrete.trace()
        self.assertEqual(len(stage.input_interface), 3)
        self.assertEqual(stage.input_interface[0].name, "position")
        self.assertEqual(stage.input_interface[0].location, 0)
        self.assertEqual(stage.input_interface[1].name, "normal")
        self.assertEqual(stage.input_interface[1].location, 1)
        self.assertEqual(stage.input_interface[2].name, "uv")
        self.assertEqual(stage.input_interface[2].location, 2)

    def test_procedural_vertex_index_backwards_compatible(self):
        @vertex
        def vs_procedural(vertex_index: Type.U32) -> Vertex:
            return Vertex(position=vec4(0.0, 0.0, 0.0, 1.0))

        stage = vs_procedural.trace()
        self.assertEqual(len(stage.input_interface), 1)
        self.assertEqual(stage.input_interface[0].builtin, "vertex_index")


class TestPipelineManifestAndViewerReflection(unittest.TestCase):
    def test_manifest_contains_input_layout_and_indexed_draw(self):
        @vertex
        def vs(vert: StandardVertex) -> Vertex:
            return Vertex(
                position=vec4(vert.position[0], vert.position[1], vert.position[2], 1.0),
                varyings={"uv": vert.uv},
            )

        @fragment
        def fs(uv: Type.VEC2) -> Value:
            return vec4(uv[0], uv[1], 0.0, 1.0)

        draw = Draw(indexed=True, index_count=36, index_type="uint16")
        pipeline = Pipeline(vertex=vs, fragment=fs, draw=draw)

        v_stage, f_stage = pipeline.trace()
        self.assertIsNotNone(v_stage.input_layout)

        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = Path(temp_dir) / "pipeline.json"
            manifest = {
                "schema_version": 3,
                "target": "vulkan",
                "target_profile": {"platform": "universal", "tier": "desktop"},
                "entries": [
                    {"stage": "vertex", "entry_point": "vs", "binary": "vertex.spv"},
                    {"stage": "fragment", "entry_point": "fs", "binary": "fragment.spv"},
                ],
                "interface": {
                    "vertex": {
                        "name": "vs",
                        "stage": "vertex",
                        "inputs": [
                            {"name": "position", "type": "f32x3", "location": 0},
                            {"name": "normal", "type": "f32x3", "location": 1},
                            {"name": "uv", "type": "f32x2", "location": 2},
                        ],
                        "outputs": [
                            {"name": "position", "type": "f32x4", "builtin": "position"},
                            {"name": "uv", "type": "f32x2", "location": 0, "interpolation": "perspective"},
                        ],
                    },
                    "fragment": {
                        "name": "fs",
                        "stage": "fragment",
                        "inputs": [
                            {"name": "uv", "type": "f32x2", "location": 0, "interpolation": "perspective"},
                        ],
                        "outputs": [{"name": "target0", "type": "f32x4", "location": 0, "interpolation": "perspective"}],
                    },
                },
                "input_layout": v_stage.input_layout,
                "draw": draw.to_dict(),
                "raster_convention": RASTER_CONVENTION,
            }
            manifest_path.write_text(json.dumps(manifest))
            (Path(temp_dir) / "vertex.spv").write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 28)
            (Path(temp_dir) / "fragment.spv").write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 28)

            program = load_program(manifest_path)
            self.assertTrue(program.draw.indexed)
            self.assertEqual(program.draw.index_count, 36)
            self.assertEqual(program.draw.index_type, "uint16")
            self.assertIsNotNone(program.input_layout)
            self.assertEqual(program.input_layout["bindings"][0]["stride"], 32)

    def test_vulkan_vertex_input_state_setup(self):
        layout = {
            "bindings": [{"binding": 0, "stride": 32, "input_rate": "per_vertex"}],
            "attributes": [
                {"name": "position", "location": 0, "binding": 0, "format": "f32x3", "offset": 0},
                {"name": "normal", "location": 1, "binding": 0, "format": "f32x3", "offset": 12},
                {"name": "uv", "location": 2, "binding": 0, "format": "f32x2", "offset": 24},
            ],
        }
        draw = Draw(indexed=True, index_count=36, index_type="uint16")
        prog = GraphicsProgram(
            vertex=Shader("vs", b"\x03\x02\x23\x07" + b"\x00" * 28),
            fragment=Shader("fs", b"\x03\x02\x23\x07" + b"\x00" * 28),
            target="vulkan",
            draw=draw,
            input_layout=layout,
        )

        mock_vk = MagicMock()
        mock_vk.VK_NULL_HANDLE = 0
        mock_vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3
        mock_vk.VK_POLYGON_MODE_FILL = 0
        mock_vk.VK_CULL_MODE_NONE = 0
        mock_vk.VK_FRONT_FACE_COUNTER_CLOCKWISE = 1
        mock_vk.VK_SAMPLE_COUNT_1_BIT = 1
        mock_vk.VK_VERTEX_INPUT_RATE_VERTEX = 0
        mock_vk.VK_INDEX_TYPE_UINT16 = 0
        mock_vk.VK_INDEX_TYPE_UINT32 = 1

        vulkan_instance = object.__new__(_Vulkan)
        vulkan_instance.vk = mock_vk
        vulkan_instance.device = 1
        vulkan_instance.max_push_constants = 128
        vulkan_instance.max_width = 4096
        vulkan_instance.max_height = 4096

        resources = MagicMock()
        vulkan_instance.pipeline(resources, prog, 10, 800, 600, False)

        input_state_call = mock_vk.VkPipelineVertexInputStateCreateInfo.call_args
        self.assertIsNotNone(input_state_call)
        self.assertEqual(input_state_call.kwargs["vertexBindingDescriptionCount"], 1)
        self.assertEqual(input_state_call.kwargs["vertexAttributeDescriptionCount"], 3)

    def test_vulkan_indexed_draw_command_dispatch(self):
        mock_vk = MagicMock()
        mock_vk.VK_INDEX_TYPE_UINT16 = 0
        mock_vk.VK_INDEX_TYPE_UINT32 = 1
        mock_vk.VK_SUBPASS_CONTENTS_INLINE = 0

        vulkan_instance = object.__new__(_Vulkan)
        vulkan_instance.vk = mock_vk
        vulkan_instance.device = 1
        vulkan_instance.pool = 2

        draw = Draw(indexed=True, index_count=36, index_type="uint16", first_index=0, vertex_offset=0)
        resources = MagicMock()
        vulkan_instance.command(resources, 1, 2, 3, 800, 600, draw, command=4)

        mock_vk.vkCmdDrawIndexed.assert_called_once_with(4, 36, 1, 0, 0, 0)
        mock_vk.vkCmdDraw.assert_not_called()

    def test_vulkan_buffer_bindings_command_dispatch(self):
        mock_vk = MagicMock()
        mock_vk.VK_INDEX_TYPE_UINT16 = 0
        mock_vk.VK_INDEX_TYPE_UINT32 = 1
        mock_vk.VK_SUBPASS_CONTENTS_INLINE = 0

        vulkan_instance = object.__new__(_Vulkan)
        vulkan_instance.vk = mock_vk
        vulkan_instance.device = 1
        vulkan_instance.pool = 2

        draw = Draw(indexed=True, index_count=36, index_type="uint16", first_index=0, vertex_offset=0)
        resources = MagicMock()
        mock_vb = MagicMock()
        mock_ib = MagicMock()
        vulkan_instance.command(
            resources, 1, 2, 3, 800, 600, draw, command=4,
            vertex_buffer=mock_vb, index_buffer=mock_ib,
        )

        mock_vk.vkCmdBindVertexBuffers.assert_called_once_with(4, 0, 1, [mock_vb], [0])
        mock_vk.vkCmdBindIndexBuffer.assert_called_once_with(4, mock_ib, 0, 0)
        mock_vk.vkCmdDrawIndexed.assert_called_once_with(4, 36, 1, 0, 0, 0)
        mock_vk.vkCmdDraw.assert_not_called()

    def test_vulkan_procedural_draw_no_buffer_bindings(self):
        mock_vk = MagicMock()
        mock_vk.VK_SUBPASS_CONTENTS_INLINE = 0

        vulkan_instance = object.__new__(_Vulkan)
        vulkan_instance.vk = mock_vk
        vulkan_instance.device = 1
        vulkan_instance.pool = 2

        draw = Draw(vertex_count=3)
        resources = MagicMock()
        vulkan_instance.command(resources, 1, 2, 3, 800, 600, draw, command=4)

        mock_vk.vkCmdBindVertexBuffers.assert_not_called()
        mock_vk.vkCmdBindIndexBuffer.assert_not_called()
        mock_vk.vkCmdDraw.assert_called_once_with(4, 3, 1, 0, 0)
        mock_vk.vkCmdDrawIndexed.assert_not_called()

    def test_native_subviewer_buffer_argument_forwarding(self):
        import sys
        if sys.platform != "darwin":
            return
        prog = GraphicsProgram(
            vertex=Shader("vs", b"\x00" * 32),
            fragment=Shader("fs", b"\x00" * 32),
            target="metal",
            draw=Draw(indexed=True, index_count=36, index_type="uint16"),
            input_layout={"bindings": [], "attributes": []},
            vertex_buffer=b"\x01" * 32,
            index_buffer=b"\x02" * 12,
        )
        mock_toolchain = MagicMock()
        mock_toolchain.find.return_value = "mlir-gpu-metal-view"
        with _metal_libraries(prog, mock_toolchain) as (directory, arguments):
            self.assertIn("--vertex-buffer", arguments)
            vb_idx = arguments.index("--vertex-buffer")
            self.assertTrue(Path(arguments[vb_idx + 1]).is_file())
            self.assertEqual(Path(arguments[vb_idx + 1]).read_bytes(), b"\x01" * 32)

            self.assertIn("--index-buffer", arguments)
            ib_idx = arguments.index("--index-buffer")
            self.assertTrue(Path(arguments[ib_idx + 1]).is_file())
            self.assertEqual(Path(arguments[ib_idx + 1]).read_bytes(), b"\x02" * 12)

    def test_sparse_vertex_attribute_access_stacked_input_signature(self):
        @vertex
        def vs_sparse(vert: StandardVertex) -> Vertex:
            # Vertex input reading position (loc 0) and uv (loc 2) while omitting normal (loc 1)
            return Vertex(
                position=vec4(vert.position[0], vert.position[1], vert.position[2], 1.0),
                varyings={"uv": vert.uv},
            )

        @fragment
        def fs_sparse(uv: Type.VEC2) -> Value:
            return vec4(uv[0], uv[1], 0.0, 1.0)

        pipe = Pipeline(vs_sparse, fs_sparse)
        stage, _ = pipe.trace()
        self.assertIsNotNone(stage.input_layout)
        # Verify sparse attribute locations: position=0, normal=1, uv=2
        attrs = stage.input_layout["attributes"]
        self.assertEqual(len(attrs), 3)
        self.assertEqual(attrs[0]["location"], 0)
        self.assertEqual(attrs[1]["location"], 1)
        self.assertEqual(attrs[2]["location"], 2)

        for target in (Target.VULKAN, Target.DIRECTX, Target.METAL):
            with self.subTest(target=target.name):
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                with tempfile.TemporaryDirectory() as tmp:
                    out = compile_pipeline(pipe, target, Path(tmp) / "out", TOOLS)
                    self.assertTrue(out.exists())
                    manifest = json.loads(out.read_text())
                    self.assertIn("input_layout", manifest)
                    self.assertEqual(len(manifest["input_layout"]["attributes"]), 3)


if __name__ == "__main__":
    unittest.main()
