"""Empirical tests for index and vertex buffer boundaries, stride calculations, and offsets."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from mlir_gpu.compiler import CompileError, Target, Toolchain, compile_pipeline
from mlir_gpu.frontend import Draw, Pipeline, Vertex, fragment, vec2, vec3, vec4, vertex
from mlir_gpu.parameters import Type, struct
from mlir_gpu.viewer import GraphicsProgram, ViewerError, _VK_FORMAT_NAMES, _Vulkan, load_program

LLVM_BIN = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
GPU_TOOLS_BIN = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
TOOLS = Toolchain(LLVM_BIN, GPU_TOOLS_BIN)


class TestIndexBufferBoundaryValues(unittest.TestCase):
    def test_uint16_index_boundary_zero(self):
        draw = Draw(indexed=True, index_count=0, index_type="uint16")
        self.assertTrue(draw.indexed)
        self.assertEqual(draw.index_count, 0)
        self.assertEqual(draw.index_type, "uint16")
        self.assertEqual(draw.to_dict()["index_count"], 0)

    def test_uint16_index_boundary_maximum(self):
        draw = Draw(indexed=True, index_count=65535, index_type="uint16")
        self.assertEqual(draw.index_count, 65535)
        self.assertEqual(draw.to_dict()["index_type"], "uint16")

    def test_uint16_index_count_exceeding_sixteen_bits(self):
        draw = Draw(indexed=True, index_count=65538, index_type="uint16")
        self.assertEqual(draw.index_count, 65538)

    def test_uint16_packing_boundary_values(self):
        @struct(layout="vertex")
        class U16Vertex:
            idx: Type.U16

        v0 = U16Vertex(0)
        self.assertEqual(v0.to_bytes()[:2], b"\x00\x00")
        v_max = U16Vertex(65535)
        self.assertEqual(v_max.to_bytes()[:2], b"\xff\xff")

    def test_uint16_packing_overflow_rejected(self):
        @struct(layout="vertex")
        class U16Vertex:
            idx: Type.U16

        with self.assertRaises(ValueError) as ctx:
            U16Vertex(65536).to_bytes()
        self.assertIn("unsigned 16-bit", str(ctx.exception))

    def test_uint16_packing_negative_rejected(self):
        @struct(layout="vertex")
        class U16Vertex:
            idx: Type.U16

        with self.assertRaises(ValueError) as ctx:
            U16Vertex(-1).to_bytes()
        self.assertIn("unsigned 16-bit", str(ctx.exception))

    def test_int16_packing_boundary_values(self):
        @struct(layout="vertex")
        class I16Vertex:
            val: Type.I16

        v_min = I16Vertex(-32768)
        self.assertEqual(v_min.to_bytes()[:2], b"\x00\x80")
        v_max = I16Vertex(32767)
        self.assertEqual(v_max.to_bytes()[:2], b"\xff\x7f")

    def test_int16_packing_overflow_rejected(self):
        @struct(layout="vertex")
        class I16Vertex:
            val: Type.I16

        with self.assertRaises(ValueError):
            I16Vertex(32768).to_bytes()
        with self.assertRaises(ValueError):
            I16Vertex(-32769).to_bytes()

    def test_uint32_index_boundary_maximum(self):
        draw = Draw(indexed=True, index_count=4294967295, index_type="uint32")
        self.assertEqual(draw.index_count, 4294967295)

    def test_uint32_index_overflow_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=4294967296, index_type="uint32")

    def test_uint32_packing_boundary_values(self):
        @struct(layout="vertex")
        class U32Vertex:
            idx: Type.U32

        v0 = U32Vertex(0)
        self.assertEqual(v0.to_bytes()[:4], b"\x00\x00\x00\x00")
        v_max = U32Vertex(0xFFFFFFFF)
        self.assertEqual(v_max.to_bytes()[:4], b"\xff\xff\xff\xff")

    def test_uint32_packing_overflow_rejected(self):
        @struct(layout="vertex")
        class U32Vertex:
            idx: Type.U32

        with self.assertRaises(ValueError):
            U32Vertex(0x100000000).to_bytes()
        with self.assertRaises(ValueError):
            U32Vertex(-1).to_bytes()


class TestTriangleListIndexCountConstraints(unittest.TestCase):
    def test_zero_index_count_accepted(self):
        draw = Draw(indexed=True, index_count=0, index_type="uint16")
        self.assertEqual(draw.index_count, 0)

    def test_non_multiple_of_three_index_counts_rejected(self):
        for invalid_count in (1, 2, 4, 5, 7, 8, 10, 65534, 65536):
            with self.subTest(invalid_count=invalid_count):
                with self.assertRaises(ValueError) as ctx:
                    Draw(indexed=True, index_count=invalid_count, index_type="uint16")
                self.assertIn("multiple of three", str(ctx.exception))

    def test_negative_index_count_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=-3, index_type="uint16")

    def test_boolean_index_count_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=True, index_type="uint16")

    def test_float_index_count_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3.0, index_type="uint16")


class TestIndexAndVertexOffsetLimits(unittest.TestCase):
    def test_first_index_nonnegative_boundary(self):
        draw_zero = Draw(indexed=True, index_count=3, index_type="uint16", first_index=0)
        self.assertEqual(draw_zero.first_index, 0)
        draw_max = Draw(indexed=True, index_count=3, index_type="uint16", first_index=0xFFFFFFFF)
        self.assertEqual(draw_max.first_index, 0xFFFFFFFF)

    def test_first_index_negative_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3, index_type="uint16", first_index=-1)

    def test_first_index_u32_overflow_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3, index_type="uint16", first_index=0x100000000)

    def test_vertex_offset_signed_range_supported(self):
        draw_neg = Draw(indexed=True, index_count=3, index_type="uint16", vertex_offset=-50)
        self.assertEqual(draw_neg.vertex_offset, -50)
        self.assertEqual(draw_neg.to_dict()["vertex_offset"], -50)

    def test_vertex_offset_negative_limit(self):
        draw_min = Draw(indexed=True, index_count=3, index_type="uint16", vertex_offset=-0x80000000)
        self.assertEqual(draw_min.vertex_offset, -2147483648)

    def test_vertex_offset_positive_limit(self):
        draw_max = Draw(indexed=True, index_count=3, index_type="uint16", vertex_offset=0x7FFFFFFF)
        self.assertEqual(draw_max.vertex_offset, 2147483647)

    def test_vertex_offset_underflow_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3, index_type="uint16", vertex_offset=-0x80000001)

    def test_vertex_offset_overflow_rejected(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3, index_type="uint16", vertex_offset=0x80000000)

    def test_instance_count_must_be_positive(self):
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3, index_type="uint16", instance_count=0)
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3, index_type="uint16", instance_count=-1)

    def test_first_instance_nonnegative_boundary(self):
        draw0 = Draw(indexed=True, index_count=3, index_type="uint16", first_instance=0)
        self.assertEqual(draw0.first_instance, 0)
        with self.assertRaises(ValueError):
            Draw(indexed=True, index_count=3, index_type="uint16", first_instance=-1)


class TestVertexBufferStrideCalculations(unittest.TestCase):
    def test_interleaved_pos_norm_uv_stride(self):
        @struct(layout="vertex")
        class VertexPosNormUv:
            position: Type.VEC3
            normal: Type.VEC3
            uv: Type.VEC2

        layout = VertexPosNormUv.as_input_layout()
        self.assertEqual(layout["bindings"][0]["stride"], 32)
        attrs = {a["name"]: a for a in layout["attributes"]}
        self.assertEqual(attrs["position"]["offset"], 0)
        self.assertEqual(attrs["position"]["size"], 12)
        self.assertEqual(attrs["position"]["semantic_name"], "POSITION")
        self.assertEqual(attrs["normal"]["offset"], 12)
        self.assertEqual(attrs["normal"]["size"], 12)
        self.assertEqual(attrs["normal"]["semantic_name"], "NORMAL")
        self.assertEqual(attrs["uv"]["offset"], 24)
        self.assertEqual(attrs["uv"]["size"], 8)
        self.assertEqual(attrs["uv"]["semantic_name"], "TEXCOORD")

    def test_compact_pos_uv_stride(self):
        @struct(layout="vertex")
        class VertexPosUv:
            position: Type.VEC3
            uv: Type.VEC2

        layout = VertexPosUv.as_input_layout()
        self.assertEqual(layout["bindings"][0]["stride"], 20)
        attrs = {a["name"]: a for a in layout["attributes"]}
        self.assertEqual(attrs["position"]["offset"], 0)
        self.assertEqual(attrs["position"]["size"], 12)
        self.assertEqual(attrs["uv"]["offset"], 12)
        self.assertEqual(attrs["uv"]["size"], 8)

    def test_mismatched_component_alignments_stride(self):
        @struct(layout="vertex")
        class MismatchedVertex:
            pos: Type.VEC3
            scalar: Type.F32
            uv: Type.VEC2

        layout = MismatchedVertex.as_input_layout()
        self.assertEqual(layout["bindings"][0]["stride"], 24)
        attrs = {a["name"]: a for a in layout["attributes"]}
        self.assertEqual(attrs["pos"]["offset"], 0)
        self.assertEqual(attrs["pos"]["size"], 12)
        self.assertEqual(attrs["scalar"]["offset"], 12)
        self.assertEqual(attrs["scalar"]["size"], 4)
        self.assertEqual(attrs["uv"]["offset"], 16)
        self.assertEqual(attrs["uv"]["size"], 8)

    def test_mixed_u16_and_vec3_stride(self):
        @struct(layout="vertex")
        class PaddedVertex:
            idx: Type.U16
            pos: Type.VEC3

        layout = PaddedVertex.as_input_layout()
        self.assertEqual(layout["bindings"][0]["stride"], 16)
        attrs = {a["name"]: a for a in layout["attributes"]}
        self.assertEqual(attrs["idx"]["offset"], 0)
        self.assertEqual(attrs["idx"]["size"], 2)
        self.assertEqual(attrs["pos"]["offset"], 4)
        self.assertEqual(attrs["pos"]["size"], 12)

    def test_vertex_struct_vs_std430_alignment_differences(self):
        @struct(layout="vertex")
        class VertexLayoutStruct:
            pos: Type.VEC3
            norm: Type.VEC3
            uv: Type.VEC2

        @struct(layout="std430")
        class Std430LayoutStruct:
            pos: Type.VEC3
            norm: Type.VEC3
            uv: Type.VEC2

        self.assertEqual(VertexLayoutStruct.__struct_size__, 32)
        self.assertEqual(VertexLayoutStruct.__struct_alignment__, 4)
        self.assertEqual(Std430LayoutStruct.__struct_size__, 48)
        self.assertEqual(Std430LayoutStruct.__struct_alignment__, 16)

    def test_unsupported_vertex_struct_types_rejected(self):
        with self.assertRaises(TypeError):
            @struct(layout="vertex")
            class UnsupportedField:
                val: Type.F64


class TestIndexedDrawPipelineCompilationAndDispatch(unittest.TestCase):
    def test_compile_indexed_pipeline_across_all_targets(self):
        @struct(layout="vertex")
        class StandardVertex:
            pos: Type.VEC3
            uv: Type.VEC2

        @vertex
        def test_vs(v: StandardVertex):
            return Vertex(vec4(v.pos, 1.0))

        @fragment
        def test_fs():
            return vec4(1.0, 1.0, 1.0, 1.0)

        for target in (Target.VULKAN, Target.DIRECTX, Target.METAL):
            with self.subTest(target=target.name):
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                with tempfile.TemporaryDirectory() as tmpdir:
                    out_dir = Path(tmpdir) / "pipeline"
                    draw = Draw(indexed=True, index_count=36, index_type="uint16", first_index=3, vertex_offset=-2)
                    pipe = Pipeline(test_vs, test_fs, draw=draw)
                    compile_pipeline(pipe, target, out_dir, TOOLS)
                    prog = load_program(out_dir / "pipeline.json")
                    self.assertTrue(prog.draw.indexed)
                    self.assertEqual(prog.draw.index_count, 36)
                    self.assertEqual(prog.draw.index_type, "uint16")
                    self.assertEqual(prog.draw.first_index, 3)
                    self.assertEqual(prog.draw.vertex_offset, -2)
                    self.assertIsNotNone(prog.input_layout)
                    self.assertEqual(prog.input_layout["bindings"][0]["stride"], 20)

    def test_vulkan_command_draw_indexed_parameter_forwarding(self):
        class MockVulkanFFI:
            def __init__(self):
                self.dispatches = []
                self.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3
                self.VK_INDEX_TYPE_UINT16 = 0
                self.VK_INDEX_TYPE_UINT32 = 1
                self.VK_NULL_HANDLE = None
                self.VK_PIPELINE_BIND_POINT_GRAPHICS = 0
                self.VK_SUBPASS_CONTENTS_INLINE = 0

            def VkCommandBufferBeginInfo(self, **kwargs): return kwargs
            def VkRenderPassBeginInfo(self, **kwargs): return kwargs
            def VkRect2D(self, **kwargs): return kwargs
            def VkExtent2D(self, **kwargs): return kwargs
            def VkClearValue(self, **kwargs): return kwargs
            def VkClearColorValue(self, **kwargs): return kwargs

            def vkResetCommandBuffer(self, *args): pass
            def vkBeginCommandBuffer(self, *args): pass
            def vkCmdBeginRenderPass(self, *args): pass
            def vkCmdBindPipeline(self, *args): pass
            def vkCmdEndRenderPass(self, *args): pass
            def vkEndCommandBuffer(self, *args): pass

            def vkCmdDrawIndexed(self, cmd, index_count, instance_count, first_index, vertex_offset, first_instance):
                self.dispatches.append(("DrawIndexed", {
                    "index_count": index_count,
                    "instance_count": instance_count,
                    "first_index": first_index,
                    "vertex_offset": vertex_offset,
                    "first_instance": first_instance,
                }))

        mock_vk = MockVulkanFFI()
        vulkan = _Vulkan.__new__(_Vulkan)
        vulkan.vk = mock_vk

        draw = Draw(indexed=True, index_count=18, index_type="uint16", first_index=3, vertex_offset=-4, instance_count=2, first_instance=1)
        vulkan.command("res", "rp", "pipe", "fb", 800, 600, draw, command="cmd")
        self.assertEqual(len(mock_vk.dispatches), 1)
        op, kwargs = mock_vk.dispatches[0]
        self.assertEqual(op, "DrawIndexed")
        self.assertEqual(kwargs["index_count"], 18)
        self.assertEqual(kwargs["instance_count"], 2)
        self.assertEqual(kwargs["first_index"], 3)
        self.assertEqual(kwargs["vertex_offset"], -4)
        self.assertEqual(kwargs["first_instance"], 1)

    def test_vulkan_command_draw_non_indexed_parameter_forwarding(self):
        class MockVulkanFFI:
            def __init__(self):
                self.dispatches = []
                self.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = 3
                self.VK_NULL_HANDLE = None
                self.VK_PIPELINE_BIND_POINT_GRAPHICS = 0
                self.VK_SUBPASS_CONTENTS_INLINE = 0

            def VkCommandBufferBeginInfo(self, **kwargs): return kwargs
            def VkRenderPassBeginInfo(self, **kwargs): return kwargs
            def VkRect2D(self, **kwargs): return kwargs
            def VkExtent2D(self, **kwargs): return kwargs
            def VkClearValue(self, **kwargs): return kwargs
            def VkClearColorValue(self, **kwargs): return kwargs

            def vkResetCommandBuffer(self, *args): pass
            def vkBeginCommandBuffer(self, *args): pass
            def vkCmdBeginRenderPass(self, *args): pass
            def vkCmdBindPipeline(self, *args): pass
            def vkCmdEndRenderPass(self, *args): pass
            def vkEndCommandBuffer(self, *args): pass

            def vkCmdDraw(self, cmd, vertex_count, instance_count, first_vertex, first_instance):
                self.dispatches.append(("Draw", {
                    "vertex_count": vertex_count,
                    "instance_count": instance_count,
                    "first_vertex": first_vertex,
                    "first_instance": first_instance,
                }))

        mock_vk = MockVulkanFFI()
        vulkan = _Vulkan.__new__(_Vulkan)
        vulkan.vk = mock_vk

        draw = Draw(indexed=False, vertex_count=9, first_vertex=3, instance_count=4, first_instance=2)
        vulkan.command("res", "rp", "pipe", "fb", 800, 600, draw, command="cmd")
        self.assertEqual(len(mock_vk.dispatches), 1)
        op, kwargs = mock_vk.dispatches[0]
        self.assertEqual(op, "Draw")
        self.assertEqual(kwargs["vertex_count"], 9)
        self.assertEqual(kwargs["instance_count"], 4)
        self.assertEqual(kwargs["first_vertex"], 3)
        self.assertEqual(kwargs["first_instance"], 2)

    def test_vulkan_vertex_input_format_mapping(self):
        self.assertEqual(_VK_FORMAT_NAMES["f32x3"][0], "VK_FORMAT_R32G32B32_SFLOAT")
        self.assertEqual(_VK_FORMAT_NAMES["f32x2"][0], "VK_FORMAT_R32G32_SFLOAT")
        self.assertEqual(_VK_FORMAT_NAMES["u16"][0], "VK_FORMAT_R16_UINT")
        self.assertEqual(_VK_FORMAT_NAMES["i16"][0], "VK_FORMAT_R16_SINT")
        self.assertEqual(_VK_FORMAT_NAMES["u32"][0], "VK_FORMAT_R32_UINT")
        self.assertEqual(_VK_FORMAT_NAMES["i32"][0], "VK_FORMAT_R32_SINT")


if __name__ == "__main__":
    unittest.main()
