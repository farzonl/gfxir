"""Unit tests for multi-draw command buffer recording and pipeline state caching."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from mlir_gpu.compiler import DrawCall
from mlir_gpu.frontend import Draw
from mlir_gpu.viewer import PipelineState, Shader, _Vulkan


class MockVulkanApi:
    """Mock Vulkan API capturing sequential command recording calls."""

    def __init__(self):
        self.calls = []
        self.VK_PIPELINE_BIND_POINT_GRAPHICS = 0
        self.VK_SUBPASS_CONTENTS_INLINE = 0
        self.VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
        self.VK_SHADER_STAGE_VERTEX_BIT = 1
        self.VK_SHADER_STAGE_FRAGMENT_BIT = 2

        class Ffi:
            def from_buffer(self, buf):
                return bytes(buf)

        self.ffi = Ffi()

    def vkResetCommandBuffer(self, command, flags):
        self.calls.append(("vkResetCommandBuffer", command, flags))

    def vkBeginCommandBuffer(self, command, info):
        self.calls.append(("vkBeginCommandBuffer", command))

    def vkCmdBeginRenderPass(self, command, info, contents):
        self.calls.append(("vkCmdBeginRenderPass", command))

    def vkCmdBindPipeline(self, command, bind_point, pipeline):
        self.calls.append(("vkCmdBindPipeline", command, bind_point, pipeline))

    def vkCmdPushConstants(self, command, layout, stages, offset, size, data):
        self.calls.append(("vkCmdPushConstants", command, layout, stages, offset, size, bytes(data) if data else None))

    def vkCmdDraw(self, command, vertex_count, instance_count, first_vertex, first_instance):
        self.calls.append(("vkCmdDraw", command, vertex_count, instance_count, first_vertex, first_instance))

    def vkCmdDrawIndexed(self, command, index_count, instance_count, first_index, vertex_offset, first_instance):
        self.calls.append(("vkCmdDrawIndexed", command, index_count, instance_count, first_index, vertex_offset, first_instance))

    def vkCmdEndRenderPass(self, command):
        self.calls.append(("vkCmdEndRenderPass", command))

    def vkEndCommandBuffer(self, command):
        self.calls.append(("vkEndCommandBuffer", command))

    def VkRenderPassBeginInfo(self, **kwargs):
        return kwargs

    def VkCommandBufferBeginInfo(self, **kwargs):
        return kwargs

    def VkRect2D(self, **kwargs):
        return kwargs

    def VkExtent2D(self, **kwargs):
        return kwargs

    def VkClearValue(self, **kwargs):
        return kwargs

    def VkClearColorValue(self, **kwargs):
        return kwargs


class MultiDrawCommandRecordingTests(unittest.TestCase):
    """Tests for command recording loops, pipeline state switching, and parameter preservation."""

    def setUp(self):
        self.mock_vk = MockVulkanApi()
        self.ctx = object.__new__(_Vulkan)
        self.ctx.vk = self.mock_vk
        self.ctx.device = "mock_device"
        self.ctx.pool = "mock_pool"

        class MockPipelineHandle:
            def __init__(self, name="default"):
                self.name = name
            def __repr__(self):
                return f"<VkPipeline {self.name}>"

        self.pipeline_alpha = MockPipelineHandle("alpha")
        self.pipeline_beta = MockPipelineHandle("beta")
        self.pipeline_map = {
            "pipe_alpha": self.pipeline_alpha,
            "pipe_beta": self.pipeline_beta,
        }

        self.root_pipeline = MockPipelineHandle("default")
        setattr(self.root_pipeline, "pipeline_map", self.pipeline_map)

    def test_single_render_pass_for_entire_multi_draw_sequence(self):
        draws = [
            DrawCall(pipeline="pipe_alpha", vertex_count=36),
            DrawCall(pipeline="pipe_beta", vertex_count=72),
            DrawCall(pipeline="pipe_alpha", vertex_count=18),
        ]
        setattr(self.root_pipeline, "draws", draws)

        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=36),
            command="mock_cmd_buffer",
        )

        begin_rp = [c for c in self.mock_vk.calls if c[0] == "vkCmdBeginRenderPass"]
        end_rp = [c for c in self.mock_vk.calls if c[0] == "vkCmdEndRenderPass"]
        self.assertEqual(len(begin_rp), 1, "Multi-draw sequence must execute within exactly one render pass begin")
        self.assertEqual(len(end_rp), 1, "Multi-draw sequence must execute within exactly one render pass end")

    def test_pipeline_switching_between_distinct_pipelines(self):
        draws = [
            DrawCall(pipeline="pipe_alpha", vertex_count=36),
            DrawCall(pipeline="pipe_beta", vertex_count=72),
        ]
        setattr(self.root_pipeline, "draws", draws)

        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=36),
            command="mock_cmd_buffer",
        )

        bind_calls = [c for c in self.mock_vk.calls if c[0] == "vkCmdBindPipeline"]
        self.assertEqual(len(bind_calls), 2)
        self.assertEqual(bind_calls[0][3], self.pipeline_alpha)
        self.assertEqual(bind_calls[1][3], self.pipeline_beta)

    def test_redundant_pipeline_bind_eliminated_for_consecutive_draws(self):
        draws = [
            DrawCall(pipeline="pipe_alpha", vertex_count=12),
            DrawCall(pipeline="pipe_alpha", vertex_count=24),
            DrawCall(pipeline="pipe_beta", vertex_count=36),
            DrawCall(pipeline="pipe_beta", vertex_count=48),
        ]
        setattr(self.root_pipeline, "draws", draws)

        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=12),
            command="mock_cmd_buffer",
        )

        bind_calls = [c for c in self.mock_vk.calls if c[0] == "vkCmdBindPipeline"]
        self.assertEqual(len(bind_calls), 2, "Redundant pipeline state binds must be skipped for consecutive draws")
        self.assertEqual(bind_calls[0][3], self.pipeline_alpha)
        self.assertEqual(bind_calls[1][3], self.pipeline_beta)

    def test_per_draw_constant_rebinding_preserves_frame_constants(self):
        frame_parameters = {
            "schema_version": 1, "name": "frame", "layout": "gfx-root-f32-v1",
            "size": 16, "alignment": 16, "stages": ["fragment"],
            "ownership": "host", "access": "read-only", "addressable": False,
            "fields": [
                {"name": "time", "type": "f32", "offset": 0, "size": 4, "alignment": 4},
                {"name": "resolution", "type": "f32x2", "offset": 8, "size": 8, "alignment": 8},
            ],
            "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 0},
                "directx": {"storage": "root_constants", "parameter_index": 0, "register": 0, "space": 0,
                            "num32bit_constants": 4, "visibility": "all", "root_signature_version": "1.1"},
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": 16},
            },
        }

        draw_obj_0 = bytes([1] * 80)
        draw_obj_1 = bytes([2] * 80)

        draws = [
            DrawCall(pipeline="pipe_alpha", vertex_count=36, constant_offset=80, constant_size=80, parameters=draw_obj_0),
            DrawCall(pipeline="pipe_alpha", vertex_count=36, constant_offset=80, constant_size=80, parameters=draw_obj_1),
        ]
        setattr(self.root_pipeline, "draws", draws)

        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=36),
            layout="mock_layout",
            parameters=frame_parameters,
            elapsed=1.5,
            command="mock_cmd_buffer",
        )

        push_calls = [c for c in self.mock_vk.calls if c[0] == "vkCmdPushConstants"]
        self.assertEqual(len(push_calls), 3)

        # Call 0: Frame constants pushed once at offset 0
        self.assertEqual(push_calls[0][4], 0, "Frame constants must be pushed at offset 0")
        self.assertEqual(push_calls[0][5], 16, "Frame constants size is 16 bytes")

        # Call 1: Draw 0 per-draw constants pushed at offset 80
        self.assertEqual(push_calls[1][4], 80, "Per-draw constants must be pushed at offset 80")
        self.assertEqual(push_calls[1][5], 80)
        self.assertEqual(push_calls[1][6], draw_obj_0)

        # Call 2: Draw 1 per-draw constants updated at offset 80
        self.assertEqual(push_calls[2][4], 80, "Per-draw constants updated at offset 80 without clobbering offset 0")
        self.assertEqual(push_calls[2][5], 80)
        self.assertEqual(push_calls[2][6], draw_obj_1)

    def test_mixed_standard_and_indexed_draws(self):
        draws = [
            DrawCall(pipeline="pipe_alpha", type="standard", vertex_count=36, first_vertex=0, instance_count=1),
            DrawCall(pipeline="pipe_beta", type="indexed", index_count=72, first_index=12, vertex_offset=4, instance_count=2),
        ]
        setattr(self.root_pipeline, "draws", draws)

        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=36),
            command="mock_cmd_buffer",
        )

        draw_calls = [c for c in self.mock_vk.calls if c[0] in ("vkCmdDraw", "vkCmdDrawIndexed")]
        self.assertEqual(len(draw_calls), 2)

        # Draw 0: standard draw
        self.assertEqual(draw_calls[0][0], "vkCmdDraw")
        self.assertEqual(draw_calls[0][2], 36)  # vertex_count
        self.assertEqual(draw_calls[0][3], 1)   # instance_count
        self.assertEqual(draw_calls[0][4], 0)   # first_vertex

        # Draw 1: indexed draw
        self.assertEqual(draw_calls[1][0], "vkCmdDrawIndexed")
        self.assertEqual(draw_calls[1][2], 72)  # index_count
        self.assertEqual(draw_calls[1][3], 2)   # instance_count
        self.assertEqual(draw_calls[1][4], 12)  # first_index
        self.assertEqual(draw_calls[1][5], 4)   # vertex_offset


    def test_empty_draw_list_fallback_records_single_draw(self):
        setattr(self.root_pipeline, "draws", [])
        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=42),
            command="mock_cmd_buffer",
        )

        draw_calls = [c for c in self.mock_vk.calls if c[0] == "vkCmdDraw"]
        self.assertEqual(len(draw_calls), 1)
        self.assertEqual(draw_calls[0][2], 42)

    def test_parameter_push_with_dictionary_payload(self):
        draws = [
            DrawCall(pipeline="pipe_alpha", vertex_count=36, constant_offset=80, parameters={"scale": 2.0, "intensity": 1.0}),
        ]
        setattr(self.root_pipeline, "draws", draws)
        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=36),
            layout="mock_layout",
            command="mock_cmd_buffer",
        )

        push_calls = [c for c in self.mock_vk.calls if c[0] == "vkCmdPushConstants"]
        self.assertEqual(len(push_calls), 1)
        self.assertEqual(push_calls[0][4], 80)
        self.assertEqual(push_calls[0][5], 8)  # 2 floats = 8 bytes

    def test_sequential_indexed_draw_parameter_preservation(self):
        draws = [
            DrawCall(pipeline="pipe_beta", type="indexed", index_count=36, first_index=0, vertex_offset=0, instance_count=3),
            DrawCall(pipeline="pipe_beta", type="indexed", index_count=36, first_index=36, vertex_offset=24, instance_count=3),
        ]
        setattr(self.root_pipeline, "draws", draws)
        self.ctx.command(
            resources=None,
            render_pass="mock_render_pass",
            pipeline=self.root_pipeline,
            framebuffer="mock_framebuffer",
            width=800,
            height=600,
            draw=Draw(vertex_count=36),
            command="mock_cmd_buffer",
        )

        draw_calls = [c for c in self.mock_vk.calls if c[0] == "vkCmdDrawIndexed"]
        self.assertEqual(len(draw_calls), 2)
        # Draw 0
        self.assertEqual(draw_calls[0][2], 36)
        self.assertEqual(draw_calls[0][3], 3)
        self.assertEqual(draw_calls[0][4], 0)
        self.assertEqual(draw_calls[0][5], 0)
        # Draw 1
        self.assertEqual(draw_calls[1][2], 36)
        self.assertEqual(draw_calls[1][3], 3)
        self.assertEqual(draw_calls[1][4], 36)
        self.assertEqual(draw_calls[1][5], 24)


if __name__ == "__main__":
    unittest.main()
