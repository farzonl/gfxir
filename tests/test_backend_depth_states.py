import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlir_gpu import (
    DepthStencil,
    Draw,
    MLIRPipeline,
    Pipeline,
    Type,
    Value,
    compile_pipeline,
    fragment,
    fullscreen_vertex,
    vec4,
    vertex,
)
from mlir_gpu.compiler import Target
from mlir_gpu.interfaces import COMPARE_OPS, SUPPORTED_DEPTH_FORMATS
from mlir_gpu.viewer import (
    GraphicsProgram,
    Shader,
    ViewerError,
    _Vulkan,
    _directx_shaders,
    _metal_libraries,
    _vulkan_depth_format,
    load_program,
)


def _create_mock_vk():
    mock_vk = MagicMock()
    mock_vk.VK_FORMAT_D32_SFLOAT = 126
    mock_vk.VK_FORMAT_D16_UNORM = 124
    mock_vk.VK_IMAGE_TYPE_2D = 1
    mock_vk.VK_IMAGE_TILING_OPTIMAL = 0
    mock_vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000020
    mock_vk.VK_SHARING_MODE_EXCLUSIVE = 0
    mock_vk.VK_IMAGE_LAYOUT_UNDEFINED = 0
    mock_vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 3
    mock_vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL = 2
    mock_vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL = 6
    mock_vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR = 1000001002
    mock_vk.VK_IMAGE_ASPECT_DEPTH_BIT = 0x00000002
    mock_vk.VK_IMAGE_ASPECT_COLOR_BIT = 0x00000001
    mock_vk.VK_IMAGE_VIEW_TYPE_2D = 1
    mock_vk.VK_SAMPLE_COUNT_1_BIT = 1
    mock_vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT = 1
    mock_vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x00000200
    mock_vk.VK_ATTACHMENT_LOAD_OP_CLEAR = 1
    mock_vk.VK_ATTACHMENT_STORE_OP_DONT_CARE = 2
    mock_vk.VK_ATTACHMENT_STORE_OP_STORE = 0
    mock_vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT = 0x00000400
    mock_vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT = 0x00000100
    mock_vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT = 0x00000100
    mock_vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x00000400
    mock_vk.VK_PIPELINE_BIND_POINT_GRAPHICS = 2
    mock_vk.VK_SUBPASS_EXTERNAL = ~0
    mock_vk.VK_NULL_HANDLE = None
    mock_vk.VK_TRUE = 1
    mock_vk.VK_FALSE = 0

    mock_vk.VK_COMPARE_OP_NEVER = 0
    mock_vk.VK_COMPARE_OP_LESS = 1
    mock_vk.VK_COMPARE_OP_EQUAL = 2
    mock_vk.VK_COMPARE_OP_LESS_OR_EQUAL = 3
    mock_vk.VK_COMPARE_OP_GREATER = 4
    mock_vk.VK_COMPARE_OP_NOT_EQUAL = 5
    mock_vk.VK_COMPARE_OP_GREATER_OR_EQUAL = 6
    mock_vk.VK_COMPARE_OP_ALWAYS = 7

    return mock_vk


class VulkanDepthAttachmentStateTests(unittest.TestCase):
    def test_vulkan_depth_image_create_info_and_memory(self):
        mock_vk = _create_mock_vk()
        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            gpu.physical = MagicMock()

            props = MagicMock()
            props.optimalTilingFeatures = mock_vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
            mock_vk.vkGetPhysicalDeviceFormatProperties.return_value = props

            mock_vk.vkCreateImage.return_value = "depth_img_handle"
            mock_vk.vkCreateImageView.return_value = "depth_view_handle"
            mock_vk.vkGetImageMemoryRequirements.return_value = MagicMock(size=4096, memoryTypeBits=1)
            mock_vk.vkAllocateMemory.return_value = "depth_mem_handle"

            def real_view(res, img, fmt, aspect_mask=None):
                return _Vulkan.view(gpu, res, img, fmt, aspect_mask)

            gpu.view = real_view
            gpu.memory_type = MagicMock(return_value=3)

            img, view = _Vulkan.depth_image(gpu, resources, 1024, 768, mock_vk.VK_FORMAT_D32_SFLOAT)

            self.assertEqual(img, "depth_img_handle")
            self.assertEqual(view, "depth_view_handle")

            # Check VkImageCreateInfo kwargs
            img_call_kwargs = mock_vk.VkImageCreateInfo.call_args.kwargs
            self.assertEqual(img_call_kwargs["imageType"], mock_vk.VK_IMAGE_TYPE_2D)
            self.assertEqual(img_call_kwargs["format"], 126)  # VK_FORMAT_D32_SFLOAT
            extent_call_kwargs = mock_vk.VkExtent3D.call_args.kwargs
            self.assertEqual(extent_call_kwargs["width"], 1024)
            self.assertEqual(extent_call_kwargs["height"], 768)
            self.assertEqual(extent_call_kwargs["depth"], 1)
            self.assertEqual(img_call_kwargs["tiling"], mock_vk.VK_IMAGE_TILING_OPTIMAL)
            self.assertEqual(img_call_kwargs["usage"], mock_vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
            self.assertEqual(img_call_kwargs["initialLayout"], mock_vk.VK_IMAGE_LAYOUT_UNDEFINED)

            # Check memory allocation
            alloc_call_kwargs = mock_vk.VkMemoryAllocateInfo.call_args.kwargs
            self.assertEqual(alloc_call_kwargs["allocationSize"], 4096)
            gpu.memory_type.assert_called_once_with(1, mock_vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    def test_vulkan_depth_image_view_aspect_mask(self):
        mock_vk = _create_mock_vk()
        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            mock_vk.vkCreateImageView.return_value = "depth_view_handle"

            view = _Vulkan.view(gpu, resources, "img_handle", 126, aspect_mask=mock_vk.VK_IMAGE_ASPECT_DEPTH_BIT)
            self.assertEqual(view, "depth_view_handle")

            view_info = mock_vk.VkImageViewCreateInfo.call_args.kwargs
            self.assertEqual(view_info["image"], "img_handle")
            self.assertEqual(view_info["viewType"], mock_vk.VK_IMAGE_VIEW_TYPE_2D)
            self.assertEqual(view_info["format"], 126)

            range_info = mock_vk.VkImageSubresourceRange.call_args.kwargs
            self.assertEqual(range_info["aspectMask"], 2)  # VK_IMAGE_ASPECT_DEPTH_BIT == 2
            self.assertEqual(range_info["levelCount"], 1)
            self.assertEqual(range_info["layerCount"], 1)

    def test_vulkan_framebuffer_attachment_order(self):
        mock_vk = _create_mock_vk()
        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            mock_vk.vkCreateFramebuffer.return_value = "fb_handle"

            fb = _Vulkan.framebuffer(gpu, resources, "rp_handle", "color_view_0", 800, 600, depth_view="depth_view_1")
            self.assertEqual(fb, "fb_handle")

            fb_info = mock_vk.VkFramebufferCreateInfo.call_args.kwargs
            self.assertEqual(fb_info["renderPass"], "rp_handle")
            # Index 0 must be color, Index 1 must be depth
            self.assertEqual(fb_info["pAttachments"], ["color_view_0", "depth_view_1"])
            self.assertEqual(fb_info["width"], 800)
            self.assertEqual(fb_info["height"], 600)
            self.assertEqual(fb_info["layers"], 1)

    def test_vulkan_render_pass_and_subpass_depth_attachment(self):
        mock_vk = _create_mock_vk()
        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            gpu.max_width = 4096
            gpu.max_height = 4096

            program = GraphicsProgram(
                Shader("vs", b"\x00" * 16),
                Shader("fs", b"\x00" * 16),
                target="vulkan",
                depth_stencil={
                    "format": "d32_sfloat",
                    "depth_test_enable": True,
                    "depth_write_enable": True,
                    "depth_compare_op": "less_equal",
                },
            )

            mock_vk.vkCreateRenderPass.return_value = "rp_handle"
            mock_vk.vkCreatePipelineLayout.return_value = "layout_handle"
            mock_vk.vkCreateShaderModule.return_value = "module_handle"
            mock_vk.vkCreateGraphicsPipelines.return_value = ["pipe_handle"]

            rp, pipe, layout = _Vulkan.pipeline(gpu, resources, program, 44, 800, 600, False)

            rp_info = mock_vk.VkRenderPassCreateInfo.call_args.kwargs
            attachments = rp_info["pAttachments"]
            self.assertEqual(len(attachments), 2)

            subpass_info = mock_vk.VkSubpassDescription.call_args.kwargs
            color_ref = subpass_info["pColorAttachments"][0]
            depth_ref = subpass_info["pDepthStencilAttachment"]

            # Color is attachment 0, Depth is attachment 1
            ref_calls = mock_vk.VkAttachmentReference.call_args_list
            depth_ref_call = ref_calls[0].kwargs
            color_ref_call = ref_calls[1].kwargs
            self.assertEqual(depth_ref_call["attachment"], 1)
            self.assertEqual(depth_ref_call["layout"], mock_vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            self.assertEqual(color_ref_call["attachment"], 0)
            self.assertEqual(color_ref_call["layout"], mock_vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)

            # Dependencies must synchronize early fragment tests
            dep_calls = mock_vk.VkSubpassDependency.call_args_list
            in_dep = dep_calls[0].kwargs
            out_dep = dep_calls[1].kwargs
            self.assertTrue(in_dep["srcStageMask"] & mock_vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
            self.assertTrue(in_dep["dstStageMask"] & mock_vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT)
            self.assertTrue(in_dep["dstAccessMask"] & mock_vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
            self.assertTrue(out_dep["srcAccessMask"] & mock_vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)

    def test_vulkan_pipeline_depth_stencil_all_compare_ops(self):
        op_mapping = {
            "never": 0,
            "less": 1,
            "equal": 2,
            "less_equal": 3,
            "greater": 4,
            "not_equal": 5,
            "greater_equal": 6,
            "always": 7,
        }
        for op_name, expected_val in op_mapping.items():
            mock_vk = _create_mock_vk()
            with ExitStack() as resources:
                gpu = MagicMock(spec=_Vulkan)
                gpu.vk = mock_vk
                gpu.device = MagicMock()
                gpu.max_width = 4096
                gpu.max_height = 4096

                program = GraphicsProgram(
                    Shader("vs", b"\x00" * 16),
                    Shader("fs", b"\x00" * 16),
                    target="vulkan",
                    depth_stencil={
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": op_name,
                    },
                )
                mock_vk.vkCreateRenderPass.return_value = "rp_handle"
                mock_vk.vkCreatePipelineLayout.return_value = "layout_handle"
                mock_vk.vkCreateShaderModule.return_value = "module_handle"
                mock_vk.vkCreateGraphicsPipelines.return_value = ["pipe_handle"]

                _Vulkan.pipeline(gpu, resources, program, 44, 640, 480, False)

                ds_call_kwargs = mock_vk.VkPipelineDepthStencilStateCreateInfo.call_args.kwargs
                self.assertEqual(ds_call_kwargs["depthTestEnable"], mock_vk.VK_TRUE)
                self.assertEqual(ds_call_kwargs["depthWriteEnable"], mock_vk.VK_TRUE)
                self.assertEqual(ds_call_kwargs["depthCompareOp"], expected_val)
                self.assertEqual(ds_call_kwargs["stencilTestEnable"], mock_vk.VK_FALSE)

    def test_vulkan_pipeline_depth_write_disable(self):
        mock_vk = _create_mock_vk()
        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            gpu.max_width = 4096
            gpu.max_height = 4096

            program = GraphicsProgram(
                Shader("vs", b"\x00" * 16),
                Shader("fs", b"\x00" * 16),
                target="vulkan",
                depth_stencil={
                    "format": "d32_sfloat",
                    "depth_test_enable": True,
                    "depth_write_enable": False,
                    "depth_compare_op": "less_equal",
                },
            )
            mock_vk.vkCreateRenderPass.return_value = "rp_handle"
            mock_vk.vkCreatePipelineLayout.return_value = "layout_handle"
            mock_vk.vkCreateShaderModule.return_value = "module_handle"
            mock_vk.vkCreateGraphicsPipelines.return_value = ["pipe_handle"]

            _Vulkan.pipeline(gpu, resources, program, 44, 640, 480, False)
            ds_call_kwargs = mock_vk.VkPipelineDepthStencilStateCreateInfo.call_args.kwargs
            self.assertEqual(ds_call_kwargs["depthTestEnable"], mock_vk.VK_TRUE)
            self.assertEqual(ds_call_kwargs["depthWriteEnable"], mock_vk.VK_FALSE)

    def test_vulkan_command_clear_values_order_and_data(self):
        mock_vk = _create_mock_vk()
        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            gpu.pool = MagicMock()
            mock_vk.vkAllocateCommandBuffers.return_value = ["cmd_buf"]

            draw = Draw(vertex_count=3)
            cmd = _Vulkan.command(gpu, resources, "rp", "pipe", "fb", 800, 600, draw, has_depth=True)

            begin_info = mock_vk.VkRenderPassBeginInfo.call_args.kwargs
            clear_values = begin_info["pClearValues"]
            self.assertEqual(len(clear_values), 2)

            color_call = mock_vk.VkClearColorValue.call_args.kwargs
            self.assertEqual(color_call["float32"], [0.02, 0.02, 0.04, 1.0])

            depth_call = mock_vk.VkClearDepthStencilValue.call_args.kwargs
            self.assertEqual(depth_call["depth"], 1.0)
            self.assertEqual(depth_call["stencil"], 0)

    def test_vulkan_backward_compatibility_no_depth(self):
        mock_vk = _create_mock_vk()
        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            gpu.max_width = 4096
            gpu.max_height = 4096
            gpu.pool = MagicMock()

            program = GraphicsProgram(
                Shader("vs", b"\x00" * 16),
                Shader("fs", b"\x00" * 16),
                target="vulkan",
                depth_stencil=None,
            )
            mock_vk.vkCreateRenderPass.return_value = "rp_handle"
            mock_vk.vkCreatePipelineLayout.return_value = "layout_handle"
            mock_vk.vkCreateShaderModule.return_value = "module_handle"
            mock_vk.vkCreateGraphicsPipelines.return_value = ["pipe_handle"]

            rp, pipe, layout = _Vulkan.pipeline(gpu, resources, program, 44, 640, 480, False)

            # Render pass attachments must be 1 (color only)
            rp_info = mock_vk.VkRenderPassCreateInfo.call_args.kwargs
            self.assertEqual(len(rp_info["pAttachments"]), 1)

            # Subpass pDepthStencilAttachment must be None
            subpass_info = mock_vk.VkSubpassDescription.call_args.kwargs
            self.assertIsNone(subpass_info["pDepthStencilAttachment"])

            # Pipeline pDepthStencilState must be None
            pipe_info = mock_vk.VkGraphicsPipelineCreateInfo.call_args.kwargs
            self.assertIsNone(pipe_info["pDepthStencilState"])

            # Framebuffer attachments must be [view]
            mock_vk.vkCreateFramebuffer.return_value = "fb_handle"
            fb = _Vulkan.framebuffer(gpu, resources, rp, "color_view", 640, 480, depth_view=None)
            fb_info = mock_vk.VkFramebufferCreateInfo.call_args.kwargs
            self.assertEqual(fb_info["pAttachments"], ["color_view"])

            # Command clear values must be 1 (color only)
            mock_vk.vkAllocateCommandBuffers.return_value = ["cmd_buf"]
            _Vulkan.command(gpu, resources, rp, pipe, fb, 640, 480, Draw(vertex_count=3), has_depth=False)
            begin_info = mock_vk.VkRenderPassBeginInfo.call_args.kwargs
            self.assertEqual(len(begin_info["pClearValues"]), 1)


class DirectXDepthStateTests(unittest.TestCase):
    def test_directx_cli_all_compare_ops(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-directx-view"

        for op in COMPARE_OPS:
            prog = GraphicsProgram(
                Shader("vs", b"DXBC" + b"\x00" * 30),
                Shader("fs", b"DXBC" + b"\x00" * 30),
                target="directx",
                depth_stencil={
                    "format": "d32_sfloat",
                    "depth_test_enable": True,
                    "depth_write_enable": True,
                    "depth_compare_op": op,
                },
            )
            with patch("sys.platform", "win32"):
                with _directx_shaders(prog, toolchain) as (dir_path, args):
                    self.assertIn("--depth", args)
                    idx = args.index("--depth-compare")
                    self.assertEqual(args[idx + 1], op)
                    w_idx = args.index("--depth-write")
                    self.assertEqual(args[w_idx + 1], "1")

    def test_directx_cli_depth_write_false(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-directx-view"

        prog = GraphicsProgram(
            Shader("vs", b"DXBC" + b"\x00" * 30),
            Shader("fs", b"DXBC" + b"\x00" * 30),
            target="directx",
            depth_stencil={
                "format": "d32_sfloat",
                "depth_test_enable": True,
                "depth_write_enable": False,
                "depth_compare_op": "less_equal",
            },
        )
        with patch("sys.platform", "win32"):
            with _directx_shaders(prog, toolchain) as (dir_path, args):
                w_idx = args.index("--depth-write")
                self.assertEqual(args[w_idx + 1], "0")

    def test_directx_cli_backward_compatibility(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-directx-view"

        prog = GraphicsProgram(
            Shader("vs", b"DXBC" + b"\x00" * 30),
            Shader("fs", b"DXBC" + b"\x00" * 30),
            target="directx",
            depth_stencil=None,
        )
        with patch("sys.platform", "win32"):
            with _directx_shaders(prog, toolchain) as (dir_path, args):
                self.assertNotIn("--depth", args)
                self.assertNotIn("--depth-compare", args)
                self.assertNotIn("--depth-write", args)
                self.assertNotIn("--depth-format", args)

    def test_directx_source_contracts(self):
        cpp_source = Path("/tmp/gfxir-131/native/directx_viewer.cpp").read_text()

        # Check compare func parser covers all 8 comparison funcs
        for op in ["less_equal", "less", "equal", "greater", "greater_equal", "not_equal", "always", "never"]:
            self.assertIn(f'L"{op}"', cpp_source, f"Missing {op} in parseD3D12ComparisonFunc")

        # Check D3D12 constants used in parseD3D12ComparisonFunc
        for func in [
            "D3D12_COMPARISON_FUNC_LESS_EQUAL",
            "D3D12_COMPARISON_FUNC_LESS",
            "D3D12_COMPARISON_FUNC_EQUAL",
            "D3D12_COMPARISON_FUNC_GREATER",
            "D3D12_COMPARISON_FUNC_GREATER_EQUAL",
            "D3D12_COMPARISON_FUNC_NOT_EQUAL",
            "D3D12_COMPARISON_FUNC_ALWAYS",
            "D3D12_COMPARISON_FUNC_NEVER",
        ]:
            self.assertIn(func, cpp_source)

        # Check ensureDepthBuffer implementation details
        self.assertIn("DXGI_FORMAT_D32_FLOAT", cpp_source)
        self.assertIn("D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL", cpp_source)
        self.assertIn("clearValue.DepthStencil.Depth = 1.0f;", cpp_source)
        self.assertIn("D3D12_RESOURCE_STATE_DEPTH_WRITE", cpp_source)
        self.assertIn("ClearDepthStencilView", cpp_source)
        self.assertIn("D3D12_CLEAR_FLAG_DEPTH", cpp_source)

        # Check PSO pipeline configuration
        self.assertIn("description.DepthStencilState.DepthEnable = TRUE;", cpp_source)
        self.assertIn("description.DepthStencilState.DepthWriteMask = depthSettings.write ? D3D12_DEPTH_WRITE_MASK_ALL : D3D12_DEPTH_WRITE_MASK_ZERO;", cpp_source)
        self.assertIn("description.DSVFormat = DXGI_FORMAT_D32_FLOAT;", cpp_source)

        # Check backward compatibility when disabled
        self.assertIn("description.DepthStencilState.DepthEnable = FALSE;", cpp_source)
        self.assertIn("description.DSVFormat = DXGI_FORMAT_UNKNOWN;", cpp_source)


class MetalDepthStateTests(unittest.TestCase):
    def test_metal_native_binary_built_and_executable(self):
        binary = Path("/tmp/gfxir-131/build/native/mlir-gpu-metal-view")
        self.assertTrue(binary.exists(), "mlir-gpu-metal-view binary does not exist")
        self.assertTrue(os.access(binary, os.X_OK), "mlir-gpu-metal-view is not executable")

    def test_metal_cli_flag_parsing_empirical(self):
        binary = "/tmp/gfxir-131/build/native/mlir-gpu-metal-view"

        # 1. Reject invalid compare op
        proc = subprocess.run(
            [binary, "--depth", "--depth-compare", "bad_compare_op",
             "--vertex", "/dev/null", "--vertex-entry", "main",
             "--fragment", "/dev/null", "--fragment-entry", "main",
             "--vertex-count", "3"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unsupported depth compare op: bad_compare_op", proc.stderr)

        # 2. Reject unsupported depth format
        proc = subprocess.run(
            [binary, "--depth", "--depth-format", "d16_unorm",
             "--vertex", "/dev/null", "--vertex-entry", "main",
             "--fragment", "/dev/null", "--fragment-entry", "main",
             "--vertex-count", "3"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("Unsupported depth format: d16_unorm", proc.stderr)

        # 3. Accept all 8 valid compare ops without parser error
        for op in COMPARE_OPS:
            proc = subprocess.run(
                [binary, "--depth", "--depth-compare", op, "--depth-format", "d32_sfloat",
                 "--depth-write", "1",
                 "--vertex", "/dev/null", "--vertex-entry", "main",
                 "--fragment", "/dev/null", "--fragment-entry", "main",
                 "--vertex-count", "3"],
                capture_output=True, text=True,
            )
            # The compare op itself must be accepted; the failure should be downstream (e.g. Missing --width)
            self.assertNotIn("Unsupported depth compare op", proc.stderr)
            self.assertIn("Missing --width", proc.stderr)

    def test_metal_source_contracts(self):
        mm_source = Path("/tmp/gfxir-131/native/metal_viewer.mm").read_text()

        # Check all 8 compare function mappings
        for op in ["less_equal", "less", "equal", "greater", "greater_equal", "not_equal", "always", "never"]:
            self.assertIn(f'"{op}"', mm_source)

        for func in [
            "MTLCompareFunctionLessEqual",
            "MTLCompareFunctionLess",
            "MTLCompareFunctionEqual",
            "MTLCompareFunctionGreater",
            "MTLCompareFunctionGreaterEqual",
            "MTLCompareFunctionNotEqual",
            "MTLCompareFunctionAlways",
            "MTLCompareFunctionNever",
        ]:
            self.assertIn(func, mm_source)

        # Check depth format and state setup
        self.assertIn("descriptor.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;", mm_source)
        self.assertIn("depthDesc.depthCompareFunction = depthSettings.compareFunction;", mm_source)
        self.assertIn("depthDesc.depthWriteEnabled = depthSettings.writeEnabled ? YES : NO;", mm_source)
        self.assertIn("[encoder setDepthStencilState:self.depthState];", mm_source)

        # Check offscreen readback depth texture allocation
        self.assertIn("texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float", mm_source)
        self.assertIn("pass.depthAttachment.clearDepth = 1.0;", mm_source)
        self.assertIn("pass.depthAttachment.loadAction = MTLLoadActionClear;", mm_source)
        self.assertIn("pass.depthAttachment.storeAction = MTLStoreActionDontCare;", mm_source)

    def test_metal_cli_backward_compatibility(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-metal-view"

        prog = GraphicsProgram(
            Shader("vs", b"MTLB" + b"\x00" * 20),
            Shader("fs", b"MTLB" + b"\x00" * 20),
            target="metal",
            depth_stencil=None,
        )
        with _metal_libraries(prog, toolchain) as (dir_path, args):
            self.assertNotIn("--depth", args)
            self.assertNotIn("--depth-compare", args)
            self.assertNotIn("--depth-write", args)
            self.assertNotIn("--depth-format", args)


class EndToEndPipelineDepthStateTests(unittest.TestCase):
    def test_pipeline_compilation_with_default_depth_stencil(self):
        @fragment
        def fs() -> Value:
            return vec4(1.0, 0.0, 0.0, 1.0)

        pipe = Pipeline(fullscreen_vertex, fs, Draw(), DepthStencil())
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_manifest = compile_pipeline(pipe, Target.VULKAN, Path(tmp_dir) / "build")
            self.assertTrue(out_manifest.exists())
            data = json.loads(out_manifest.read_text())
            self.assertIn("depth_stencil", data)
            self.assertEqual(data["depth_stencil"], {
                "format": "d32_sfloat",
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_compare_op": "less_equal",
                "clear_depth": 1.0,
            })

            # Check load_program parses depth_stencil
            prog = load_program(out_manifest)
            self.assertIsNotNone(prog.depth_stencil)
            self.assertEqual(prog.depth_stencil["format"], "d32_sfloat")
            self.assertEqual(prog.depth_stencil["depth_compare_op"], "less_equal")

    def test_pipeline_compilation_backward_compatibility_no_depth(self):
        @fragment
        def fs() -> Value:
            return vec4(1.0, 0.0, 0.0, 1.0)

        pipe = Pipeline(fullscreen_vertex, fs, Draw(), depth_stencil=None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_manifest = compile_pipeline(pipe, Target.VULKAN, Path(tmp_dir) / "build")
            data = json.loads(out_manifest.read_text())
            self.assertNotIn("depth_stencil", data)

            prog = load_program(out_manifest)
            self.assertIsNone(prog.depth_stencil)

    def test_pipeline_compilation_custom_depth_dict(self):
        @fragment
        def fs() -> Value:
            return vec4(1.0, 0.0, 0.0, 1.0)

        custom_dict = {
            "format": "d32_sfloat",
            "depth_test_enable": True,
            "depth_write_enable": False,
            "depth_compare_op": "greater",
            "clear_depth": 0.0,
        }
        pipe = Pipeline(fullscreen_vertex, fs, Draw(), depth_stencil=custom_dict)
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_manifest = compile_pipeline(pipe, Target.VULKAN, Path(tmp_dir) / "build")
            data = json.loads(out_manifest.read_text())
            self.assertIn("depth_stencil", data)
            self.assertEqual(data["depth_stencil"], custom_dict)

            prog = load_program(out_manifest)
            self.assertFalse(prog.depth_stencil["depth_write_enable"])
            self.assertEqual(prog.depth_stencil["depth_compare_op"], "greater")
            self.assertEqual(prog.depth_stencil["clear_depth"], 0.0)


if __name__ == "__main__":
    unittest.main()
