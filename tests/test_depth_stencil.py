import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from mlir_gpu import (
    DepthStencil,
    Draw,
    MLIRPipeline,
    Pipeline,
    Type,
    compile_pipeline,
    fragment,
    vertex,
)
from mlir_gpu.compiler import RASTER_CONVENTION, PIPELINE_SCHEMA_VERSION, PARAMETER_SCHEMA_VERSION, Target
from mlir_gpu.interfaces import (
    COMPARE_OPS,
    SUPPORTED_DEPTH_FORMATS,
    verify_depth_stencil,
)
from mlir_gpu.pipeline import DepthStencil as PipelineDepthStencil
from mlir_gpu.viewer import (
    GraphicsProgram,
    Shader,
    ViewerError,
    _directx_shaders,
    _metal_libraries,
    _vulkan_depth_format,
    load_program,
)


class DepthStencilDataclassTests(unittest.TestCase):
    def test_default_construction_and_dictionary(self):
        ds = DepthStencil()
        self.assertEqual(ds.format, "d32_sfloat")
        self.assertTrue(ds.depth_test_enable)
        self.assertTrue(ds.depth_write_enable)
        self.assertEqual(ds.depth_compare_op, "less_equal")
        self.assertEqual(ds.clear_depth, 1.0)

        expected = {
            "format": "d32_sfloat",
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_compare_op": "less_equal",
            "clear_depth": 1.0,
        }
        self.assertEqual(ds.to_dict(), expected)

    def test_custom_values_and_immutability(self):
        ds = DepthStencil(
            format="d32_sfloat",
            depth_test_enable=True,
            depth_write_enable=False,
            depth_compare_op="greater",
            clear_depth=0.0,
        )
        self.assertFalse(ds.depth_write_enable)
        self.assertEqual(ds.depth_compare_op, "greater")
        self.assertEqual(ds.clear_depth, 0.0)

        with self.assertRaises(FrozenInstanceError):
            ds.depth_write_enable = True

    def test_module_reexport(self):
        self.assertIs(DepthStencil, PipelineDepthStencil)


class DepthStencilValidationTests(unittest.TestCase):
    def test_valid_configurations_accepted(self):
        for op in COMPARE_OPS:
            ds = {"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": op}
            verify_depth_stencil(ds)

    def test_clear_depth_range_validation(self):
        verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": 0.0})
        verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": 1.0})
        verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": 0.5})

        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": -0.01})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": 1.01})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": "1.0"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": True})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal", "clear_depth": False})

    def test_reject_non_dict(self):
        with self.assertRaises(ValueError):
            verify_depth_stencil("d32_sfloat")
        with self.assertRaises(ValueError):
            verify_depth_stencil(None)
        with self.assertRaises(ValueError):
            verify_depth_stencil([1, 2, 3])

    def test_reject_unsupported_format(self):
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d24_unorm", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": 126, "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal"})

    def test_reject_non_boolean_flags(self):
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": 1, "depth_write_enable": True, "depth_compare_op": "less_equal"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": "true", "depth_compare_op": "less_equal"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": None, "depth_write_enable": True, "depth_compare_op": "less_equal"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": 0, "depth_compare_op": "less_equal"})

    def test_reject_invalid_compare_op(self):
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "greater_or_equal"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "invalid_op"})
        with self.assertRaises(ValueError):
            verify_depth_stencil({"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": 3})


class PipelineManifestDepthTests(unittest.TestCase):
    def test_pipeline_depth_stencil_field(self):
        @vertex
        def vs():
            return None

        @fragment
        def fs():
            return None

        p_default = Pipeline(vs, fs)
        self.assertIsNone(p_default.depth_stencil)

        p_custom = Pipeline(vs, fs, depth_stencil=DepthStencil(depth_compare_op="less"))
        self.assertIsInstance(p_custom.depth_stencil, DepthStencil)
        self.assertEqual(p_custom.depth_stencil.depth_compare_op, "less")

        p_dict = Pipeline(vs, fs, depth_stencil={"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": False, "depth_compare_op": "always"})
        self.assertIsInstance(p_dict.depth_stencil, DepthStencil)
        self.assertFalse(p_dict.depth_stencil.depth_write_enable)
        self.assertEqual(p_dict.depth_stencil.depth_compare_op, "always")

        with self.assertRaises(TypeError):
            Pipeline(vs, fs, depth_stencil="unsupported")

    def test_mlir_pipeline_depth_stencil_field(self):
        mlp = MLIRPipeline(Path("v.mlir"), Path("f.mlir"), "vs", "fs", depth_stencil=DepthStencil())
        self.assertIsInstance(mlp.depth_stencil, DepthStencil)

        mlp_dict = MLIRPipeline(Path("v.mlir"), Path("f.mlir"), "vs", "fs", depth_stencil={"format": "d32_sfloat", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less_equal"})
        self.assertIsInstance(mlp_dict.depth_stencil, DepthStencil)

        with self.assertRaises(TypeError):
            MLIRPipeline(Path("v.mlir"), Path("f.mlir"), "vs", "fs", depth_stencil=123)

    def test_manifest_depth_stencil_serialization(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "pipeline.json"
            pipe = MLIRPipeline(Path("v.mlir"), Path("f.mlir"), "vs", "fs", depth_stencil=DepthStencil(depth_compare_op="less"))
            manifest = {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "target": "vulkan",
                "entries": [],
                "interface": {"vertex": {}, "fragment": {}},
                "draw": pipe.draw.to_dict(),
                "raster_convention": RASTER_CONVENTION,
                "runtime_verified": False,
                "backend_status": "validated-binary",
            }
            if pipe.depth_stencil is not None:
                manifest["depth_stencil"] = pipe.depth_stencil.to_dict()
            manifest_path.write_text(json.dumps(manifest, indent=2))

            loaded = json.loads(manifest_path.read_text())
            self.assertIn("depth_stencil", loaded)
            self.assertEqual(loaded["depth_stencil"]["depth_compare_op"], "less")
            self.assertEqual(loaded["raster_convention"], RASTER_CONVENTION)

    def test_manifest_depth_stencil_omitted_when_none(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = Path(temp) / "pipeline.json"
            pipe = MLIRPipeline(Path("v.mlir"), Path("f.mlir"), "vs", "fs", depth_stencil=None)
            manifest = {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "target": "vulkan",
                "entries": [],
                "interface": {"vertex": {}, "fragment": {}},
                "draw": pipe.draw.to_dict(),
                "raster_convention": RASTER_CONVENTION,
                "runtime_verified": False,
                "backend_status": "validated-binary",
            }
            if pipe.depth_stencil is not None:
                manifest["depth_stencil"] = pipe.depth_stencil.to_dict()
            manifest_path.write_text(json.dumps(manifest, indent=2))

            loaded = json.loads(manifest_path.read_text())
            self.assertNotIn("depth_stencil", loaded)


class ViewerDepthParsingTests(unittest.TestCase):
    def test_load_program_with_depth_stencil(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = directory / "pipeline.json"
            (directory / "v.spv").write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 28)
            (directory / "f.spv").write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 28)
            data = {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "target": "vulkan",
                "entries": [
                    {"stage": "vertex", "entry_point": "main", "binary": "v.spv"},
                    {"stage": "fragment", "entry_point": "main", "binary": "f.spv"},
                ],
                "interface": {
                    "vertex": {"stage": "vertex", "name": "main", "inputs": [], "outputs": [{"builtin": "position", "type": "f32x4", "name": "p"}]},
                    "fragment": {"stage": "fragment", "name": "main", "inputs": [], "outputs": [{"location": 0, "type": "f32x4", "name": "c", "interpolation": "perspective"}]},
                },
                "draw": {"topology": "triangle-list", "vertex_count": 3},
                "raster_convention": RASTER_CONVENTION,
                "runtime_verified": False,
                "backend_status": "validated-binary",
                "depth_stencil": {
                    "format": "d32_sfloat",
                    "depth_test_enable": True,
                    "depth_write_enable": True,
                    "depth_compare_op": "less_equal",
                    "clear_depth": 1.0,
                },
            }
            manifest.write_text(json.dumps(data))
            prog = load_program(manifest)
            self.assertIsNotNone(prog.depth_stencil)
            self.assertEqual(prog.depth_stencil["format"], "d32_sfloat")
            self.assertEqual(prog.depth_stencil["depth_compare_op"], "less_equal")

    def test_load_program_corrupt_depth_raises(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = directory / "pipeline.json"
            (directory / "v.spv").write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 28)
            (directory / "f.spv").write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 28)
            data = {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "target": "vulkan",
                "entries": [
                    {"stage": "vertex", "entry_point": "main", "binary": "v.spv"},
                    {"stage": "fragment", "entry_point": "main", "binary": "f.spv"},
                ],
                "interface": {
                    "vertex": {"stage": "vertex", "name": "main", "inputs": [], "outputs": [{"builtin": "position", "type": "f32x4", "name": "p"}]},
                    "fragment": {"stage": "fragment", "name": "main", "inputs": [], "outputs": [{"location": 0, "type": "f32x4", "name": "c", "interpolation": "perspective"}]},
                },
                "draw": {"topology": "triangle-list", "vertex_count": 3},
                "raster_convention": RASTER_CONVENTION,
                "runtime_verified": False,
                "backend_status": "validated-binary",
                "depth_stencil": {"format": "unsupported_depth_fmt"},
            }
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ViewerError):
                load_program(manifest)


class VulkanDepthPipelineTests(unittest.TestCase):
    def test_vulkan_depth_format_mapping(self):
        mock_vk = MagicMock()
        mock_vk.VK_FORMAT_D32_SFLOAT = 126
        mock_vk.VK_FORMAT_D16_UNORM = 124
        self.assertEqual(_vulkan_depth_format(mock_vk, "d32_sfloat"), 126)
        self.assertEqual(_vulkan_depth_format(mock_vk, "d16_unorm"), 124)

        with self.assertRaises(ViewerError):
            _vulkan_depth_format(mock_vk, "unsupported_format")

    def test_vulkan_depth_pipeline_state_mock(self):
        from contextlib import ExitStack
        from mlir_gpu.viewer import _Vulkan

        mock_vk = MagicMock()
        mock_vk.VK_FORMAT_D32_SFLOAT = 126
        mock_vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT = 32
        mock_vk.VK_IMAGE_ASPECT_DEPTH_BIT = 2
        mock_vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 3
        mock_vk.VK_IMAGE_ASPECT_COLOR_BIT = 1
        mock_vk.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT = 0x200
        mock_vk.VK_COMPARE_OP_LESS_OR_EQUAL = 3
        mock_vk.VK_TRUE = 1
        mock_vk.VK_FALSE = 0
        mock_vk.VK_SAMPLE_COUNT_1_BIT = 1
        mock_vk.VK_ATTACHMENT_LOAD_OP_CLEAR = 1
        mock_vk.VK_ATTACHMENT_STORE_OP_DONT_CARE = 2
        mock_vk.VK_ATTACHMENT_STORE_OP_STORE = 0
        mock_vk.VK_IMAGE_LAYOUT_UNDEFINED = 0
        mock_vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT = 0x400
        mock_vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT = 0x100
        mock_vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT = 0x100
        mock_vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x400

        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            gpu.physical = MagicMock()
            gpu.max_width = 4096
            gpu.max_height = 4096
            gpu.pool = MagicMock()

            # Test depth image creation
            props = MagicMock()
            props.optimalTilingFeatures = 0x200
            mock_vk.vkGetPhysicalDeviceFormatProperties.return_value = props

            mock_vk.vkCreateImage.return_value = "depth_image_handle"
            mock_vk.vkCreateImageView.return_value = "depth_view_handle"
            mock_vk.vkGetImageMemoryRequirements.return_value = MagicMock(size=1024, memoryTypeBits=1)
            mock_vk.vkAllocateMemory.return_value = "depth_memory_handle"

            # Call real view method
            def real_view(res, img, fmt, aspect_mask=None):
                return _Vulkan.view(gpu, res, img, fmt, aspect_mask)

            gpu.view = real_view
            gpu.memory_type = MagicMock(return_value=0)

            image, view = _Vulkan.depth_image(gpu, resources, 640, 480, mock_vk.VK_FORMAT_D32_SFLOAT)
            self.assertEqual(image, "depth_image_handle")
            self.assertEqual(view, "depth_view_handle")

            # Verify depth aspect mask used in image view
            create_view_call = mock_vk.VkImageViewCreateInfo.call_args
            subresource_call = mock_vk.VkImageSubresourceRange.call_args
            self.assertEqual(subresource_call.kwargs["aspectMask"], mock_vk.VK_IMAGE_ASPECT_DEPTH_BIT)

            # Test framebuffer with depth view
            mock_vk.vkCreateFramebuffer.return_value = "fb_handle"
            fb = _Vulkan.framebuffer(gpu, resources, "render_pass", "color_view", 640, 480, depth_view="depth_view")
            fb_info = mock_vk.VkFramebufferCreateInfo.call_args.kwargs
            self.assertEqual(fb_info["pAttachments"], ["color_view", "depth_view"])

            # Test command clear values with depth
            mock_vk.vkAllocateCommandBuffers.return_value = ["cmd_buffer"]
            draw = Draw(vertex_count=3)
            cmd = _Vulkan.command(gpu, resources, "render_pass", "pipeline", "fb", 640, 480, draw, has_depth=True)
            begin_info = mock_vk.VkRenderPassBeginInfo.call_args.kwargs
            self.assertEqual(len(begin_info["pClearValues"]), 2)

    def test_vulkan_depth_pipeline_state_creation(self):
        from contextlib import ExitStack
        from mlir_gpu.viewer import _Vulkan

        mock_vk = MagicMock()
        mock_vk.VK_FORMAT_D32_SFLOAT = 126
        mock_vk.VK_COMPARE_OP_LESS_OR_EQUAL = 3
        mock_vk.VK_TRUE = 1
        mock_vk.VK_FALSE = 0
        mock_vk.VK_SAMPLE_COUNT_1_BIT = 1
        mock_vk.VK_ATTACHMENT_LOAD_OP_CLEAR = 1
        mock_vk.VK_ATTACHMENT_STORE_OP_DONT_CARE = 2
        mock_vk.VK_ATTACHMENT_STORE_OP_STORE = 0
        mock_vk.VK_IMAGE_LAYOUT_UNDEFINED = 0
        mock_vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL = 3
        mock_vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL = 2
        mock_vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT = 0x400
        mock_vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT = 0x100
        mock_vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT = 0x100
        mock_vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT = 0x400
        mock_vk.VK_PIPELINE_BIND_POINT_GRAPHICS = 2
        mock_vk.VK_SUBPASS_EXTERNAL = ~0
        mock_vk.VK_NULL_HANDLE = None

        with ExitStack() as resources:
            gpu = MagicMock(spec=_Vulkan)
            gpu.vk = mock_vk
            gpu.device = MagicMock()
            gpu.max_width = 4096
            gpu.max_height = 4096

            program = GraphicsProgram(
                Shader("main", b"\x00" * 16),
                Shader("main", b"\x00" * 16),
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
            mock_vk.vkCreateGraphicsPipelines.return_value = ["pipeline_handle"]

            rp, pipe, layout = _Vulkan.pipeline(gpu, resources, program, 44, 640, 480, False)

            # Verify render pass creation had 2 attachments and depth attachment reference
            rp_call = mock_vk.VkRenderPassCreateInfo.call_args.kwargs
            self.assertEqual(len(rp_call["pAttachments"]), 2)
            subpass_call = mock_vk.VkSubpassDescription.call_args.kwargs
            self.assertIsNotNone(subpass_call["pDepthStencilAttachment"])

            # Verify pipeline creation had depth stencil state
            p_call = mock_vk.VkGraphicsPipelineCreateInfo.call_args.kwargs
            self.assertIsNotNone(p_call["pDepthStencilState"])


class NativeViewerCliArgumentTests(unittest.TestCase):
    def test_directx_shaders_cli_arguments(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-directx-view"

        prog_depth = GraphicsProgram(
            Shader("vs", b"DXBC" + b"\x00" * 30),
            Shader("fs", b"DXBC" + b"\x00" * 30),
            target="directx",
            depth_stencil={
                "format": "d32_sfloat",
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_compare_op": "less_equal",
            },
        )
        with patch("sys.platform", "win32"):
            with _directx_shaders(prog_depth, toolchain) as (dir_path, args):
                self.assertIn("--depth", args)
                self.assertIn("--depth-compare", args)
                idx = args.index("--depth-compare")
                self.assertEqual(args[idx + 1], "less_equal")
                self.assertIn("--depth-write", args)
                w_idx = args.index("--depth-write")
                self.assertEqual(args[w_idx + 1], "1")
                self.assertIn("--depth-format", args)
                f_idx = args.index("--depth-format")
                self.assertEqual(args[f_idx + 1], "d32_sfloat")

        prog_no_depth = GraphicsProgram(
            Shader("vs", b"DXBC" + b"\x00" * 30),
            Shader("fs", b"DXBC" + b"\x00" * 30),
            target="directx",
            depth_stencil=None,
        )
        with patch("sys.platform", "win32"):
            with _directx_shaders(prog_no_depth, toolchain) as (dir_path, args):
                self.assertNotIn("--depth", args)

    @unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
    def test_metal_libraries_cli_arguments(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-metal-view"

        prog_depth = GraphicsProgram(
            Shader("vs", b"MTLB" + b"\x00" * 20),
            Shader("fs", b"MTLB" + b"\x00" * 20),
            target="metal",
            depth_stencil={
                "format": "d32_sfloat",
                "depth_test_enable": True,
                "depth_write_enable": False,
                "depth_compare_op": "greater",
            },
        )
        with _metal_libraries(prog_depth, toolchain) as (dir_path, args):
            self.assertIn("--depth", args)
            self.assertIn("--depth-compare", args)
            idx = args.index("--depth-compare")
            self.assertEqual(args[idx + 1], "greater")
            self.assertIn("--depth-write", args)
            w_idx = args.index("--depth-write")
            self.assertEqual(args[w_idx + 1], "0")
            self.assertIn("--depth-format", args)
            f_idx = args.index("--depth-format")
            self.assertEqual(args[f_idx + 1], "d32_sfloat")

        prog_no_depth = GraphicsProgram(
            Shader("vs", b"MTLB" + b"\x00" * 20),
            Shader("fs", b"MTLB" + b"\x00" * 20),
            target="metal",
            depth_stencil=None,
        )
        with _metal_libraries(prog_no_depth, toolchain) as (dir_path, args):
            self.assertNotIn("--depth", args)


if __name__ == "__main__":
    unittest.main()
