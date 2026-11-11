import json
import os
import tempfile
import unittest
from pathlib import Path

from mlir_gpu.compiler import CompileError, Target, Toolchain, compile_pipeline
from mlir_gpu.frontend import Draw, Pipeline, Value, Vertex, fragment, select, vec4, vertex
from mlir_gpu.parameters import struct, Type
import mlir_gpu.shader_math as sm

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
class MultiUVNamedVertex:
    position: Type.VEC3
    uv0: Type.VEC2
    uv1: Type.VEC2


@struct(layout="vertex")
class TexcoordNamedVertex:
    position: Type.VEC3
    texcoord0: Type.VEC2
    texcoord1: Type.VEC2


class TestVertexLoweringBackends(unittest.TestCase):
    def test_cross_backend_compilation_and_validation(self):
        @vertex
        def vs_pbr(vert: StandardPbrVertex) -> Vertex:
            p = vert.position + vert.normal * 0.1
            return Vertex(vec4(p, 1.0), varyings={"normal": vert.normal, "uv": vert.uv})

        @fragment
        def fs_pbr(normal: Value, uv: Value) -> Value:
            return vec4(normal.x, normal.y, uv.x, 1.0)

        pipe = Pipeline(vs_pbr, fs_pbr, draw=Draw(indexed=True, index_count=36, index_type="uint16"))

        for target in (Target.VULKAN, Target.DIRECTX, Target.METAL):
            with self.subTest(target=target.name):
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                with tempfile.TemporaryDirectory() as tmp:
                    out = compile_pipeline(pipe, target, Path(tmp) / "out", TOOLS)
                    manifest = json.loads(out.read_text())
                    self.assertIn("input_layout", manifest)
                    self.assertEqual(manifest["input_layout"]["bindings"][0]["stride"], 32)
                    self.assertEqual(len(manifest["input_layout"]["attributes"]), 3)
                    self.assertTrue(manifest["draw"]["indexed"])
                    self.assertEqual(manifest["draw"]["index_type"], "uint16")
                    self.assertEqual(manifest["draw"]["index_count"], 36)

    def test_vertex_shader_conditional_and_derived_expressions(self):
        @vertex
        def vs_dynamic(vert: StandardPbrVertex) -> Vertex:
            is_positive = vert.uv.x > 0.5
            displacement = select(is_positive, 0.2, 0.0)
            n_len = sm.length(vert.normal)
            modified_pos = vert.position + vert.normal * displacement * n_len
            return Vertex(vec4(modified_pos, 1.0), varyings={"uv": vert.uv})

        @fragment
        def fs_dynamic(uv: Value) -> Value:
            return vec4(uv.x, uv.y, 0.0, 1.0)

        pipe = Pipeline(vs_dynamic, fs_dynamic)

        for target in (Target.VULKAN, Target.DIRECTX, Target.METAL):
            with self.subTest(target=target.name):
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                with tempfile.TemporaryDirectory() as tmp:
                    out = compile_pipeline(pipe, target, Path(tmp) / "out", TOOLS)
                    manifest = json.loads(out.read_text())
                    self.assertEqual(manifest["target"], target.value)

    def test_vertex_shader_partial_attribute_access_preserves_layout(self):
        @vertex
        def vs_pos_only_access(vert: StandardPbrVertex) -> Vertex:
            # Vertex shader only reads position, leaving normal and uv unread
            return Vertex(vec4(vert.position, 1.0))

        @fragment
        def fs_simple() -> Value:
            return vec4(1.0, 0.5, 0.25, 1.0)

        pipe = Pipeline(vs_pos_only_access, fs_simple)

        for target in (Target.VULKAN, Target.DIRECTX, Target.METAL):
            with self.subTest(target=target.name):
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                with tempfile.TemporaryDirectory() as tmp:
                    out = compile_pipeline(pipe, target, Path(tmp) / "out", TOOLS)
                    manifest = json.loads(out.read_text())
                    self.assertEqual(manifest["input_layout"]["bindings"][0]["stride"], 32)
                    self.assertEqual(len(manifest["input_layout"]["attributes"]), 3)

    def test_multi_uv_attribute_naming_backend_behavior(self):
        # Multi-UV with uv0/uv1 compiles cleanly across all three backends
        @vertex
        def vs_multi_uv(vert: MultiUVNamedVertex) -> Vertex:
            return Vertex(vec4(vert.position, 1.0), varyings={"uv": vert.uv0 + vert.uv1})

        @fragment
        def fs_multi_uv(uv: Value) -> Value:
            return vec4(uv.x, uv.y, 0.0, 1.0)

        pipe = Pipeline(vs_multi_uv, fs_multi_uv)
        for target in (Target.VULKAN, Target.DIRECTX, Target.METAL):
            with self.subTest(target=target.name):
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                with tempfile.TemporaryDirectory() as tmp:
                    out = compile_pipeline(pipe, target, Path(tmp) / "out", TOOLS)
                    manifest = json.loads(out.read_text())
                    self.assertEqual(len(manifest["input_layout"]["attributes"]), 3)

    def test_texcoord_attribute_duplicate_semantic_hazard(self):
        # Attribute names starting with 'texcoord' collide on semantic 'TEXCOORD' index 0 in Lowering.cpp
        @vertex
        def vs_tex(vert: TexcoordNamedVertex) -> Vertex:
            return Vertex(vec4(vert.position, 1.0), varyings={"uv": vert.texcoord0 + vert.texcoord1})

        @fragment
        def fs_tex(uv: Value) -> Value:
            return vec4(uv.x, uv.y, 0.0, 1.0)

        pipe = Pipeline(vs_tex, fs_tex)

        # Vulkan compiles successfully because SPIR-V uses location indices, not semantic names
        with self.subTest(target="VULKAN"):
            try:
                TOOLS.preflight(Target.VULKAN)
            except CompileError as error:
                self.skipTest(str(error))
            with tempfile.TemporaryDirectory() as tmp:
                out = compile_pipeline(pipe, Target.VULKAN, Path(tmp) / "out", TOOLS)
                self.assertTrue(out.exists())

        # DirectX and Metal fail due to duplicate semantic (TEXCOORD, 0) emitted by Lowering.cpp
        for target in (Target.DIRECTX, Target.METAL):
            with self.subTest(target=target.name):
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                with tempfile.TemporaryDirectory() as tmp:
                    with self.assertRaises(CompileError) as ctx:
                        compile_pipeline(pipe, target, Path(tmp) / "out", TOOLS)
                    self.assertIn("duplicate semantic name and index", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
