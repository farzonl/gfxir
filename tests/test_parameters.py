import copy
import json
import runpy
import struct
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import (
    CompileError, FrameParameters, ParameterBlock, Pipeline, Target, Type, Vertex,
    compile_pipeline, fragment, fullscreen_vertex, pack_parameters, vec3, vec4, vertex,
)
from mlir_gpu.parameters import verify_parameters
from mlir_gpu.viewer import ViewerError, load_program
from test_native_ir import ROOT, TOOLS, available
import test_structured_input as structured


Material = ParameterBlock("material", (
    ("normal", Type.VEC3), ("roughness", Type.F32),
    ("pair", Type.VEC2), ("weight", Type.F32),
))


@fragment
def material_fragment(material: Material):
    return vec4(material.normal * material.roughness + vec3(material.pair, material.weight), 1)


def example():
    return runpy.run_path(str(ROOT / "examples/frame_bands.py"))["pipeline"]


def member_offsets(binary):
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    offsets = {}
    position = 5
    while position < len(words):
        count, opcode = words[position] >> 16, words[position] & 65535
        if opcode == 72 and words[position + 3] == 35:
            offsets[words[position + 2]] = words[position + 4]
        position += count
    return [offsets[index] for index in sorted(offsets)]


class ParameterAuthoringTests(unittest.TestCase):
    def test_parameters_are_distinct_from_raster_inputs_and_inside_helpers(self):
        pipeline = example()
        stage = pipeline.trace()[1]
        self.assertEqual(len(stage.input_interface), 1)
        self.assertEqual(stage.input_interface[0].builtin, "fragment_coord")
        module = stage.to_dict()
        self.assertEqual(module["parameters"], FrameParameters.to_dict())
        self.assertEqual(sum(op["op"] == "gfx.parameter" for op in structured.operations(module["functions"][0])), 2)

        def color(frame):
            return vec4(frame.time, frame.resolution.x, frame.resolution.y, 1)
        @fragment
        def helper(frame: FrameParameters):
            return color(frame)
        self.assertEqual(Pipeline(fullscreen_vertex, helper).trace()[1].input_interface, ())

    def test_unsupported_parameter_declarations_and_field_access_fail(self):
        for fields in ((("a", Type.U32),), (("a", Type.F32), ("a", Type.VEC2)), (("_private", Type.F32),)):
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                ParameterBlock("invalid", fields)
        @fragment
        def missing(frame: FrameParameters):
            return vec4(frame.missing, 0, 0, 1)
        with self.assertRaisesRegex(AttributeError, r"test_parameters.py:\d+:\d+: Unknown parameter field"):
            Pipeline(fullscreen_vertex, missing).trace()


@unittest.skipUnless(available(), "Build the native compiler for parameter ABI tests")
class ParameterABITests(unittest.TestCase):
    def compile(self, shader, directory, target="vulkan"):
        return structured.StructuredInputTests.compile(self, shader, directory, target)

    def test_host_packing_matches_encoded_spirv_and_llvm_member_offsets(self):
        for pipeline, expected_offsets, expected_size, values, expected_words in (
            (example(), [0, 8], 16, {"time": 1.25, "resolution": (640, 480)}, (1.25, 0, 640, 480)),
            (Pipeline(fullscreen_vertex, material_fragment), [0, 12, 16, 24], 32,
             {"normal": (1, 2, 3), "roughness": 4, "pair": (5, 6), "weight": 7},
             (1, 2, 3, 4, 5, 6, 7, 0)),
        ):
            source = pipeline.trace()[1].to_dict()
            for target in ("vulkan", "directx"):
                with self.subTest(size=expected_size, target=target), tempfile.TemporaryDirectory() as work:
                    directory = Path(work)
                    compiled = self.compile(source, directory, target)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    reflection = json.loads((directory / "reflection.json").read_text())["parameters"]
                    verify_parameters(reflection)
                    self.assertEqual(reflection["size"], expected_size)
                    self.assertEqual([field["offset"] for field in reflection["fields"]], expected_offsets)
                    self.assertEqual(pack_parameters(reflection, values), struct.pack(f"<{len(expected_words)}f", *expected_words))
                    if target == "vulkan":
                        self.assertEqual(member_offsets((directory / "shader.bin").read_bytes()), expected_offsets)
                    else:
                        llvm = (directory / "shader.bin").read_text()
                        self.assertIn("target(\"dx.CBuffer\"", llvm)
                        self.assertIn("!dx.rootsignatures", llvm)
                        for offset in expected_offsets:
                            self.assertRegex(llvm, rf"getpointer[^\n]*i32 {offset}\)")

    def test_invalid_native_declarations_reads_and_stale_inputs_fail(self):
        original = example().trace()[1].to_dict()
        for case in ("missing", "bad_block", "bad_field", "bad_type", "duplicate",
                     "unsupported", "oversized", "authored_offset", "authored_binding", "stale"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as work:
                shader = copy.deepcopy(original)
                op = next(op for op in structured.operations(shader["functions"][0]) if op["op"] == "gfx.parameter")
                if case == "missing":
                    del shader["parameters"]
                elif case == "bad_block":
                    op["attributes"]["block"]["value"] = "missing"
                elif case == "bad_field":
                    op["attributes"]["field"]["value"] = "missing"
                elif case == "bad_type":
                    op["results"][0]["type"] = "f32x3"
                elif case == "duplicate":
                    shader["parameters"]["fields"].append(shader["parameters"]["fields"][0])
                elif case == "unsupported":
                    shader["parameters"]["fields"][0]["type"] = "u32"
                elif case == "oversized":
                    shader["parameters"]["fields"] = [{"name": f"field{i}", "type": "f32x4"} for i in range(9)]
                elif case == "authored_offset":
                    shader["parameters"]["fields"][0]["offset"] = 12
                elif case == "authored_binding":
                    shader["parameters"]["register"] = 2
                else:
                    shader["schema_version"] = 1
                result = self.compile(shader, Path(work))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("parameter", result.stderr)

    def test_host_packing_rejects_malformed_layouts_bindings_and_values(self):
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            self.assertEqual(self.compile(example().trace()[1].to_dict(), directory).returncode, 0)
            block = json.loads((directory / "reflection.json").read_text())["parameters"]
            for case in ("version", "offset", "alignment", "size", "type", "binding", "bool_binding", "stages"):
                with self.subTest(case=case):
                    invalid = copy.deepcopy(block)
                    if case == "version":
                        invalid["schema_version"] = 2
                    elif case in ("offset", "alignment", "type"):
                        invalid["fields"][1][case] = "u32" if case == "type" else 4
                    elif case == "size":
                        invalid["size"] = 32
                    elif case == "stages":
                        invalid["stages"] = [{}]
                    else:
                        invalid["bindings"]["metal"]["buffer_index"] = True if case == "bool_binding" else 3
                    with self.assertRaises(ValueError):
                        pack_parameters(invalid, {"time": 1, "resolution": (640, 480)})
            for values in ({}, {"time": True, "resolution": (640, 480)}, {"time": float("nan"), "resolution": (640, 480)},
                           {"time": 10**400, "resolution": (640, 480)}, {"time": 0, "resolution": (640,)},
                           {"time": 0, "resolution": (640, 480), "extra": 1}):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    pack_parameters(block, values)

    def test_real_binaries_and_compiler_only_viewer_boundary(self):
        for pipeline in (example(), Pipeline(fullscreen_vertex, material_fragment)):
            for target in Target:
                with self.subTest(target=target, shader=pipeline.fragment.function.__name__), tempfile.TemporaryDirectory() as work:
                    manifest = compile_pipeline(pipeline, target, Path(work) / "pipeline", TOOLS)
                    data = json.loads(manifest.read_text())
                    self.assertEqual(data["schema_version"], 4)
                    self.assertEqual(data["parameters"]["stages"], ["fragment"])
                    self.assertEqual(data["parameters"], data["interface"]["fragment"]["parameters"])
                    if target == Target.VULKAN and data["parameters"]["name"] == "frame":
                        self.assertEqual(load_program(manifest).parameters, data["parameters"])
                    else:
                        message = "FrameParameters only" if target == Target.VULKAN else "runtime binding adapters"
                        with self.assertRaisesRegex(ViewerError, message):
                            load_program(manifest)
                    invalid = copy.deepcopy(data)
                    invalid["parameters"]["bindings"]["vulkan"]["offset"] = False
                    manifest.write_text(json.dumps(invalid))
                    with self.assertRaisesRegex(ViewerError, "target bindings"):
                        load_program(manifest)
                    data["schema_version"] = 3
                    manifest.write_text(json.dumps(data))
                    with self.assertRaisesRegex(ViewerError, "schema version 4"):
                        load_program(manifest)

    def test_visibility_merges_and_cross_stage_layout_mismatch_is_atomic(self):
        @vertex
        def frame_vertex(index, frame: FrameParameters):
            return Vertex(fullscreen_vertex.function(index).position.with_components(z=frame.time * 0.01))
        with tempfile.TemporaryDirectory() as work:
            manifest = compile_pipeline(Pipeline(frame_vertex, example().fragment), Target.VULKAN, Path(work) / "pipeline", TOOLS)
            self.assertEqual(json.loads(manifest.read_text())["parameters"]["stages"], ["vertex", "fragment"])

        OtherFrame = ParameterBlock("frame", (("time", Type.VEC2), ("resolution", Type.VEC2)))
        @fragment
        def incompatible(frame: OtherFrame):
            return vec4(frame.time, frame.resolution)
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / "pipeline"
            with self.assertRaisesRegex(ValueError, "matching parameter block"):
                compile_pipeline(Pipeline(frame_vertex, incompatible), Target.VULKAN, output, TOOLS)
            self.assertFalse(output.exists())

    def test_exact_128_byte_limit_and_unreachable_helper_resources(self):
        Limit = ParameterBlock("limit", tuple((f"v{i}", Type.VEC4) for i in range(8)))
        @fragment
        def limit(parameters: Limit):
            return parameters.v0
        with tempfile.TemporaryDirectory() as work:
            manifest = compile_pipeline(Pipeline(fullscreen_vertex, limit), Target.DIRECTX, Path(work) / "pipeline", TOOLS)
            self.assertEqual(json.loads(manifest.read_text())["parameters"]["size"], 128)
        shader = example().trace()[1].to_dict()
        ops = structured.operations(shader["functions"][0])
        read = copy.deepcopy(next(op for op in ops if op["op"] == "gfx.parameter"))
        shader["functions"].append({
            "name": "unused", "results": ["f32"],
            "body": {"blocks": [structured.block([
                read, {"op": "func.return", "args": [read["results"][0]["name"]]},
            ])]},
        })
        output = copy.deepcopy(next(op for op in ops if op["op"] == "gfx.output"))
        output["args"] = ["color"]
        shader["functions"][0]["body"]["blocks"][0]["operations"] = [
            {"op": "arith.constant", "results": [structured.result("color", "f32x4")],
             "attributes": {"value": {"kind": "dense", "type": "f32x4", "value": [1]}}},
            output, {"op": "func.return"},
        ]
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            result = self.compile(shader, directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("parameters", json.loads((directory / "reflection.json").read_text()))
