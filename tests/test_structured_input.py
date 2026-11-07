import copy
import json
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import CompileError, MLIRPipeline, Target, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_native_ir import TOOLS, available


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/ir/julia.json"


def operations(function):
    return function["body"]["blocks"][0]["operations"]


def source():
    return json.loads(SOURCE.read_text())


def result(name, typ):
    return {"name": name, "type": typ}


def constant(name, typ, value):
    return {
        "op": "arith.constant", "results": [result(name, typ)],
        "attributes": {"value": {"kind": "literal", "type": typ, "value": value}},
    }


def block(ops, arguments=(), name="entry"):
    return {"name": name, "arguments": list(arguments), "operations": ops}


@unittest.skipUnless(available(), "Build the native compiler for structured input tests")
class StructuredInputTests(unittest.TestCase):
    def compile(self, shader, directory, target="vulkan"):
        path = directory / "shader.json"
        path.write_text(json.dumps(shader, allow_nan=False))
        return subprocess.run(
            [TOOLS.find("gfx-compile"), str(path), "--target", target,
             "--output", str(directory / "shader.bin"),
             "--ir-output", str(directory / "original.mlir"),
             "--reflection", str(directory / "reflection.json")],
            text=True, capture_output=True, check=False,
        )

    def test_helper_order_and_structured_ir_are_language_independent(self):
        for reverse in (False, True):
            for target in ("vulkan", "directx"):
                with self.subTest(reverse=reverse, target=target), tempfile.TemporaryDirectory() as work:
                    shader = source()
                    if reverse:
                        shader["functions"].reverse()
                    directory = Path(work)
                    compiled = self.compile(shader, directory, target)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    original = (directory / "original.mlir").read_text()
                    for op in ("call @palette", "scf.while", "scf.if", "scf.condition", "scf.yield"):
                        self.assertIn(op, original)
                    self.assertIn("i32, f32, f32, i1, vector<3xf32>", original)
                    reflection = json.loads((directory / "reflection.json").read_text())
                    self.assertEqual(reflection["name"], "julia")
                    self.assertEqual(reflection["inputs"][0]["builtin"], "fragment_coord")
                    if target == "directx":
                        self.assertNotIn("@palette", (directory / "shader.bin").read_text())

    def test_invalid_shapes_regions_calls_and_properties_are_located_diagnostics(self):
        original = source()
        entry = operations(original["functions"][1])
        loop_index = next(i for i, op in enumerate(entry) if op["op"] == "scf.while")
        call_index = next(i for i, op in enumerate(entry) if op["op"] == "func.call")
        for case in (
            "unknown_operand", "wrong_args_shape", "duplicate_result", "unknown_op",
            "unknown_property", "missing_property", "wrong_property_type",
            "wrong_call_type", "missing_callee", "wrong_loop_state", "wrong_condition",
            "wrong_yield", "wrong_if_yield", "escaped_region_value",
            "wrong_region_shape", "nonterminator_successor", "bad_dense_shape",
            "bad_enum", "bad_location",
        ):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as work:
                shader = copy.deepcopy(original)
                shader.update(source_file="malformed.json", source_line=37)
                ops = operations(shader["functions"][1])
                loop = ops[loop_index]
                before = loop["regions"][0]["blocks"][0]["operations"]
                after = loop["regions"][1]["blocks"][0]["operations"]
                conditional = next(op for op in after if op["op"] == "scf.if")
                if case == "unknown_operand":
                    ops[1]["args"] = ["missing"]
                elif case == "wrong_args_shape":
                    ops[1]["args"] = {}
                elif case == "duplicate_result":
                    ops[1]["results"][0]["name"] = "pixel"
                elif case == "unknown_op":
                    ops[1]["op"] = "gfx.made_up"
                elif case == "unknown_property":
                    ops[1]["attributes"]["bogus"] = {"kind": "string", "value": "bad"}
                elif case == "missing_property":
                    ops[3]["attributes"] = {}
                elif case == "wrong_property_type":
                    ops[1]["attributes"]["static_position"] = {"kind": "string", "value": "bad"}
                elif case == "wrong_call_type":
                    ops[call_index]["args"] = ["glow"]
                elif case == "missing_callee":
                    ops[call_index]["attributes"]["callee"]["value"] = "missing"
                elif case == "wrong_loop_state":
                    loop["args"].pop()
                elif case == "wrong_condition":
                    before[-1]["args"][0] = "i"
                elif case == "wrong_yield":
                    after[-1]["args"].pop()
                elif case == "wrong_if_yield":
                    conditional["regions"][0]["blocks"][0]["operations"][-1]["args"] = ["old_glow"]
                elif case == "escaped_region_value":
                    ops[call_index]["args"] = ["updated_x"]
                elif case == "wrong_region_shape":
                    loop["regions"][0] = "bad"
                elif case == "nonterminator_successor":
                    ops[1]["successors"] = ["entry"]
                elif case == "bad_dense_shape":
                    next(op for op in ops if op.get("results") == [result("emission", "f32x3")])["attributes"]["value"]["value"] = [1, 2]
                elif case == "bad_enum":
                    before[0]["attributes"]["predicate"]["value"] = "not_a_predicate"
                else:
                    ops[1].update(source_file="malformed.json", source_line=-1)
                directory = Path(work)
                compiled = self.compile(shader, directory)
                self.assertEqual(compiled.returncode, 1, compiled.stderr)
                self.assertIn('malformed.json":37:1', compiled.stderr)
                self.assertIn("error:", compiled.stderr)
                self.assertFalse((directory / "shader.bin").exists())

    def test_schema_versions_preserve_flat_graphs_and_reject_unknown_versions(self):
        flat = runpy.run_path(str(ROOT / "examples/triangle.py"))["pipeline"].trace()[1].to_dict()
        for version in (None, 1, 2, 3, True, "2"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as work:
                shader = copy.deepcopy(flat)
                if version is not None:
                    shader["schema_version"] = version
                compiled = self.compile(shader, Path(work))
                self.assertEqual(compiled.returncode, 0 if version is None or type(version) is int and version == 1 else 1,
                                 compiled.stderr)
                if version not in (None, 1, 2):
                    self.assertIn("migrate", compiled.stderr)

    def test_matching_input_reads_merge_and_dead_inputs_do_not_shift_reflection(self):
        shader = source()
        ops = operations(shader["functions"][1])
        attrs = {
            "location": {"kind": "literal", "type": "i32", "value": 3},
            "name": {"kind": "string", "value": "uv"},
            "interpolation": {"kind": "interpolation", "value": "perspective"},
        }
        ops[0] = {"op": "gfx.input", "results": [result("pixel", "f32x2")], "attributes": attrs}
        duplicate = copy.deepcopy(ops[0])
        duplicate["results"][0]["name"] = "duplicate_read"
        dead = copy.deepcopy(duplicate)
        dead["results"][0]["name"] = "dead_read"
        dead["attributes"]["location"]["value"] = 0
        dead["attributes"]["name"]["value"] = "unused"
        ops[1:1] = [dead, duplicate]
        for target in ("vulkan", "directx"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as work:
                directory = Path(work)
                compiled = self.compile(shader, directory, target)
                self.assertEqual(compiled.returncode, 0, compiled.stderr)
                reflected = json.loads((directory / "reflection.json").read_text())
                self.assertEqual(reflected["inputs"], [
                    {"name": "uv", "type": "f32x2", "location": 3, "interpolation": "perspective"},
                ])
        duplicate["attributes"]["name"]["value"] = "conflict"
        with tempfile.TemporaryDirectory() as work:
            compiled = self.compile(shader, Path(work))
            self.assertEqual(compiled.returncode, 1, compiled.stderr)
            self.assertIn("conflicting declarations", compiled.stderr)

    def test_stage_writes_in_regions_are_rejected_instead_of_being_hoisted(self):
        shader = source()
        ops = operations(shader["functions"][1])
        output = ops.pop(-2)
        ops.insert(-1, {
            "op": "scf.if", "args": ["false"],
            "regions": [
                {"blocks": [block([output, {"op": "scf.yield"}])]},
                {"blocks": [block([{"op": "scf.yield"}])]},
            ],
        })
        with tempfile.TemporaryDirectory() as work:
            compiled = self.compile(shader, Path(work))
            self.assertEqual(compiled.returncode, 1, compiled.stderr)
            self.assertIn("stage IO must be in the entry block", compiled.stderr)

    def test_cf_is_registered_and_verifiable_but_raw_cfg_target_limits_remain_explicit(self):
        shader = source()
        palette = shader["functions"][0]
        palette["body"]["blocks"] = [
            block([
                constant("threshold", "f32", 0.5),
                {"op": "arith.cmpf", "args": ["level", "threshold"], "results": [result("condition", "bool")],
                 "attributes": {"predicate": {"kind": "cmpf", "value": "ogt"}}},
                {"op": "cf.cond_br", "args": ["condition"], "successors": ["yes", "no"],
                 "attributes": {"operandSegmentSizes": {"kind": "i32_array", "value": [1, 0, 0]}}},
            ], [result("level", "f32")]),
            block([
                {"op": "vector.broadcast", "args": ["level"], "results": [result("yes_color", "f32x3")]},
                {"op": "func.return", "args": ["yes_color"]},
            ], name="yes"),
            block([
                {"op": "arith.mulf", "args": ["level", "level"], "results": [result("squared", "f32")]},
                {"op": "vector.broadcast", "args": ["squared"], "results": [result("no_color", "f32x3")]},
                {"op": "func.return", "args": ["no_color"]},
            ], name="no"),
        ]
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            compiled = self.compile(shader, directory)
            self.assertEqual(compiled.returncode, 1, compiled.stderr)
            self.assertIn("structured body", compiled.stderr)
            self.assertTrue((directory / "original.mlir").exists(), compiled.stderr)
            verified = subprocess.run(
                [TOOLS.find("gfx-opt"), str(directory / "original.mlir"), "--gfx-verify"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertIn("cf.cond_br", verified.stdout)

    def test_compute_stage_contract_is_preserved_for_structured_input(self):
        shader = {
            "schema_version": 2,
            "functions": [{
                "name": "compute", "stage": "compute", "workgroup_size": [8, 8, 1],
                "results": [], "body": {"blocks": [block([{"op": "func.return"}])]},
            }],
        }
        for size, expected in (([8, 8, 1], "graphics-first"), ([0, 8, 1], "three positive")):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as work:
                shader["functions"][0]["workgroup_size"] = size
                compiled = self.compile(shader, Path(work))
                self.assertEqual(compiled.returncode, 1, compiled.stderr)
                self.assertIn(expected, compiled.stderr)

    def test_block_dominance_is_checked_by_upstream_mlir(self):
        shader = source()
        shader.update(source_file="dominance.json", source_line=19)
        shader["functions"][0]["body"]["blocks"] = [
            block([
                constant("threshold", "f32", 0.5),
                {"op": "arith.cmpf", "args": ["level", "threshold"],
                 "results": [result("condition", "bool")],
                 "attributes": {"predicate": {"kind": "cmpf", "value": "ogt"}}},
                {"op": "cf.cond_br", "args": ["condition"], "successors": ["yes", "no"],
                 "attributes": {"operandSegmentSizes": {"kind": "i32_array", "value": [1, 0, 0]}}},
            ], [result("level", "f32")]),
            block([
                {"op": "vector.broadcast", "args": ["level"], "results": [result("yes_color", "f32x3")]},
                {"op": "cf.br", "successors": ["join"]},
            ], name="yes"),
            block([{"op": "cf.br", "successors": ["join"]}], name="no"),
            block([{"op": "func.return", "args": ["yes_color"]}], name="join"),
        ]
        with tempfile.TemporaryDirectory() as work:
            compiled = self.compile(shader, Path(work))
            self.assertEqual(compiled.returncode, 1, compiled.stderr)
            self.assertIn("does not dominate", compiled.stderr)
            self.assertIn('dominance.json":19:1', compiled.stderr)

    def test_recursive_structured_calls_are_rejected_before_backend_conversion(self):
        shader = source()
        ops = operations(shader["functions"][0])
        ops[-2] = {
            "op": "func.call", "args": ["level"], "results": [result("color", "f32x3")],
            "attributes": {"callee": {"kind": "symbol", "value": "palette"}},
        }
        for target in ("vulkan", "directx"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as work:
                compiled = self.compile(shader, Path(work), target)
                self.assertEqual(compiled.returncode, 1, compiled.stderr)
                self.assertIn("recursive shader calls", compiled.stderr)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable Vulkan and Metal GPU tests for structured input",
)
class StructuredInputRenderingTests(unittest.TestCase):
    def render(self, manifest, target, width=128, height=128):
        program = load_program(manifest)
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        if target == Target.VULKAN:
            return render_offscreen(program, width, height, validation)
        return render_metal_offscreen(program, width, height, validation, TOOLS)

    def test_julia_iteration_state_helper_and_escape_match_all_targets(self):
        pipeline = runpy.run_path(str(ROOT / "examples/julia.py"))["pipeline"]
        images = {}
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                manifest = compile_pipeline(pipeline, target, Path(work) / target.value, TOOLS)
                images[target] = self.render(manifest, target).rgba
                for px, py in ((0, 0), (64, 64), (20, 64), (60, 50), (100, 20)):
                    x, y = (px + 0.5 - 64) / 48, (py + 0.5 - 64) / 48
                    count, escaped = 0, False
                    while count < 32 and not escaped:
                        xx, yy = x * x, y * y
                        escaped = xx + yy > 4
                        if not escaped:
                            x, y = xx - yy - 0.7, 2 * x * y + 0.27
                        count += 1
                    level = count / 32
                    expected = (level, level * level + count * 0.003, level * 0.5 + count * 0.008, 1)
                    offset = (py * 128 + px) * 4
                    for actual, channel in zip(images[target][offset:offset + 4], expected):
                        self.assertAlmostEqual(actual, round(min(channel, 1) * 255), delta=2)
            for target in (Target.DIRECTX, Target.METAL):
                error = max(abs(a - b) for a, b in zip(images[Target.VULKAN], images[target]))
                self.assertLessEqual(error, 2)

    def test_structured_for_zero_trip_and_carried_values_execute_on_all_targets(self):
        for upper in (0, 4):
            shader = {
                "schema_version": 2, "functions": [{
                    "name": "loop", "stage": "fragment", "results": [],
                    "body": {"blocks": [block([
                        constant("lower", "index", 0), constant("upper", "index", upper),
                        constant("step", "index", 1), constant("zero", "f32", 0.0),
                        constant("increment", "f32", 0.125),
                        {"op": "scf.for", "args": ["lower", "upper", "step", "zero"],
                         "results": [result("sum", "f32")],
                         "regions": [{"blocks": [block([
                             {"op": "arith.addf", "args": ["carried", "increment"], "results": [result("next", "f32")]},
                             {"op": "scf.yield", "args": ["next"]},
                         ], [result("i", "index"), result("carried", "f32")])]}]},
                        {"op": "vector.broadcast", "args": ["sum"], "results": [result("color", "f32x4")]},
                        {"op": "gfx.output", "args": ["color"],
                         "attributes": operations(source()["functions"][1])[-2]["attributes"]},
                        {"op": "func.return"},
                    ])]},
                }],
            }
            with tempfile.TemporaryDirectory() as work:
                directory = Path(work)
                fragment = directory / "loop.json"
                fragment.write_text(json.dumps(shader))
                pipeline = MLIRPipeline(ROOT / "examples/ir/fullscreen.mlir", fragment, "fullscreen", "loop")
                for target in Target:
                    with self.subTest(upper=upper, target=target.value):
                        manifest = compile_pipeline(pipeline, target, directory / target.value, TOOLS)
                        self.assertEqual(self.render(manifest, target, 1, 1).rgba,
                                         bytes([round(upper * 0.125 * 255)] * 4))

    def test_structured_entry_mismatch_leaves_no_published_artifacts(self):
        pipeline = MLIRPipeline(ROOT / "examples/ir/fullscreen.mlir", SOURCE, "fullscreen", "wrong")
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / "output"
            with self.assertRaisesRegex(CompileError, "Reflected entry"):
                compile_pipeline(pipeline, Target.VULKAN, output, TOOLS)
            self.assertFalse(output.exists())
