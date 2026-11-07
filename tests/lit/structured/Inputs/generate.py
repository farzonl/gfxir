"""Emit structured input variants; expected behavior lives in FileCheck tests."""

import copy
import json
import runpy
import sys
from pathlib import Path

from compiler_inputs import ROOT, block, constant, operations, result, source


def generate(case):
    if case.startswith("schema-"):
        shader = runpy.run_path(str(ROOT / "examples/triangle.py"))["pipeline"].trace()[1].to_dict()
        version = case.removeprefix("schema-")
        if version != "absent":
            shader["schema_version"] = json.loads(version)
        return shader
    if case.startswith("compute"):
        return {
            "schema_version": 2,
            "functions": [{
                "name": "compute", "stage": "compute",
                "workgroup_size": [0 if case == "compute-invalid" else 8, 8, 1],
                "results": [], "body": {"blocks": [block([{"op": "func.return"}])]},
            }],
        }
    shader = source()
    if case == "original":
        return shader
    if case == "reverse":
        shader["functions"].reverse()
        return shader
    ops = operations(shader["functions"][1])
    if case in ("merge", "conflict"):
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
        if case == "conflict":
            duplicate["attributes"]["name"]["value"] = "conflict"
        return shader
    if case == "stage-write":
        output = ops.pop(-2)
        ops.insert(-1, {
            "op": "scf.if", "args": ["false"],
            "regions": [
                {"blocks": [block([output, {"op": "scf.yield"}])]},
                {"blocks": [block([{"op": "scf.yield"}])]},
            ],
        })
        return shader
    if case == "recursive":
        operations(shader["functions"][0])[-2] = {
            "op": "func.call", "args": ["level"], "results": [result("color", "f32x3")],
            "attributes": {"callee": {"kind": "symbol", "value": "palette"}},
        }
        return shader
    if case in ("cfg", "dominance"):
        blocks = [
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
                {"op": "func.return", "args": ["yes_color"]},
            ], name="yes"),
            block([
                {"op": "arith.mulf", "args": ["level", "level"], "results": [result("squared", "f32")]},
                {"op": "vector.broadcast", "args": ["squared"], "results": [result("no_color", "f32x3")]},
                {"op": "func.return", "args": ["no_color"]},
            ], name="no"),
        ]
        if case == "dominance":
            shader.update(source_file="dominance.json", source_line=19)
            blocks[1]["operations"][-1] = {"op": "cf.br", "successors": ["join"]}
            blocks[2]["operations"] = [{"op": "cf.br", "successors": ["join"]}]
            blocks.append(block([{"op": "func.return", "args": ["yes_color"]}], name="join"))
        shader["functions"][0]["body"]["blocks"] = blocks
        return shader
    shader.update(source_file="malformed.json", source_line=37)
    loop = next(op for op in ops if op["op"] == "scf.while")
    call = next(op for op in ops if op["op"] == "func.call")
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
        call["args"] = ["glow"]
    elif case == "missing_callee":
        call["attributes"]["callee"]["value"] = "missing"
    elif case == "wrong_loop_state":
        loop["args"].pop()
    elif case == "wrong_condition":
        before[-1]["args"][0] = "i"
    elif case == "wrong_yield":
        after[-1]["args"].pop()
    elif case == "wrong_if_yield":
        conditional["regions"][0]["blocks"][0]["operations"][-1]["args"] = ["old_glow"]
    elif case == "escaped_region_value":
        call["args"] = ["updated_x"]
    elif case == "wrong_region_shape":
        loop["regions"][0] = "bad"
    elif case == "nonterminator_successor":
        ops[1]["successors"] = ["entry"]
    elif case == "bad_dense_shape":
        next(op for op in ops if op.get("results") == [result("emission", "f32x3")])["attributes"]["value"]["value"] = [1, 2]
    elif case == "bad_enum":
        before[0]["attributes"]["predicate"]["value"] = "not_a_predicate"
    elif case == "bad_location":
        ops[1].update(source_file="malformed.json", source_line=-1)
    else:
        raise ValueError(f"Unknown input variant: {case}")
    return shader


Path(sys.argv[2]).write_text(json.dumps(generate(sys.argv[1]), allow_nan=False))
