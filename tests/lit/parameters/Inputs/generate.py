import copy
import json
import sys
from pathlib import Path

from compiler_inputs import block, operations, result
from programs import pipeline


def generate(case):
    if case in ("frame", "material", "limit", "limit256"):
        return pipeline(case).trace()[1].to_dict()
    shader = pipeline("frame").trace()[1].to_dict()
    ops = operations(shader["functions"][0])
    op = next(op for op in ops if op["op"] == "gfx.parameter")
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
        shader["parameters"]["fields"][0]["type"] = "unsupported"
    elif case == "oversized":
        shader["parameters"]["fields"] = [{"name": f"field{i}", "type": "f32x4"} for i in range(17)]
    elif case == "authored_offset":
        shader["parameters"]["fields"][0]["offset"] = 12
    elif case == "authored_binding":
        shader["parameters"]["register"] = 2
    elif case == "stale":
        shader["schema_version"] = 1
    elif case == "duplicate_block":
        shader["parameters"] = [
            {"name": "b1", "fields": [{"name": "f0", "type": "f32"}]},
            {"name": "b1", "fields": [{"name": "f1", "type": "f32"}]},
        ]
    elif case == "cumulative_oversized":
        shader["parameters"] = [
            {"name": "b1", "fields": [{"name": f"f{i}", "type": "f32x4"} for i in range(10)]},
            {"name": "b2", "fields": [{"name": f"g{i}", "type": "f32x4"} for i in range(7)]},
        ]
    elif case == "unreachable":
        read = copy.deepcopy(op)
        shader["functions"].append({
            "name": "unused", "results": ["f32"],
            "body": {"blocks": [block([
                read, {"op": "func.return", "args": [read["results"][0]["name"]]},
            ])]},
        })
        output = copy.deepcopy(next(op for op in ops if op["op"] == "gfx.output"))
        output["args"] = ["color"]
        shader["functions"][0]["body"]["blocks"][0]["operations"] = [
            {"op": "arith.constant", "results": [result("color", "f32x4")],
             "attributes": {"value": {"kind": "dense", "type": "f32x4", "value": [1]}}},
            output, {"op": "func.return"},
        ]
    else:
        raise ValueError(case)
    return shader


Path(sys.argv[2]).write_text(json.dumps(generate(sys.argv[1]), allow_nan=False))
