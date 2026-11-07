import argparse
import json

from test_compiler import arithmetic_pipeline, triangle


parser = argparse.ArgumentParser()
parser.add_argument("case")
parser.add_argument("--stage", type=int, default=1)
args = parser.parse_args()
shader = triangle().trace()[args.stage].to_dict()
nodes = {
    "broadcast": {"op": "broadcast", "type": "f32x2", "args": [0, 0]},
    "cast": {"op": "cast", "type": "i32", "args": [0]},
    "literal": {"op": "constant", "type": "i32", "args": [], "literal": 2**31},
    "float-predicate": {"op": "cmp", "type": "bool", "args": [1, 1], "literal": "slt"},
    "integer-predicate": {"op": "cmp", "type": "bool", "args": [2, 2], "literal": "olt"},
    "bool-predicate": {"op": "cmp", "type": "bool", "args": [3, 3], "literal": "ult"},
    "div": {"op": "div", "type": "u32", "args": [2, 2]},
    "neg": {"op": "neg", "type": "bool", "args": [3]},
    "sin": {"op": "sin", "type": "u32", "args": [2]},
    "sqrt": {"op": "sqrt", "type": "f32x2", "args": [1]},
    "pow": {"op": "pow", "type": "u32", "args": [2, 2]},
}
if args.case == "reference":
    shader["nodes"][0]["args"] = [999]
elif args.case == "arithmetic":
    shader = arithmetic_pipeline().trace()[args.stage].to_dict()
elif args.case != "triangle":
    shader["nodes"].extend([
        {"op": "constant", "type": "f32", "args": [], "literal": 1.0},
        {"op": "constant", "type": "u32", "args": [], "literal": 1},
        {"op": "constant", "type": "bool", "args": [], "literal": True},
        nodes[args.case],
    ])
print(json.dumps(shader))
