"""Shared structured compiler input builders."""

import json
from pathlib import Path


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
