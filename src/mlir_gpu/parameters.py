from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from collections.abc import Mapping

from .frontend import Type, Value


@dataclass(frozen=True)
class ParameterBlock:
    name: str
    fields: tuple[tuple[str, Type], ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError("Parameter blocks require a valid name.")
        if not isinstance(self.fields, tuple) or not self.fields:
            raise TypeError("Parameter fields must be a nonempty tuple of (name, Type) pairs.")
        names = set()
        for field in self.fields:
            if not isinstance(field, tuple) or len(field) != 2:
                raise TypeError("Parameter fields require (name, Type) pairs.")
            name, typ = field
            if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
                raise ValueError("Parameter field names must be valid, public, and unique.")
            if typ not in (Type.F32, Type.VEC2, Type.VEC3, Type.VEC4):
                raise TypeError("Parameter fields currently support f32 and f32x2/3/4.")
            names.add(name)

    def to_dict(self):
        return {"name": self.name, "fields": [
            {"name": name, "type": typ.schema_name} for name, typ in self.fields
        ]}


FrameParameters = ParameterBlock("frame", (("time", Type.F32), ("resolution", Type.VEC2)))


@dataclass(frozen=True)
class _ParameterValues:
    _values: dict[str, Value]

    def __getattr__(self, name):
        if name not in self._values:
            raise AttributeError(f"Unknown parameter field {name!r}.")
        return self._values[name]


def verify_parameters(block: object) -> None:
    keys = {"schema_version", "name", "layout", "size", "alignment", "fields", "stages",
            "ownership", "access", "addressable", "bindings"}
    if not isinstance(block, dict) or set(block) != keys:
        raise ValueError("Parameter reflection requires the complete layout and target bindings.")
    if (type(block["schema_version"]) is not int or block["schema_version"] != 1
            or block["layout"] != "gfx-root-f32-v1" or block["ownership"] != "host"
            or block["access"] != "read-only" or block["addressable"] is not False):
        raise ValueError("Unsupported parameter schema, layout, ownership, or access.")
    if not isinstance(block["name"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", block["name"]):
        raise ValueError("Invalid parameter block name.")
    fields, stages = block["fields"], block["stages"]
    if (not isinstance(fields, list) or not 1 <= len(fields) <= 32
            or not isinstance(stages, list) or not stages
            or any(not isinstance(stage, str) or stage not in ("vertex", "fragment", "compute") for stage in stages)
            or len(set(stages)) != len(stages)):
        raise ValueError("Parameters require typed fields and unique supported stage visibility.")
    cursor, names = 0, set()
    for field in fields:
        if not isinstance(field, dict) or set(field) != {"name", "type", "offset", "size", "alignment"}:
            raise ValueError("Invalid reflected parameter field.")
        name, typ = field["name"], field["type"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in names:
            raise ValueError("Invalid or duplicate parameter field name.")
        components = {"f32": 1, "f32x2": 2, "f32x3": 3, "f32x4": 4}.get(typ) if isinstance(typ, str) else None
        if components is None:
            raise ValueError("Unsupported parameter field type.")
        alignment = 16 if components == 3 else components * 4
        offset = (cursor + alignment - 1) // alignment * alignment
        if any(type(field[key]) is not int for key in ("offset", "size", "alignment")) or (
            field["offset"], field["size"], field["alignment"]
        ) != (offset, components * 4, alignment):
            raise ValueError(f"Parameter field {name!r} has a mismatched offset, size, or alignment.")
        names.add(name)
        cursor = offset + components * 4
    size = (cursor + 15) // 16 * 16
    if (size > 128 or type(block["size"]) is not int or block["size"] != size
            or type(block["alignment"]) is not int or block["alignment"] != 16):
        raise ValueError("Parameter block has a mismatched size/alignment or exceeds 128 bytes.")
    bindings = {
        "vulkan": {"storage": "push_constant", "offset": 0},
        "directx": {"storage": "root_constants", "parameter_index": 0, "register": 0, "space": 0,
                    "num32bit_constants": size // 4, "visibility": "all", "root_signature_version": "1.1"},
        "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": size},
    }
    actual = block["bindings"]
    if not isinstance(actual, dict) or set(actual) != set(bindings):
        raise ValueError("Parameter target bindings do not match the compiler ABI.")
    for target, expected in bindings.items():
        binding = actual[target]
        if (not isinstance(binding, dict) or set(binding) != set(expected)
                or any(type(binding[key]) is not type(value) or binding[key] != value for key, value in expected.items())):
            raise ValueError("Parameter target bindings do not match the compiler ABI.")


def pack_parameters(block: dict, values: Mapping[str, object]) -> bytes:
    verify_parameters(block)
    if not isinstance(values, Mapping) or set(values) != {field["name"] for field in block["fields"]}:
        raise ValueError("Supply exactly the reflected parameter fields.")
    data = bytearray(block["size"])
    for field in block["fields"]:
        value = values[field["name"]]
        components = (value,) if field["type"] == "f32" else value
        if not isinstance(components, (tuple, list)) or len(components) != field["size"] // 4:
            raise ValueError(f"Parameter {field['name']!r} requires matching scalar/vector components.")
        for index, component in enumerate(components):
            if type(component) not in (int, float) or abs(component) > 3.4028234663852886e38 or not math.isfinite(component):
                raise ValueError(f"Parameter {field['name']!r} requires finite f32 numbers.")
            struct.pack_into("<f", data, field["offset"] + index * 4, component)
    return bytes(data)
