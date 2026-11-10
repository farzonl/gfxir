from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Union

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


# =============================================================================
# std430 Aggregate Types, Layout, Reflection, and C Header Emission
# =============================================================================

@dataclass(frozen=True)
class ArrayType:
    element_type: Any  # Type | ArrayType | StructType
    count: int

    def __post_init__(self):
        if not isinstance(self.count, int) or self.count <= 0:
            raise ValueError("Array count must be a positive integer.")


@dataclass(frozen=True)
class StructType:
    name: str
    fields: list[AggregateField] | tuple[AggregateField, ...]

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Struct name must be a non-empty string.")


AggregateType = Union[Type, ArrayType, StructType]


@dataclass(frozen=True)
class AggregateField:
    name: str
    type: AggregateType

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.name):
            raise ValueError(f"Invalid field name: {self.name!r}.")


@dataclass(frozen=True)
class FieldLayout:
    name: str
    type: Any
    offset: int
    size: int
    alignment: int
    stride: int = 0
    count: int = 0
    nested_fields: tuple[FieldLayout, ...] = ()


@dataclass(frozen=True)
class StructLayout:
    name: str
    fields: tuple[FieldLayout, ...]
    size: int
    alignment: int

    def to_reflection_dict(self, name: str | None = None, stage: str = "compute") -> dict:
        block_name = name or self.name

        def _field_to_refl(f: FieldLayout) -> dict:
            if isinstance(f.type, ArrayType):
                elem_t = f.type.element_type
                if isinstance(elem_t, StructType):
                    return {
                        "alignment": f.alignment,
                        "count": f.count,
                        "element_name": elem_t.name,
                        "element_type": "struct",
                        "fields": [_field_to_refl(sub) for sub in f.nested_fields],
                        "name": f.name,
                        "offset": f.offset,
                        "size": f.size,
                        "stride": f.stride,
                        "type": "array",
                    }
                elif isinstance(elem_t, Type):
                    elem_name = elem_t.schema_name
                elif isinstance(elem_t, str):
                    elem_name = elem_t
                else:
                    elem_name = getattr(elem_t, "schema_name", str(elem_t))
                return {
                    "alignment": f.alignment,
                    "count": f.count,
                    "element_type": elem_name,
                    "name": f.name,
                    "offset": f.offset,
                    "size": f.size,
                    "stride": f.stride,
                    "type": "array",
                }
            elif isinstance(f.type, StructType):
                return {
                    "alignment": f.alignment,
                    "fields": [_field_to_refl(sub) for sub in f.nested_fields],
                    "name": f.name,
                    "offset": f.offset,
                    "size": f.size,
                    "struct_name": f.type.name,
                    "type": "struct",
                }
            else:
                type_str = f.type.schema_name if isinstance(f.type, Type) else str(f.type)
                return {
                    "alignment": f.alignment,
                    "name": f.name,
                    "offset": f.offset,
                    "size": f.size,
                    "type": type_str,
                }

        fields_json = [_field_to_refl(f) for f in self.fields]
        return {
            "schema_version": 2,
            "name": block_name,
            "layout": "gfx-std430-v1",
            "size": self.size,
            "alignment": self.alignment,
            "fields": fields_json,
            "stages": [stage],
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "bindings": {
                "vulkan": {"storage": "storage_buffer", "set": 0, "binding": 0},
                "directx": {"storage": "structured_buffer", "register": 0, "space": 0},
                "metal": {"storage": "buffer", "buffer_index": 0},
            },
        }


def _type_size_align(t: Any) -> tuple[int, int]:
    if isinstance(t, Type):
        mapping = {
            Type.F32: (4, 4),
            Type.U32: (4, 4),
            Type.I32: (4, 4),
            Type.F16: (2, 2),
            Type.U16: (2, 2),
            Type.I16: (2, 2),
            Type.F64: (8, 8),
            Type.U64: (8, 8),
            Type.I64: (8, 8),
            Type.U8: (1, 1),
            Type.I8: (1, 1),
            Type.BOOL: (1, 1),
            Type.VEC2: (8, 8),
            Type.VEC3: (12, 16),
            Type.VEC4: (16, 16),
        }
        if t in mapping:
            return mapping[t]
        raise ValueError(f"Unsupported Type: {t}")
    if isinstance(t, ArrayType):
        elem_size, elem_align = _type_size_align(t.element_type)
        stride = (elem_size + elem_align - 1) // elem_align * elem_align
        return (t.count * stride, elem_align)
    if isinstance(t, StructType):
        max_align = max((_type_size_align(f.type)[1] for f in t.fields), default=1)
        cursor = 0
        for f in t.fields:
            sz, al = _type_size_align(f.type)
            offset = (cursor + al - 1) // al * al
            cursor = offset + sz
        size = (cursor + max_align - 1) // max_align * max_align
        return (size, max_align)
    if isinstance(t, str):
        str_map = {
            "f32": (4, 4), "i32": (4, 4), "u32": (4, 4),
            "f16": (2, 2), "i16": (2, 2), "u16": (2, 2),
            "f64": (8, 8), "i64": (8, 8), "u64": (8, 8),
            "i8": (1, 1), "u8": (1, 1), "bool": (1, 1),
            "f32x2": (8, 8), "f32x3": (12, 16), "f32x4": (16, 16),
            "i32x2": (8, 8), "i32x3": (12, 16), "i32x4": (16, 16),
            "u32x2": (8, 8), "u32x3": (12, 16), "u32x4": (16, 16),
        }
        if t in str_map:
            return str_map[t]
    raise ValueError(f"Unknown type for std430 layout: {t!r}")


def compute_std430_layout(fields: list[AggregateField] | tuple[AggregateField, ...], struct_name: str = "Root") -> StructLayout:
    cursor = 0
    max_align = 1
    field_layouts: list[FieldLayout] = []

    for f in fields:
        if isinstance(f.type, StructType):
            sub_layout = compute_std430_layout(f.type.fields, f.type.name)
            f_align = sub_layout.alignment
            f_size = sub_layout.size
            offset = (cursor + f_align - 1) // f_align * f_align
            field_layouts.append(FieldLayout(
                name=f.name,
                type=f.type,
                offset=offset,
                size=f_size,
                alignment=f_align,
                nested_fields=sub_layout.fields,
            ))
            cursor = offset + f_size
            max_align = max(max_align, f_align)
        elif isinstance(f.type, ArrayType):
            elem_nested = ()
            if isinstance(f.type.element_type, StructType):
                sub_layout = compute_std430_layout(f.type.element_type.fields, f.type.element_type.name)
                elem_sz = sub_layout.size
                elem_al = sub_layout.alignment
                elem_nested = sub_layout.fields
            else:
                elem_sz, elem_al = _type_size_align(f.type.element_type)
            stride = (elem_sz + elem_al - 1) // elem_al * elem_al
            f_size = f.type.count * stride
            f_align = elem_al
            offset = (cursor + f_align - 1) // f_align * f_align
            field_layouts.append(FieldLayout(
                name=f.name,
                type=f.type,
                offset=offset,
                size=f_size,
                alignment=f_align,
                stride=stride,
                count=f.type.count,
                nested_fields=elem_nested,
            ))
            cursor = offset + f_size
            max_align = max(max_align, f_align)
        else:
            f_size, f_align = _type_size_align(f.type)
            offset = (cursor + f_align - 1) // f_align * f_align
            field_layouts.append(FieldLayout(
                name=f.name,
                type=f.type,
                offset=offset,
                size=f_size,
                alignment=f_align,
            ))
            cursor = offset + f_size
            max_align = max(max_align, f_align)

    total_size = (cursor + max_align - 1) // max_align * max_align
    return StructLayout(
        name=struct_name,
        fields=tuple(field_layouts),
        size=total_size,
        alignment=max_align,
    )


def _dict_to_field_layout(f: dict) -> FieldLayout:
    f_type_str = f.get("type")
    f_name = f.get("name", "field")
    f_offset = f.get("offset", 0)
    f_size = f.get("size", 0)
    f_align = f.get("alignment", 1)
    f_stride = f.get("stride", 0)
    f_count = f.get("count", 0)
    nested_fields = ()
    if f_type_str == "struct":
        nested_fields = tuple(_dict_to_field_layout(sub) for sub in f.get("fields", []))
        struct_name = f.get("struct_name") or f.get("element_name") or f_name.capitalize()
        agg_fields = [AggregateField(sub.name, sub.type) for sub in nested_fields]
        actual_type = StructType(struct_name, agg_fields)
    elif f_type_str == "array":
        elem_type_val = f.get("element_type", "f32")
        if elem_type_val == "struct" or "fields" in f:
            nested_fields = tuple(_dict_to_field_layout(sub) for sub in f.get("fields", []))
            elem_name = f.get("element_name") or f.get("struct_name") or f"{f_name}_elem".capitalize()
            agg_fields = [AggregateField(sub.name, sub.type) for sub in nested_fields]
            actual_type = ArrayType(StructType(elem_name, agg_fields), f_count)
        else:
            actual_type = ArrayType(elem_type_val, f_count)
    else:
        actual_type = f_type_str

    return FieldLayout(
        name=f_name,
        type=actual_type,
        offset=f_offset,
        size=f_size,
        alignment=f_align,
        stride=f_stride,
        count=f_count,
        nested_fields=nested_fields,
    )


def generate_c_header(layout: StructLayout | dict, struct_name: str | None = None) -> str:
    if isinstance(layout, dict):
        sname = struct_name or layout.get("name", "Root")
        f_list = [_dict_to_field_layout(f) for f in layout.get("fields", [])]
        struct_layout = StructLayout(
            name=sname,
            fields=tuple(f_list),
            size=layout["size"],
            alignment=layout["alignment"],
        )
    else:
        struct_layout = layout
        if struct_name:
            struct_layout = StructLayout(
                name=struct_name,
                fields=layout.fields,
                size=layout.size,
                alignment=layout.alignment,
            )

    upper_name = re.sub(r"[^A-Za-z0-9_]", "_", struct_layout.name.upper())

    scalar_map = {
        "f32": "float", "i32": "int32_t", "u32": "uint32_t",
        "f16": "uint16_t", "i16": "int16_t", "u16": "uint16_t",
        "f64": "double", "i64": "int64_t", "u64": "uint64_t",
        "i8": "int8_t", "u8": "uint8_t", "bool": "uint8_t",
        "f32x2": "float", "f32x3": "float", "f32x4": "float",
        "i32x2": "int32_t", "i32x3": "int32_t", "i32x4": "int32_t",
        "u32x2": "uint32_t", "u32x3": "uint32_t", "u32x4": "uint32_t",
    }

    def _c_type_declaration(typ: Any, name: str, count: int = 0) -> str:
        if isinstance(typ, StructType):
            return f"{typ.name} {name}[{count}]" if count > 0 else f"{typ.name} {name}"
        if isinstance(typ, ArrayType):
            return _c_type_declaration(typ.element_type, name, typ.count)

        if isinstance(typ, Type):
            t_schema = typ.schema_name
        elif isinstance(typ, str):
            t_schema = typ
        else:
            t_schema = str(typ)

        if count > 0:
            if t_schema in ("f32", "i32", "u32", "f16", "i16", "u16", "f64", "i64", "u64", "i8", "u8", "bool"):
                base = scalar_map[t_schema]
                return f"{base} {name}[{count}]"
            elif t_schema in ("f32x3", "i32x3", "u32x3"):
                base_elem = {"f32x3": "float", "i32x3": "int32_t", "u32x3": "uint32_t"}[t_schema]
                return f"struct {{ {base_elem} v[3]; uint8_t _p[4]; }} {name}[{count}]"
            elif t_schema in ("f32x2", "f32x4", "i32x2", "i32x4", "u32x2", "u32x4"):
                base = scalar_map[t_schema]
                dim = int(t_schema.split("x")[-1])
                return f"{base} {name}[{count}][{dim}]"
            else:
                return f"{t_schema} {name}[{count}]"
        else:
            if t_schema in ("f32", "i32", "u32", "f16", "i16", "u16", "f64", "i64", "u64", "i8", "u8", "bool"):
                return f"{scalar_map[t_schema]} {name}"
            elif t_schema in ("f32x2", "f32x3", "f32x4", "i32x2", "i32x3", "i32x4", "u32x2", "u32x3", "u32x4"):
                base = scalar_map[t_schema]
                dim = int(t_schema.split("x")[-1])
                return f"{base} {name}[{dim}]"
            return f"{t_schema} {name}"

    def _emit_struct_body(fields: tuple[FieldLayout, ...], total_size: int, pad_start: int = 0) -> tuple[list[str], int]:
        lines = []
        cursor = 0
        pad_idx = pad_start
        for f in fields:
            if f.offset > cursor:
                gap = f.offset - cursor
                lines.append(f"    uint8_t _pad{pad_idx}[{gap}];")
                pad_idx += 1
            if f.count > 0:
                decl = _c_type_declaration(f.type, f.name, f.count)
            else:
                decl = _c_type_declaration(f.type, f.name)
            lines.append(f"    {decl};")
            cursor = f.offset + f.size
        if cursor < total_size:
            gap = total_size - cursor
            lines.append(f"    uint8_t _pad{pad_idx}[{gap}];")
            pad_idx += 1
        return lines, pad_idx

    header_lines = [
        "/* Auto-generated by gfxir std430 reflection engine. Do not edit. */",
        f"#ifndef {upper_name}_H",
        f"#define {upper_name}_H",
        "",
        "#pragma once",
        "",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <assert.h>",
        "",
        "#if !defined(__cplusplus) && !defined(static_assert)",
        "#  if defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L",
        "#    define static_assert _Static_assert",
        "#  endif",
        "#endif",
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
    ]

    # Collect all StructTypes recursively in topological order (dependencies first)
    collected_structs: list[StructType] = []
    seen_structs: set[str] = {struct_layout.name}

    def _collect_struct_types(typ: Any):
        if isinstance(typ, StructType):
            for f in typ.fields:
                _collect_struct_types(f.type)
            if typ.name not in seen_structs:
                seen_structs.add(typ.name)
                collected_structs.append(typ)
        elif isinstance(typ, ArrayType):
            _collect_struct_types(typ.element_type)

    for f in struct_layout.fields:
        _collect_struct_types(f.type)

    for st in collected_structs:
        st_layout = compute_std430_layout(st.fields, st.name)
        header_lines.append(f"typedef struct {st.name} {{")
        body_lines, _ = _emit_struct_body(st_layout.fields, st_layout.size)
        header_lines.extend(body_lines)
        header_lines.append(f"}} {st.name};")
        header_lines.append("")
        header_lines.append(f'static_assert(sizeof({st.name}) == {st_layout.size}, "size mismatch for {st.name}");')
        for sub in st_layout.fields:
            header_lines.append(f'static_assert(offsetof({st.name}, {sub.name}) == {sub.offset}, "offset mismatch for {sub.name}");')
        header_lines.append("")

    header_lines.append(f"typedef struct {struct_layout.name} {{")
    body_lines, _ = _emit_struct_body(struct_layout.fields, struct_layout.size)
    header_lines.extend(body_lines)
    header_lines.append(f"}} {struct_layout.name};")
    header_lines.append("")

    header_lines.append(f'static_assert(sizeof({struct_layout.name}) == {struct_layout.size}, "size mismatch for {struct_layout.name}");')
    for f in struct_layout.fields:
        header_lines.append(f'static_assert(offsetof({struct_layout.name}, {f.name}) == {f.offset}, "offset mismatch for {f.name}");')
    header_lines.append("")

    header_lines.extend([
        "#ifdef __cplusplus",
        "}",
        "#endif",
        "",
        f"#endif // {upper_name}_H",
        "",
    ])

    return "\n".join(header_lines)


emit_c_header = generate_c_header


def verify_parameters(block: object) -> None:
    if not isinstance(block, dict):
        raise ValueError("Parameter reflection requires a dictionary.")
    schema_version = block.get("schema_version")

    if schema_version == 1:
        keys = {"schema_version", "name", "layout", "size", "alignment", "fields", "stages",
                "ownership", "access", "addressable", "bindings"}
        if set(block) != keys:
            raise ValueError("Parameter reflection requires the complete layout and target bindings.")
        if (block["layout"] != "gfx-root-f32-v1" or block["ownership"] != "host"
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
        return

    elif schema_version == 2:
        if block.get("ownership") != "host" or block.get("access") != "read-only" or block.get("addressable") is not False:
            raise ValueError("Unsupported parameter ownership or access.")
        if not isinstance(block.get("name"), str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", block["name"]):
            raise ValueError("Invalid parameter block name.")
        if block.get("layout") != "gfx-std430-v1":
            raise ValueError("Unsupported parameter layout.")
        fields = block.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("Parameters require non-empty fields list.")
        names = set()
        for f in fields:
            if not isinstance(f, dict) or "name" not in f or "offset" not in f or "size" not in f:
                raise ValueError("Invalid reflected field.")
            name = f["name"]
            if not isinstance(name, str) or name in names:
                raise ValueError(f"Duplicate or invalid field name {name!r}.")
            names.add(name)
        if type(block.get("size")) is not int or block["size"] <= 0:
            raise ValueError("Invalid parameter block size.")
        if type(block.get("alignment")) is not int or block["alignment"] <= 0:
            raise ValueError("Invalid parameter block alignment.")
        bindings = block.get("bindings")
        if not isinstance(bindings, dict) or not {"vulkan", "directx", "metal"}.issubset(bindings):
            raise ValueError("Missing target bindings.")
        return

    raise ValueError("Unsupported parameter schema, layout, ownership, or access.")


def pack_parameters(block: dict, values: Mapping[str, object]) -> bytes:
    verify_parameters(block)
    if not isinstance(values, Mapping) or set(values) != {field["name"] for field in block["fields"]}:
        raise ValueError("Supply exactly the reflected parameter fields.")
    data = bytearray(block["size"])

    def _pack_value(fmt: str, offset: int, val: Any):
        struct.pack_into(fmt, data, offset, val)

    def _pack_field(f: dict, val: Any, base_offset: int = 0):
        typ = f.get("type")
        offset = base_offset + f["offset"]
        if typ == "f32":
            if type(val) not in (int, float) or type(val) is bool or abs(val) > 3.4028234663852886e38 or not math.isfinite(val):
                raise ValueError("f32 requires finite float number.")
            _pack_value("<f", offset, float(val))
        elif typ == "f64":
            if type(val) not in (int, float) or type(val) is bool or not math.isfinite(val):
                raise ValueError("f64 requires finite float number.")
            _pack_value("<d", offset, float(val))
        elif typ == "f16":
            if type(val) not in (int, float) or type(val) is bool or not math.isfinite(val):
                raise ValueError("f16 requires finite float number.")
            _pack_value("<e", offset, float(val))
        elif typ in ("i32", "si32"):
            if type(val) is not int or type(val) is bool:
                raise ValueError("i32 requires int.")
            _pack_value("<i", offset, int(val))
        elif typ in ("u32", "ui32"):
            if type(val) is not int or type(val) is bool:
                raise ValueError("u32 requires int.")
            _pack_value("<I", offset, int(val))
        elif typ in ("i16", "si16"):
            if type(val) is not int or type(val) is bool:
                raise ValueError("i16 requires int.")
            _pack_value("<h", offset, int(val))
        elif typ in ("u16", "ui16"):
            if type(val) is not int or type(val) is bool:
                raise ValueError("u16 requires int.")
            _pack_value("<H", offset, int(val))
        elif typ in ("i8", "si8"):
            if type(val) is not int or type(val) is bool:
                raise ValueError("i8 requires int.")
            _pack_value("<b", offset, int(val))
        elif typ in ("u8", "ui8"):
            if type(val) is not int or type(val) is bool:
                raise ValueError("u8 requires int.")
            _pack_value("<B", offset, int(val))
        elif typ in ("bool", "i1"):
            _pack_value("<?", offset, bool(val))
        elif typ in ("f32x2", "f32x3", "f32x4"):
            count = int(typ[-1])
            if not isinstance(val, (tuple, list)) or len(val) != count:
                raise ValueError(f"{typ} requires sequence of {count} numbers.")
            for c in val:
                if type(c) not in (int, float) or type(c) is bool or not math.isfinite(c):
                    raise ValueError(f"{typ} requires finite numbers.")
            struct.pack_into(f"<{count}f", data, offset, *val)
        elif typ in ("i32x2", "i32x3", "i32x4"):
            count = int(typ[-1])
            if not isinstance(val, (tuple, list)) or len(val) != count:
                raise ValueError(f"{typ} requires sequence of {count} integers.")
            struct.pack_into(f"<{count}i", data, offset, *val)
        elif typ in ("u32x2", "u32x3", "u32x4"):
            count = int(typ[-1])
            if not isinstance(val, (tuple, list)) or len(val) != count:
                raise ValueError(f"{typ} requires sequence of {count} integers.")
            struct.pack_into(f"<{count}I", data, offset, *val)
        elif typ == "array":
            count = f.get("count", len(val) if isinstance(val, (tuple, list)) else 0)
            if not isinstance(val, (tuple, list)) or len(val) != count:
                raise ValueError("array requires matching sequence.")
            stride = f.get("stride", f.get("size", 0) // max(count, 1))
            elem_type = f.get("element_type", "f32")
            is_struct_elem = (elem_type == "struct") or ("fields" in f)
            for idx, item in enumerate(val):
                if is_struct_elem:
                    sub_f = {
                        "type": "struct",
                        "offset": idx * stride,
                        "size": stride,
                        "fields": f.get("fields", []),
                    }
                else:
                    sub_f = {"type": elem_type, "offset": idx * stride, "size": stride}
                _pack_field(sub_f, item, offset)
        elif typ == "struct":
            if isinstance(val, Mapping):
                for sub in f.get("fields", []):
                    _pack_field(sub, val[sub["name"]], offset)
            elif isinstance(val, (tuple, list)):
                for idx, sub in enumerate(f.get("fields", [])):
                    _pack_field(sub, val[idx], offset)
            else:
                raise ValueError("struct requires mapping or sequence.")
        else:
            raise ValueError(f"Unsupported field type for packing: {typ!r}")

    for field in block["fields"]:
        val = values[field["name"]]
        _pack_field(field, val)

    return bytes(data)
