import re

from .frontend import Type
from .parameters import verify_parameters


SUPPORTED_DEPTH_FORMATS = {"d32_sfloat"}
COMPARE_OPS = {
    "never", "less", "equal", "less_equal",
    "greater", "not_equal", "greater_equal", "always",
}


def verify_depth_stencil(depth_stencil: object) -> None:
    if not isinstance(depth_stencil, dict):
        raise ValueError("depth_stencil configuration must be a dictionary.")

    fmt = depth_stencil.get("format")
    if not isinstance(fmt, str) or fmt not in SUPPORTED_DEPTH_FORMATS:
        raise ValueError(f"Unsupported depth format: {fmt!r}. Supported formats: {sorted(SUPPORTED_DEPTH_FORMATS)}.")

    test_enable = depth_stencil.get("depth_test_enable")
    if type(test_enable) is not bool:
        raise ValueError(f"depth_test_enable must be a boolean, got {type(test_enable).__name__}.")

    write_enable = depth_stencil.get("depth_write_enable")
    if type(write_enable) is not bool:
        raise ValueError(f"depth_write_enable must be a boolean, got {type(write_enable).__name__}.")

    op = depth_stencil.get("depth_compare_op")
    if not isinstance(op, str) or op not in COMPARE_OPS:
        raise ValueError(f"Unsupported depth_compare_op: {op!r}. Supported operators: {sorted(COMPARE_OPS)}.")

    if "clear_depth" in depth_stencil:
        clear = depth_stencil["clear_depth"]
        if type(clear) not in (int, float) or isinstance(clear, bool) or not (0.0 <= clear <= 1.0):
            raise ValueError(f"clear_depth must be a float in [0.0, 1.0], got {clear!r}.")


def verify_graphics_interface(interface: object, *, preview: bool = False) -> None:
    if not isinstance(interface, dict) or set(interface) != {"vertex", "fragment"}:
        raise ValueError("Graphics reflection requires vertex and fragment interfaces.")
    ports = {}
    for stage in ("vertex", "fragment"):
        reflection = interface[stage]
        if not isinstance(reflection, dict) or reflection.get("stage") != stage:
            raise ValueError(f"Invalid {stage} reflection.")
        if "parameters" in reflection:
            params = reflection["parameters"]
            verify_parameters(params)
            param_list = params if isinstance(params, list) else [params]
            for param_block in param_list:
                if not isinstance(param_block, dict) or param_block.get("stages") != [stage]:
                    raise ValueError("Parameter visibility does not match the reflected stage.")
        name = reflection.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid reflected {stage} entry-point name.")
        for direction in ("inputs", "outputs"):
            descriptions = reflection.get(direction)
            if not isinstance(descriptions, list):
                raise ValueError(f"{stage} {direction} must be a reflected interface list.")
            locations, builtins = {}, set()
            for io in descriptions:
                if not isinstance(io, dict):
                    raise ValueError("Shader interface elements must be objects.")
                label, type_name = io.get("name"), io.get("type")
                if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
                    raise ValueError("Shader interface elements require valid names.")
                if not isinstance(type_name, str):
                    raise ValueError("Shader interface elements require explicit types.")
                typ = Type.from_schema(type_name)
                builtin = io.get("builtin")
                if builtin is not None:
                    expected = {
                        ("vertex", "inputs", "vertex_index"): Type.U32,
                        ("vertex", "outputs", "position"): Type.VEC4,
                        ("fragment", "inputs", "fragment_coord"): Type.VEC2,
                    }.get((stage, direction, builtin)) if isinstance(builtin, str) else None
                    if expected != typ or builtin in builtins or "location" in io:
                        raise ValueError("Unsupported, duplicated, or incorrectly typed graphics builtin.")
                    builtins.add(builtin)
                else:
                    location = io.get("location")
                    mode = io.get("interpolation")
                    if type(location) is not int or location < 0 or location in locations:
                        raise ValueError("Shader interface locations must be unique nonnegative integers.")
                    if mode not in ("perspective", "flat", "no_perspective") or typ == Type.BOOL:
                        raise ValueError("Invalid rasterization interface type or interpolation.")
                    if typ == Type.U32 and mode != "flat":
                        raise ValueError("Integer interfaces require flat interpolation.")
                    locations[location] = io
            if stage == "vertex" and direction == "outputs" and builtins != {"position"}:
                raise ValueError("Vertex shaders must declare exactly one clip-position output.")
            if preview and stage == "vertex" and direction == "inputs" and locations:
                raise ValueError("The preview does not implement vertex-buffer inputs yet.")
            ports[stage, direction] = locations
    for location, consumer in ports["fragment", "inputs"].items():
        producer = ports["vertex", "outputs"].get(location)
        if producer is None or any(producer[key] != consumer[key] for key in ("type", "interpolation")):
            raise ValueError(f"Fragment input location {location} does not match the vertex output interface.")
    if preview:
        outputs = ports["fragment", "outputs"]
        if set(outputs) != {0} or outputs[0]["type"] != "f32x4":
            raise ValueError("The preview currently supports one f32x4 render target at location 0.")


def reflect_pipeline_parameters(interface: dict) -> dict | list[dict] | None:
    stage_blocks: list[tuple[str, dict]] = []
    for stage in ("vertex", "fragment"):
        stage_desc = interface.get(stage, {})
        if "parameters" in stage_desc and stage_desc["parameters"]:
            params = stage_desc["parameters"]
            if isinstance(params, dict):
                stage_blocks.append((stage, params))
            elif isinstance(params, list):
                for p in params:
                    stage_blocks.append((stage, p))

    if not stage_blocks:
        return None

    # Reconcile blocks by name
    reconciled: list[dict] = []
    by_name: dict[str, dict] = {}

    for stage, block in stage_blocks:
        verify_parameters(block)
        name = block["name"]
        if name not in by_name:
            entry = {**block, "stages": list(block.get("stages", [stage]))}
            if stage not in entry["stages"]:
                entry["stages"].append(stage)
            by_name[name] = entry
            reconciled.append(entry)
        else:
            existing = by_name[name]
            if existing.get("layout") != block.get("layout"):
                if {existing.get("layout"), block.get("layout")}.issubset({"gfx-root-f32-v1", "gfx-std430-v1"}):
                    existing["layout"] = "gfx-std430-v1"
                    existing["schema_version"] = 2
                else:
                    raise ValueError(f"Graphics stages require matching parameter block layout: {name!r}.")
            for key in ("size", "alignment"):
                if existing.get(key) != block.get(key):
                    raise ValueError(f"Graphics stages require matching parameter block layout: {name!r}.")
            if len(existing.get("fields", [])) != len(block.get("fields", [])):
                raise ValueError(f"Graphics stages require matching parameter block fields: {name!r}.")
            for f1, f2 in zip(existing["fields"], block["fields"]):
                if {k: v for k, v in f1.items() if k != "stages"} != {k: v for k, v in f2.items() if k != "stages"}:
                    raise ValueError(f"Graphics stages require matching parameter block fields: {name!r}.")
            for s in block.get("stages", [stage]):
                if s not in existing["stages"]:
                    existing["stages"].append(s)

    # Compute sequential target bindings across all reconciled blocks
    cumulative_offset = 0
    for i, entry in enumerate(reconciled):
        block_size = entry["size"]
        num32 = (block_size + 3) // 4
        entry["bindings"] = {
            "vulkan": {"storage": "push_constant", "offset": cumulative_offset},
            "directx": {
                "storage": "root_constants",
                "parameter_index": i,
                "register": i,
                "space": 0,
                "num32bit_constants": num32,
                "visibility": "all",
                "root_signature_version": "1.1",
            },
            "metal": {
                "storage": "converter_argument_buffer",
                "buffer_index": 2 + i,
                "offset": cumulative_offset,
                "size": block_size,
            },
        }
        cumulative_offset += (block_size + 15) // 16 * 16

    if cumulative_offset > 256:
        raise ValueError(f"Cumulative parameter blocks size ({cumulative_offset}B) exceeds 256 bytes.")

    # Verify that each stage's compiled bindings match the reconciled pipeline bindings
    has_compiled_bindings = any(
        block.get("bindings", {}).get("vulkan", {}).get("offset", 0) > 0
        or block.get("bindings", {}).get("directx", {}).get("register", 0) > 0
        for _, block in stage_blocks
    )
    if has_compiled_bindings:
        for stage, block in stage_blocks:
            name = block["name"]
            pipeline_block = by_name[name]
            stage_vk_offset = block.get("bindings", {}).get("vulkan", {}).get("offset")
            pipe_vk_offset = pipeline_block["bindings"]["vulkan"]["offset"]
            if stage_vk_offset is not None and stage_vk_offset != pipe_vk_offset:
                raise ValueError(
                    f"Stage {stage!r} parameter block {name!r} Vulkan offset ({stage_vk_offset}) "
                    f"desynchronized from pipeline layout ({pipe_vk_offset})."
                )
            stage_dx_reg = block.get("bindings", {}).get("directx", {}).get("register")
            pipe_dx_reg = pipeline_block["bindings"]["directx"]["register"]
            if stage_dx_reg is not None and stage_dx_reg != pipe_dx_reg:
                raise ValueError(
                    f"Stage {stage!r} parameter block {name!r} DirectX register (b{stage_dx_reg}) "
                    f"desynchronized from pipeline layout (b{pipe_dx_reg})."
                )

    if len(reconciled) == 1:
        return reconciled[0]
    return reconciled
