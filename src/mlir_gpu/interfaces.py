import re

from .frontend import Type


def verify_graphics_interface(interface: object, *, preview: bool = False) -> None:
    if not isinstance(interface, dict) or set(interface) != {"vertex", "fragment"}:
        raise ValueError("Graphics reflection requires vertex and fragment interfaces.")
    ports = {}
    for stage in ("vertex", "fragment"):
        reflection = interface[stage]
        if not isinstance(reflection, dict) or reflection.get("stage") != stage:
            raise ValueError(f"Invalid {stage} reflection.")
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
