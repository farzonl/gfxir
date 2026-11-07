# RUN: %python %s loop | %FileCheck %s --check-prefix=LOOP
# RUN: %python %s short-circuit | %FileCheck %s --check-prefix=SHORT
# RUN: %python %s gallery | %FileCheck %s --check-prefix=GALLERY
# LOOP: schema versions: [2, 2, 2]
# LOOP-NEXT: while counts: [1, 1, 1]
# LOOP-NEXT: serialized size delta: {{([0-9]|[12][0-9]|3[0-2])$}}
# SHORT: root operations:
# SHORT-NOT: math.log
# SHORT: nested math.log: 1
# SHORT-NEXT: scf.if: {{[1-9][0-9]*}}
# GALLERY: parameters: frame
# GALLERY-DAG: scf.while
# GALLERY-DAG: gfx.parameter

import json
import runpy
import sys

from mlir_gpu import (
    FragmentCoord, Pipeline, fragment, fullscreen_vertex, select,
    shader_math as sm, vec4,
)
from test_support import ROOT


def count_ops(value, operation):
    if isinstance(value, dict):
        return int(value.get("op") == operation) + sum(
            count_ops(item, operation) for item in value.values()
        )
    if isinstance(value, list):
        return sum(count_ops(item, operation) for item in value)
    return 0


def loop_output():
    modules = []
    for limit in (1, 64, 1024):
        @fragment
        def loop(pixel: FragmentCoord):
            value = 0.0
            for i in range(limit):
                value += pixel.x * 0.001
            return vec4(value, 0, 0, 1)

        modules.append(Pipeline(fullscreen_vertex, loop).trace()[1].to_dict())
    lengths = [len(json.dumps(module)) for module in modules]
    print("schema versions:", [module["schema_version"] for module in modules])
    print("while counts:", [count_ops(module, "scf.while") for module in modules])
    print("serialized size delta:", max(lengths) - min(lengths))


def short_circuit_output():
    @fragment
    def shader(pixel: FragmentCoord):
        x = pixel.x - 0.5
        safe = x > 0.0 and sm.log(x) > 0.0
        return vec4(select(safe, 1.0, 0.0), 0, 0, 1)

    module = Pipeline(fullscreen_vertex, shader).trace()[1].to_dict()
    root = module["functions"][0]["body"]["blocks"][0]["operations"]
    print("root operations:", [op["op"] for op in root])
    print("nested math.log:", count_ops(module, "math.log"))
    print("scf.if:", count_ops(module, "scf.if"))


def gallery_output():
    shader = runpy.run_path(str(ROOT / "examples/beacon_gallery.py"))["pipeline"].trace()[1].to_dict()
    print("parameters:", shader["parameters"]["name"])
    for function in shader["functions"]:
        for op in function["body"]["blocks"][0]["operations"]:
            print(op["op"])


{"loop": loop_output, "short-circuit": short_circuit_output, "gallery": gallery_output}[sys.argv[1]]()
