"""Print public pipeline/viewer boundary facts for FileCheck."""

import copy
import json
import shutil
import sys
from pathlib import Path

from mlir_gpu import Target, compile_pipeline
from mlir_gpu.viewer import ViewerError, load_program
from programs import pipeline
from test_support import TOOLS


name, target, output = sys.argv[1], Target(sys.argv[2]), Path(sys.argv[3])
if output.exists():
    shutil.rmtree(output)
if name == "mismatch":
    try:
        compile_pipeline(pipeline(name), target, output, TOOLS)
    except ValueError as error:
        print("layout mismatch:", error)
    print("published output exists:", output.exists())
else:
    manifest = compile_pipeline(pipeline(name), target, output, TOOLS)
    data = json.loads(manifest.read_text())
    print("manifest schema:", data["schema_version"])
    print("visibility:", data["parameters"]["stages"])
    print("fragment reflection matches:", data["parameters"] == data["interface"]["fragment"]["parameters"])
    if name != "visibility":
        try:
            print("viewer parameters match:", load_program(manifest).parameters == data["parameters"])
        except ViewerError as error:
            print("viewer:", error)
        invalid = copy.deepcopy(data)
        invalid["parameters"]["bindings"]["vulkan"]["offset"] = False
        manifest.write_text(json.dumps(invalid))
        try:
            load_program(manifest)
        except ViewerError as error:
            print("invalid binding:", error)
        data["schema_version"] = 3
        manifest.write_text(json.dumps(data))
        try:
            load_program(manifest)
        except ViewerError as error:
            print("old schema:", error)
