import json
import sys
from pathlib import Path


directory = Path(sys.argv[1])
manifest = json.loads((directory / "pipeline.json").read_text())
print("entry count:", len(manifest["entries"]))
for entry in manifest["entries"]:
    binary = directory / entry["binary"]
    data = binary.read_bytes()
    print(entry["stage"], "magic:", data[:4].hex())
    dxil = directory / f"{entry['stage']}.dxil"
    if dxil.exists():
        print(entry["stage"], "DXIL digest:", dxil.read_bytes()[4:20].hex())
