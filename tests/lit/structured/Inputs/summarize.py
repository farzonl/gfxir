import json
import sys
from pathlib import Path


if sys.argv[1] == "exists":
    print(f"artifact exists: {Path(sys.argv[2]).exists()}")
else:
    reflected = json.loads(Path(sys.argv[2]).read_text())
    print(f"name: {reflected['name']}")
    print("inputs:", json.dumps(reflected["inputs"], sort_keys=True))
