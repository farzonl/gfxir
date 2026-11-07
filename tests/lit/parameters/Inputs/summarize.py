import json
import struct
import sys
from pathlib import Path


if sys.argv[1] == "spirv":
    binary = Path(sys.argv[2]).read_bytes()
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    offsets = {}
    position = 5
    while position < len(words):
        count, opcode = words[position] >> 16, words[position] & 65535
        if opcode == 72 and words[position + 3] == 35:
            offsets[words[position + 2]] = words[position + 4]
        position += count
    print("SPIR-V member offsets:", [offsets[index] for index in sorted(offsets)])
else:
    reflection = json.loads(Path(sys.argv[2]).read_text())
    print("parameters:", json.dumps(reflection.get("parameters"), sort_keys=True))
