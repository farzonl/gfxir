import collections
import re
import sys
from pathlib import Path


first, second = (Path(path) for path in sys.argv[1:3])
extension = sys.argv[3]
for directory in (first, second):
    counts = collections.Counter(re.findall(
        r"^\s*(?:%.*?=\s*)?([a-z]+\.[\w]+)",
        (directory / "fragment.mlir").read_text(), re.MULTILINE,
    ))
    (directory / "counts.txt").write_text(
        "\n".join(f"{op}: {count}" for op, count in sorted(counts.items())) + "\n"
    )
print("size delta:", abs(
    (first / f"fragment.{extension}").stat().st_size
    - (second / f"fragment.{extension}").stat().st_size
))
