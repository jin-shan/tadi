import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tathate.data.build import build_unified_corpus


def main() -> None:
    data_root = Path("data")
    out_root = data_root / "unified"
    metadata = build_unified_corpus(data_root, out_root)
    print("Built unified corpus at", out_root)
    print("Merged split sizes:", metadata["merged_split_sizes"])


if __name__ == "__main__":
    main()
