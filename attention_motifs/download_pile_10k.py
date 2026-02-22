"""Download Pile-10k parquet from HuggingFace and convert to JSONL."""

import argparse
import json
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

URL: str = "https://huggingface.co/datasets/NeelNanda/pile-10k/resolve/main/data/train-00000-of-00001-4746b8785c874cc7.parquet"


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Download Pile-10k parquet and convert to JSONL.",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for the JSONL file",
    )
    args: argparse.Namespace = parser.parse_args()
    output: Path = args.output

    with tempfile.TemporaryDirectory() as tmp_dir:
        parquet_path: Path = Path(tmp_dir) / "pile_10k.parquet"

        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, parquet_path)
        print(f"Saved parquet to {parquet_path}")

        df: pd.DataFrame = pd.read_parquet(parquet_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for _, row in df.iterrows():
            line: dict[str, object] = {"text": row["text"], "meta": row["meta"]}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(df)} lines to {output}")


if __name__ == "__main__":
    main()
