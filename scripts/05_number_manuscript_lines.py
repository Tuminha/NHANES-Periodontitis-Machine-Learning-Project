#!/usr/bin/env python3
"""Create a line-numbered plain-text manuscript copy from Markdown source."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="docs/publication/ARTICLE_DRAFT.md")
    parser.add_argument("--output", default="reports/ARTICLE_DRAFT_line_numbered.txt")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"Manuscript source not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = source.read_text(encoding="utf-8").splitlines()
    numbered = [
        f"{idx:04d}  {line}" if line else f"{idx:04d}"
        for idx, line in enumerate(lines, start=1)
    ]
    output.write_text("\n".join(numbered) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
