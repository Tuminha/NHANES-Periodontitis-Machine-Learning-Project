#!/usr/bin/env python3
"""Deprecated notebook scaffolding helper.

The previous version generated a notebook skeleton with placeholder cells. That
is not appropriate for a publication-facing repository because it looks like a
reproducible analysis while leaving core steps unimplemented.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "This scaffolding helper is deprecated. Use the checked-in notebooks and "
        "script targets in the Makefile instead."
    )


if __name__ == "__main__":
    main()
