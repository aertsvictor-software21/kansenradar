"""Simple progress messages for long runs (printed to stderr, Dutch)."""

from __future__ import annotations

import sys
import time

_START = time.monotonic()


def progress(message: str) -> None:
    minutes = (time.monotonic() - _START) / 60
    print(f"  [{minutes:4.1f} min] {message}", file=sys.stderr, flush=True)
