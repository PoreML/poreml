"""Flat CSV logs that grow while a run is going: one row per epoch, per sample, per eval.

Lists become indexed columns (`hist/pred[0]`, ...), non-finite floats blank, and every
write is flushed so the file can be plotted mid-run. The header is fixed by the first
rows written; later rows may lack a column (blank) or bring a new one (dropped) rather
than crash a training run over a log line.
"""

import csv
from pathlib import Path
from typing import Any

from .metrics import json_safe


def flatten(record: dict[str, Any]) -> dict[str, Any]:
    """One CSV row: lists become indexed columns, None stays empty."""
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                flat[f"{key}[{i}]"] = item
        else:
            flat[key] = value
    return flat


def _row(record: dict[str, Any]) -> dict[str, Any]:
    return {k: ("" if v is None else v) for k, v in flatten(json_safe(record)).items()}


def append_csv(path: Path, record: dict[str, Any]) -> None:
    """Append one flattened row, writing the header on first use, and flush."""
    append_rows(path, [record])


def append_rows(path: Path, records: list[dict[str, Any]]) -> None:
    """Append flattened rows under the file's existing header (or the union of theirs on first use)."""
    rows = [_row(r) for r in records]
    if not rows:
        return
    if path.exists():
        with path.open(newline="") as f:
            header = next(csv.reader(f), None)
        header = header or []
    else:
        header = list(dict.fromkeys(k for row in rows for k in row))
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", restval="")
        if new:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
