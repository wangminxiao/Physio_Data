"""Dirty-comma CSV repair for UCSF EHR .txt shards.

The Filtered_*_New tables under /labs/hulab/UCSF/rdb_new/ are latin-1 CSV with
unquoted commas inside free-text fields (lab common names, drug names, etc.).
polars/pandas naive read_csv mis-aligns columns silently. Run rows through
remove_bad_commas (numbers + words) or remove_bad_commas_quotes (quote-aware)
before parsing.

Vendored from /labs/hulab/mxwang/data/ucsf_EHR/EHR_encounter_polars.py:12-63.
"""
from __future__ import annotations
import re
from io import StringIO
from typing import Iterable

import polars as pl


def remove_bad_commas(line: str) -> str:
    """Strip commas that are clearly not field separators.

    Two passes:
      1. Thousands separators inside numbers (1,234 -> 1234).
      2. Trailing commas after non-numeric phrases (CO2, ALT, ...).
    """
    line = re.sub(r"(?<=[^\d,])(\d{1,3}),(?=\d{3}(?!\d))", r"\1", line)
    line = re.sub(
        r"\b(?!\d+(?:\.\d+)?\b)([^\",\s]+(?:\s[^\",\s]+)*?),",
        r"\1",
        line,
    )
    return line


def remove_bad_commas_quotes(line: str, expected_n_fields: int) -> str:
    """Quote-aware repair: keeps overflow commas inside quoted free-text fields.

    expected_n_fields = number of header columns. Lines with more comma-split
    parts get their excess folded back into the quoted field.
    """
    new_line = remove_bad_commas(line)
    fields = new_line.split(",")
    overflow = max(0, len(fields) - expected_n_fields)

    out: list[str] = []
    held: str | None = None
    for f in fields:
        if "\"" in f:
            if overflow > 0 and f.count("\"") % 2 != 0:
                held = f.replace("\"", "")
                overflow -= 1
            else:
                clean = f.replace("\"", "")
                if held is not None:
                    clean = held + clean
                    held = None
                out.append(f"\"{clean}\"")
        else:
            out.append(f)
    return ",".join(out)


def read_dirty_csv(
    path: str,
    *,
    expected_columns: list[str] | None = None,
    encoding: str = "latin-1",
    quote_aware: bool = True,
    schema_overrides: dict | None = None,
) -> pl.DataFrame:
    """Read a dirty-comma latin-1 .txt shard into a polars DataFrame.

    If expected_columns is given, header validation runs and quote_aware repair
    uses len(expected_columns) as the field count. Otherwise the first line is
    treated as the header.
    """
    with open(path, "r", encoding=encoding, errors="replace") as f:
        header = f.readline().rstrip("\n").rstrip("\r")
        body_lines = f.readlines()

    cols = header.split(",")
    if expected_columns is not None:
        if cols != expected_columns:
            raise ValueError(
                f"{path}: header mismatch.\n  got: {cols}\n  expected: {expected_columns}"
            )

    n_fields = len(cols)
    repair = (
        (lambda s: remove_bad_commas_quotes(s, n_fields)) if quote_aware else remove_bad_commas
    )

    repaired = StringIO()
    repaired.write(header + "\n")
    for line in body_lines:
        line = line.rstrip("\n").rstrip("\r")
        if not line:
            continue
        repaired.write(repair(line) + "\n")
    repaired.seek(0)
    return pl.read_csv(
        repaired,
        schema_overrides=schema_overrides or {},
        truncate_ragged_lines=True,
        ignore_errors=True,
    )


def iter_repaired_lines(lines: Iterable[str], expected_n_fields: int, *, quote_aware: bool = True) -> Iterable[str]:
    """Generator wrapper for streaming use."""
    repair = (
        (lambda s: remove_bad_commas_quotes(s, expected_n_fields)) if quote_aware else remove_bad_commas
    )
    for line in lines:
        line = line.rstrip("\n").rstrip("\r")
        if line:
            yield repair(line)
