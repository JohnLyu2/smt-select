from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

CSV_HEADER = [
    "benchmark",
    "selected",
    "solved",
    "runtime",
    "solver_runtime",
    "overhead",
    "feature_fail",
]


def load_eval_csv(path: Path) -> list[dict]:
    """Load eval CSV; return list of row dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_eval_csv(path: Path, rows: Iterable[dict], header: list[str] | None = None) -> None:
    """Write rows using header order (defaults to standard AS CSV header)."""
    hdr = header or CSV_HEADER
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(hdr)
        for row in rows:
            writer.writerow([row.get(k, "") for k in hdr])


def load_fallback_lookup(path: Path) -> dict[str, dict]:
    """Load fallback (Lite or Lite+Text) eval CSV; return benchmark -> row dict."""
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = (row.get("benchmark") or "").strip()
            if bench:
                out[bench] = row
    return out


def merge_with_fallback(
    primary_rows: list[dict],
    fb_lookup: dict[str, dict],
    timeout: float,
) -> list[dict]:
    """
    Replace feature_fail rows with fallback results.

    For feature_fail rows:
      overhead_out = primary_overhead + fb_overhead
      runtime_out  = fb_solver_runtime + overhead_out  (capped at timeout)

    This matches the Lite and Lite+Text fallback semantics used in WL / Fusion / Graph.
    """
    out: list[dict] = []
    for row in primary_rows:
        bench = (row.get("benchmark") or "").strip()
        try:
            ff = int(row.get("feature_fail", 0) or 0)
        except (ValueError, TypeError):
            ff = 0

        if ff == 0:
            out.append(row)
            continue

        if bench not in fb_lookup:
            raise ValueError(f"Missing fallback row for feature-fail benchmark: {bench!r}")
        fb_row = fb_lookup[bench]

        raw_primary_overhead = row.get("overhead", "")
        try:
            primary_overhead = float(raw_primary_overhead) if raw_primary_overhead not in ("", None) else 0.0
        except (TypeError, ValueError):
            primary_overhead = 0.0

        try:
            fb_solver_runtime = float(fb_row.get("solver_runtime") or 0.0)
        except (TypeError, ValueError):
            fb_solver_runtime = 0.0

        raw_fb_overhead = fb_row.get("overhead", "")
        try:
            fb_overhead = float(raw_fb_overhead) if raw_fb_overhead not in ("", None) else 0.0
        except (TypeError, ValueError):
            fb_overhead = 0.0

        overhead_out = primary_overhead + fb_overhead
        runtime_out = fb_solver_runtime + overhead_out

        if runtime_out > timeout:
            solved = 0
            runtime_out = timeout
            overhead_out = max(0.0, timeout - fb_solver_runtime)
        else:
            try:
                solved = int(fb_row.get("solved") or 0)
            except (TypeError, ValueError):
                solved = 0

        row_out = dict(row)
        row_out["selected"] = fb_row.get("selected", row.get("selected"))
        row_out["solved"] = str(solved)
        row_out["runtime"] = f"{runtime_out:.6f}"
        row_out["solver_runtime"] = f"{fb_solver_runtime:.6f}"
        row_out["overhead"] = f"{overhead_out:.6f}"
        row_out["feature_fail"] = "1"
        out.append(row_out)

    return out

