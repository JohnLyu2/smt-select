#!/usr/bin/env python3
"""
Generate doc/cp26/smtcomp24_logic_info.tex from meta_info and performance data.

Data locations (different from output):
- Meta-info: data/raw_data/meta_info/<logic>.json (benchmark list, description, size)
- Performance: data/raw_data/smtcomp24_performance/<logic>.json (SMT-COMP 2024)

Output: doc/cp26/smtcomp24_logic_info.tex (LaTeX table only).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.performance import parse_performance_json

DEFAULT_META_DIR = PROJECT_ROOT / "data" / "raw_data" / "meta_info"
DEFAULT_PERF_DIR = PROJECT_ROOT / "data" / "raw_data" / "smtcomp24_performance"
DEFAULT_TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "smtcomp24_logic_info.tex"
DEFAULT_CSV_PATH = PROJECT_ROOT / "doc" / "logic_filter" / "smtcomp24_logic_info.csv"


def latex_escape(s: str) -> str:
    """Escape characters that are special in LaTeX."""
    s = str(s)
    s = s.replace("\\", "\\textbackslash{}")
    s = s.replace("&", "\\&")
    s = s.replace("%", "\\%")
    s = s.replace("$", "\\$")
    s = s.replace("#", "\\#")
    s = s.replace("_", "\\_")
    s = s.replace("{", "\\{")
    s = s.replace("}", "\\}")
    s = s.replace("~", "\\textasciitilde{}")
    s = s.replace("^", "\\textasciicircum{}")
    return s


def format_rate(val: str | float) -> str:
    """Format description_rate for LaTeX (e.g. 0.986 -> 98.6). Unit in column header."""
    if val == "" or val is None:
        return "---"
    try:
        x = float(val)
        return f"{x * 100:.1f}"
    except (TypeError, ValueError):
        return str(val)


def load_meta_info(meta_dir: Path, logic: str) -> tuple[int, int, int, int] | None:
    """Load meta_info JSON for a logic. Returns (num_benchmarks, num_families, num_with_description, total_size) or None."""
    path = meta_dir / f"{logic}.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return None
    families = set()
    with_desc = 0
    total_size = 0
    for item in data:
        if isinstance(item, dict):
            if "family" in item:
                families.add(item["family"])
            desc = item.get("description")
            if desc is not None and isinstance(desc, str) and desc.strip():
                with_desc += 1
            s = item.get("size")
            if isinstance(s, (int, float)) and s >= 0:
                total_size += int(s)
    return len(data), len(families), with_desc, total_size


def build_rows(meta_dir: Path, perf_dir: Path, timeout: float) -> list[dict[str, int | str | float]]:
    """Build logic info rows from meta_info and performance JSONs."""
    rows: list[dict[str, int | str | float]] = []
    for perf_path in sorted(perf_dir.glob("*.json")):
        logic = perf_path.stem
        try:
            multi = parse_performance_json(str(perf_path), timeout)
        except (ValueError, OSError) as e:
            print(f"Warning: skip {perf_path.name}: {e}", file=sys.stderr)
            continue

        num_solvers = multi.num_solvers()
        sbs_ds = multi.get_best_solver_dataset()
        vbs_ds = multi.get_virtual_best_solver_dataset()
        sbs = sbs_ds.get_solved_count()
        sbs_solver = sbs_ds.get_solver_name()
        vbs = vbs_ds.get_solved_count()
        num_benchmarks_perf = len(multi)

        meta = load_meta_info(meta_dir, logic)
        if meta is not None:
            num_benchmarks, num_families, num_with_desc, total_size = meta
            rate = num_with_desc / num_benchmarks if num_benchmarks else 0.0
            description_rate = round(rate, 3)
            ave_size = round(total_size / num_benchmarks / 1024, 1) if num_benchmarks else 0  # KB
        else:
            num_benchmarks = num_benchmarks_perf
            num_families = 0
            description_rate = ""
            ave_size = ""

        rows.append({
            "logic": logic,
            "num_benchmarks": num_benchmarks,
            "num_families": num_families,
            "num_solvers": num_solvers,
            "sbs": sbs,
            "sbs_solver": sbs_solver or "",
            "vbs": vbs,
            "description_rate": description_rate,
            "ave_size": ave_size,
        })
    return rows


def rows_to_latex(rows: list[dict[str, int | str | float]], no_header: bool) -> str:
    """Convert data rows to LaTeX tabular content."""
    exclude = {"num_families"}
    fieldnames = ["logic", "num_benchmarks", "num_families", "num_solvers", "sbs", "vbs", "description_rate", "ave_size"]
    columns = [k for k in fieldnames if k not in exclude]

    header_map = {
        "logic": "Logic",
        "num_benchmarks": "Benchmarks",
        "num_solvers": "Solvers",
        "sbs": "SBS solved",
        "vbs": "VBS solved",
        "description_rate": "Desc.\\%",
        "ave_size": "Avg. file size (KB)",
    }
    headers = [header_map.get(k, k.replace("_", " ").title()) for k in columns]
    ncols = len(columns)
    col_spec = "l" + "c" * (ncols - 1)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
    ]
    if not no_header:
        lines.append(" & ".join(headers) + " \\\\")
        lines.append("\\midrule")

    for row in rows:
        cells = []
        for key in columns:
            val = row.get(key, "")
            if key == "logic":
                cells.append(latex_escape(val))
            elif key == "description_rate":
                cells.append(format_rate(val))
            elif key == "ave_size":
                cells.append(str(val) if val != "" else "---")
            elif key == "sbs":
                sbs_str = str(val) if val != "" else "---"
                sbs_solver = row.get("sbs_solver") or ""
                if sbs_solver:
                    sbs_str += " (" + latex_escape(sbs_solver) + ")"
                cells.append(sbs_str)
            else:
                cells.append(str(val) if val != "" else "---")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Meta-information for the selected SMT-COMP 2024 logics: benchmark count, number of competition solvers, SBS and VBS solved counts, description rate, and average file size.}",
        "\\label{tab:logic-info}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SMT-COMP 2024 logic info from data/ and write doc/cp26/smtcomp24_logic_info.tex."
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=DEFAULT_META_DIR,
        help=f"Meta-info JSONs (default: data/raw_data/meta_info)",
    )
    parser.add_argument(
        "--perf-dir",
        type=Path,
        default=DEFAULT_PERF_DIR,
        help=f"Performance JSONs (default: {DEFAULT_PERF_DIR.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_TEX_PATH,
        help=f"Output .tex path (default: doc/cp26/smtcomp24_logic_info.tex)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="PAR-2 timeout in seconds (default: 1200)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Omit the tabular header row",
    )
    args = parser.parse_args()

    meta_dir = args.meta_dir.resolve()
    perf_dir = args.perf_dir.resolve()
    if not perf_dir.is_dir():
        raise SystemExit(f"Performance directory not found: {perf_dir}")

    rows = build_rows(meta_dir, perf_dir, args.timeout)
    if not rows:
        raise SystemExit("No logic data produced; check --meta-dir and --perf-dir.")

    # Write LaTeX
    tex_path = args.output.resolve()
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(rows_to_latex(rows, args.no_header), encoding="utf-8")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
