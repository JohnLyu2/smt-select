#!/usr/bin/env python3
"""
Collect graph feature extraction fail counts from extraction_times CSVs.

Reads CSVs from data/features/graph/<logic>/seed*/extraction_times.csv,
which have columns: path,time_sec,failed. For each seed, compute
fail_rate = (# rows with failed==1) / (total rows). Then average over
seeds per logic. Also reads data/results/sibyl/graph_log/graph_build_<logic>.csv
(same format) for Sibyl fail rate. Writes doc/cp26/feature_fail.tex with
columns: Logic, Fail rate (%), Sibyl (%).
"""

import csv
from pathlib import Path

# Script is under scripts/collect_results/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRAPH_FEATURE_ROOT = PROJECT_ROOT / "data" / "features" / "graph"
SIBYL_GRAPH_LOG = PROJECT_ROOT / "data" / "results" / "sibyl" / "graph_log"
OUTPUT_TEX = PROJECT_ROOT / "doc" / "cp26" / "feature_fail.tex"


def latex_escape(s: str) -> str:
    """Escape special characters for LaTeX (e.g. _ -> \\_)."""
    return s.replace("_", "\\_")


def main() -> None:
    if not GRAPH_FEATURE_ROOT.is_dir():
        raise FileNotFoundError(f"Graph feature directory not found: {GRAPH_FEATURE_ROOT}")

    rows: list[dict[str, str | float]] = []
    for logic_dir in sorted(GRAPH_FEATURE_ROOT.iterdir()):
        if not logic_dir.is_dir():
            continue
        logic = logic_dir.name
        fail_rates: list[float] = []
        for seed_dir in sorted(logic_dir.glob("seed*")):
            if not seed_dir.is_dir():
                continue
            csv_path = seed_dir / "extraction_times.csv"
            if not csv_path.is_file():
                continue
            try:
                with csv_path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    total = 0
                    failed = 0
                    for row in reader:
                        total += 1
                        try:
                            failed_flag = int(row.get("failed", "0"))
                        except ValueError:
                            failed_flag = 0
                        if failed_flag:
                            failed += 1
            except OSError:
                continue
            rate = failed / total if total else 0.0
            fail_rates.append(rate)
        if not fail_rates:
            continue
        avg_fail_rate = sum(fail_rates) / len(fail_rates)
        rows.append({"logic": logic, "avg_fail_rate": avg_fail_rate})

    # Sibyl fail rate from graph_log (one CSV per logic: graph_build_<logic>.csv)
    sibyl_rates: dict[str, float] = {}
    if SIBYL_GRAPH_LOG.is_dir():
        for csv_path in SIBYL_GRAPH_LOG.glob("graph_build_*.csv"):
            logic = csv_path.stem.replace("graph_build_", "")
            total = 0
            failed = 0
            try:
                with csv_path.open(encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        total += 1
                        try:
                            if int(row.get("failed", "0")):
                                failed += 1
                        except ValueError:
                            pass
            except OSError:
                continue
            if total:
                sibyl_rates[logic] = (failed / total) * 100

    # Write LaTeX table
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Logic & Sibyl & SMT-Select-Graph \\\\",
        "\\midrule",
    ]
    for row in rows:
        logic = latex_escape(row["logic"])
        rate = float(row["avg_fail_rate"])
        rate_str = f"{rate * 100:.1f}"
        sibyl_val = sibyl_rates.get(row["logic"])
        sibyl_str = f"{sibyl_val:.1f}" if sibyl_val is not None else "---"
        lines.append(f"{logic} & {sibyl_str} & {rate_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Average failure rate (\\%) of graph construction for Sibyl and SMT-Select-Graph.}",
        "\\label{tab:feature-fail}",
        "\\end{table}",
    ])
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {OUTPUT_TEX}")


if __name__ == "__main__":
    main()
