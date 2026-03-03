#!/usr/bin/env python3
"""
Generate a LaTeX table of PAR-2 gap closed (%) for text-only and lite+text models.

Reads summary.json from data/cp26/results/text/<model>/<logic>/summary.json
and data/cp26/results/lite+text/<model>/<logic>/summary.json,
extracts test gap_cls_par2_mean, and writes doc/cp26/desc.tex.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_RESULTS_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "text"
LITE_TEXT_RESULTS_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "lite+text"
TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "desc.tex"

MODEL_ORDER = ["all-mpnet-base-v2", "Qwen3-Embedding-0.6B"]
MODEL_DISPLAY = {
    "all-mpnet-base-v2": "mpnet",
    "Qwen3-Embedding-0.6B": "Qwen3",
}

GRAPH_TEXT_RESULTS_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "graph+text"

GROUPS = [
    ("SMT-Select-Text", TEXT_RESULTS_ROOT),
    ("SMT-Select-Lite+Text", LITE_TEXT_RESULTS_ROOT),
    ("SMT-Select-Graph+Text", GRAPH_TEXT_RESULTS_ROOT),
]


def latex_escape(s: str) -> str:
    s = str(s)
    for old, new in [
        ("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"),
        ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"),
        ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}"),
    ]:
        s = s.replace(old, new)
    return s


def read_test_gap(summary_path: Path) -> float | None:
    if not summary_path.is_file():
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    agg = data.get("aggregated", {}).get("test", {})
    val = agg.get("gap_cls_par2_mean")
    if val is not None:
        return float(val)
    seeds = data.get("seeds") or []
    values = [float(s["test_metrics"]["gap_cls_par2"]) for s in seeds if s and "test_metrics" in s and "gap_cls_par2" in s["test_metrics"]]
    return sum(values) / len(values) if values else None


def collect_values(root: Path, models: list[str], logics: list[str]) -> dict[tuple[str, str], float]:
    vals: dict[tuple[str, str], float] = {}
    for model in models:
        for logic in logics:
            val = read_test_gap(root / model / logic / "summary.json")
            if val is not None:
                vals[(logic, model)] = val
    return vals


def best_in_group(row_vals: dict[str, float]) -> set[str]:
    if not row_vals:
        return set()
    formatted = {m: f"{v * 100:.2f}" for m, v in row_vals.items()}
    best_model = max(row_vals, key=lambda m: row_vals[m])
    best_str = formatted[best_model]
    return {m for m, f in formatted.items() if f == best_str}


def main() -> None:
    for _, root in GROUPS:
        if not root.is_dir():
            raise FileNotFoundError(f"Results directory not found: {root}")

    models = [m for m in MODEL_ORDER if any((root / m).is_dir() for _, root in GROUPS)]

    all_logics: set[str] = set()
    for _, root in GROUPS:
        for model in models:
            model_dir = root / model
            if model_dir.is_dir():
                for p in model_dir.iterdir():
                    if p.is_dir() and (p / "summary.json").is_file():
                        all_logics.add(p.name)
    logics = sorted(all_logics)

    group_values: list[dict[tuple[str, str], float]] = []
    for _, root in GROUPS:
        group_values.append(collect_values(root, models, logics))

    n_models = len(models)
    n_groups = len(GROUPS)
    ncols = 1 + n_models * n_groups
    col_spec = "l" + "c" * (n_models * n_groups)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
    ]

    # Group header row
    group_headers = [""]
    for group_name, _ in GROUPS:
        group_headers.append(f"\\multicolumn{{{n_models}}}{{c}}{{{group_name}}}")
    lines.append(" & ".join(group_headers) + " \\\\")

    # Cmidrules for each group
    col_offset = 2
    for i in range(n_groups):
        start = col_offset + i * n_models
        end = start + n_models - 1
        lines.append(f"\\cmidrule(lr){{{start}-{end}}}")

    # Model sub-header row
    sub_headers = [""]
    for _ in GROUPS:
        for m in models:
            sub_headers.append(MODEL_DISPLAY.get(m, m))
    lines.append(" & ".join(sub_headers) + " \\\\")
    lines.append("\\midrule")

    # Data rows
    for logic in logics:
        # Find best models within each group
        group_bests: list[set[str]] = []
        for gv in group_values:
            row_vals = {m: gv[(logic, m)] for m in models if (logic, m) in gv}
            group_bests.append(best_in_group(row_vals))

        cells = [latex_escape(logic)]
        for gi, gv in enumerate(group_values):
            for model in models:
                val = gv.get((logic, model))
                if val is not None:
                    pct = f"{val * 100:.2f}"
                    if model in group_bests[gi]:
                        pct = "\\textbf{" + pct + "}"
                    cells.append(pct)
                else:
                    cells.append("---")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{PAR-2 SBS--VBS gap closed (\\%) for description-based models, averaged over five splits.}",
        "\\label{tab:desc}",
        "\\end{table}",
    ])

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
