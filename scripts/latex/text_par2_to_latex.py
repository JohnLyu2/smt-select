#!/usr/bin/env python3
"""
Generate a LaTeX table of PAR-2 gap closed (%) for text-only models.

Reads summary.json from data/cp26/results/text/<model>/<logic>/summary.json,
extracts test gap_cls_par2_mean, and writes doc/cp26/text_par2.tex.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXT_RESULTS_ROOT = PROJECT_ROOT / "data" / "cp26" / "results" / "text"
TEX_PATH = PROJECT_ROOT / "doc" / "cp26" / "desc.tex"

MODEL_ORDER = ["all-mpnet-base-v2", "embeddinggemma-300m", "Qwen3-Embedding-0.6B"]
MODEL_DISPLAY = {
    "all-mpnet-base-v2": "MPNet",
    "embeddinggemma-300m": "GemmaEmb",
    "Qwen3-Embedding-0.6B": "Qwen3Emb",
}


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


def main() -> None:
    if not TEXT_RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"Text results directory not found: {TEXT_RESULTS_ROOT}")

    models = [m for m in MODEL_ORDER if (TEXT_RESULTS_ROOT / m).is_dir()]

    all_logics: set[str] = set()
    for model in models:
        for p in (TEXT_RESULTS_ROOT / model).iterdir():
            if p.is_dir() and (p / "summary.json").is_file():
                all_logics.add(p.name)
    logics = sorted(all_logics)

    # Collect values: {(logic, model): float}
    values: dict[tuple[str, str], float] = {}
    for model in models:
        for logic in logics:
            val = read_test_gap(TEXT_RESULTS_ROOT / model / logic / "summary.json")
            if val is not None:
                values[(logic, model)] = val

    ncols = 1 + len(models)
    col_spec = "l" + "c" * len(models)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join([""] + [MODEL_DISPLAY.get(m, m) for m in models]) + " \\\\",
        "\\midrule",
    ]

    for logic in logics:
        row_vals = {m: values[(logic, m)] for m in models if (logic, m) in values}
        row_vals = {m: values[(logic, m)] for m in models if (logic, m) in values}
        formatted: dict[str, str] = {m: f"{v * 100:.2f}" for m, v in row_vals.items()}
        if row_vals:
            best_model = max(row_vals, key=lambda m: row_vals[m])
            best_str = formatted[best_model]
            best_models = {m for m, f in formatted.items() if f == best_str}
        else:
            best_models = set()

        cells = [latex_escape(logic)]
        for model in models:
            if (logic, model) in values:
                pct = formatted[model]
                if model in best_models:
                    pct = "\\textbf{" + pct + "}"
                cells.append(pct)
            else:
                cells.append("---")
        lines.append(" & ".join(cells) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Text-only model results: PAR-2 SBS--VBS gap closed (\\%), averaged over five splits.}",
        "\\label{tab:desc}",
        "\\end{table}",
    ])

    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {TEX_PATH}")


if __name__ == "__main__":
    main()
