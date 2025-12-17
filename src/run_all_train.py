import csv
import subprocess
import time
from pathlib import Path

# ================= CONFIG =================
GENERAL_DATA_DIR = Path("/home/paul/Dokumente/SMT_AS_LLM/data/generalData")

FEATURE_SOURCES = {
    "embeddings": {
        "dir": Path("/home/paul/Dokumente/SMT_AS_LLM/data/embeddings"),
        "suffix": "_description_embeddings.csv",
    },
    "embeddings_combined": {
        "dir": Path("/home/paul/Dokumente/SMT_AS_LLM/data/embeddings_combined"),
        "suffix": "_description_with_features_embeddings.csv",
    },
    "features": {
        "dir": Path("/home/paul/Dokumente/SMT_AS_LLM/data/features"),
        "suffix": "_features_filtered.csv",
    },
    "features_embeddings": {
        "dir": Path("/home/paul/Dokumente/SMT_AS_LLM/data/features+Embedding"),
        "suffix": "_feat+emb.csv",
    },
}

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
TIMINGS_CSV = Path("training_times.csv")
# Known logic prefixes (ordered longest → shortest)
KNOWN_LOGICS = ["abv", "alia", "bv", "qf_idl", "qf_lia", "qf_lra", "qf_nra", "qf_slia", "uf", "ufdt", "uflia", "ufnia"]

# ==========================================


def extract_logic(filename: str) -> str:
    for logic in KNOWN_LOGICS:
        if filename.startswith(logic + "_"):
            return logic
    raise ValueError(f"Unknown logic in filename: {filename}")


write_header = not TIMINGS_CSV.exists()

with open(TIMINGS_CSV, "a", newline="") as f:
    writer = csv.writer(f)

    if write_header:
        writer.writerow([
            "logic",
            "feature_source",
            "training_time_seconds",
            "model_file",
            "perf_csv",
            "feature_csv",
        ])

    for perf_csv in sorted(GENERAL_DATA_DIR.glob("*_train.csv")):
        logic = extract_logic(perf_csv.name)

        print(f"\n==============================")
        print(f"🧠 Logic: {logic}")
        print(f"==============================")

        for source_name, cfg in FEATURE_SOURCES.items():
            feature_csv = cfg["dir"] / f"{logic}{cfg['suffix']}"

            if not feature_csv.exists():
                print(f"⚠️  Missing {source_name} → skipping")
                continue

            model_name = f"{logic}_{source_name}_model.joblib"

            print(f"\n🚀 Training with: {source_name}")
            start = time.time()

            cmd = [
                "python3",
                "-m", "src.pwc",
                "--save-dir", str(MODEL_DIR) + "/" + model_name,
                "--perf-csv", str(perf_csv),
                "--feature-csv", str(feature_csv),
            ]

            subprocess.run(cmd, check=True)

            duration = time.time() - start

            writer.writerow([
                logic,
                source_name,
                f"{duration:.4f}",
                model_name,
                perf_csv.name,
                feature_csv.name,
            ])

            f.flush()  # ensure data is written immediately

            print(f"✅ Done in {duration:.2f}s → {model_name}")

print("\n📊 Timings saved to training_times.csv")
