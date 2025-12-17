import subprocess
from pathlib import Path

# ================= CONFIG =================
BASE = Path("/home/paul/Dokumente/SMT_AS_LLM/data")

GENERAL_DATA_DIR = BASE / "generalData"
MODELS_DIR = Path("/home/paul/Dokumente/SMT_AS_LLM/models")
OUTPUT_BASE = BASE / "folks26" / "selector_res"

FEATURE_PATHS = {
    "embeddings": BASE / "embeddings",
    "embeddings_combined": BASE / "embeddings_combined",
    "features": BASE / "features",
    "features_embeddings": BASE / "features+Embedding",
}

FEATURE_SUFFIX = {
    "embeddings": "_description_embeddings.csv",
    "embeddings_combined": "_description_with_features_embeddings.csv",
    "features": "_features_filtered.csv",
    "features_embeddings": "_feat+emb.csv",
}

OUTPUT_NAMES = {
    "embeddings": "des.csv",
    "embeddings_combined": "des_exp.csv",
    "features": "syn.csv",
    "features_embeddings": "des_syn.csv",
}

SPLITS = ["train", "test"]

# =========================================
def resolve_model_path(path: Path) -> Path:
    s = str(path)

    triple = "model.joblib/model.joblib/model.joblib"
    double = "model.joblib/model.joblib"
    single = "model.joblib"

    if s.endswith(triple):
        # remove ONE
        return Path(s[:-len("/model.joblib")])

    if s.endswith(double):
        # fine
        return path

    if s.endswith(single):
        # add ONE
        return path / "model.joblib"

    # otherwise: do nothing
    return path

def discover_logics(general_data_dir):
    k = sorted(
        f.name.removesuffix("_train.csv")
        for f in general_data_dir.glob("*_train.csv")
    )
    print(k)
    return k

def parse_model_dir(dir_name: str, known_logics):
    # strip training artefact
    if dir_name.endswith("_model.joblib"):
        base = dir_name.removesuffix("_model.joblib")
    else:
        base = dir_name

    # longest-prefix match
    for logic in sorted(known_logics, key=len, reverse=True):
        if base.startswith(logic):
            rest = base[len(logic):].lstrip("_").lower()

            if "feature" in rest and "embedding" in rest:
                source = "features_embeddings"
            elif "feature" in rest:
                source = "features"
            elif "combined" in rest:
                source = "embeddings_combined"
            elif "embedding" in rest:
                source = "embeddings"
            else:
                raise ValueError(f"Cannot infer feature source from: {dir_name}")

            return logic, source

    raise ValueError(f"No matching logic for model dir: {dir_name}")

    return logic, source
model_dirs = [
    d for d in MODELS_DIR.iterdir()
    if d.is_dir() and (d / "model.joblib").exists()
]

print(f"🧠 Found {len(model_dirs)} trained models")

for model_dir in sorted(model_dirs):

    logic, source = parse_model_dir(model_dir.name, discover_logics(GENERAL_DATA_DIR))

    for split in SPLITS:
        perf_csv = GENERAL_DATA_DIR / f"{logic}_{split}.csv"
        if not perf_csv.exists():
            print(f"⚠️ Missing {perf_csv.name} → skipping")
            continue

        feature_csv = FEATURE_PATHS[source] / f"{logic}{FEATURE_SUFFIX[source]}"
        if not feature_csv.exists():
            print(f"⚠️ Missing features for {logic} ({source}) → skipping")
            continue

        output_dir = OUTPUT_BASE / logic / split
        output_dir.mkdir(parents=True, exist_ok=True)

        output_csv = output_dir / OUTPUT_NAMES[source]

        print(f"\n🚀 Evaluating")
        print(f"   logic:  {logic}")
        print(f"   split:  {split}")
        print(f"   source: {source}")
        print(f"   model:  {model_dir}")
        print(f"   out:    {output_csv}")
        model_file = resolve_model_path(model_dir)
        print(model_file)
        cmd = [
            "python3", "-m", "src.evaluate",
            "--model", str(model_file),  # ✅ directory
            "--perf-csv", str(perf_csv),
            "--feature-csv", str(feature_csv),
            "--output-csv", str(output_csv),
        ]

        subprocess.run(cmd, check=True)

print("\n✅ All evaluations completed")
