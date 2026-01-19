import argparse
import json
from pathlib import Path

from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from .parser import parse_performance_csv


def create_setfit_data(
    perf_csv_path: str,
    desc_json_path: str,
    timeout: float = 1200.0,
) -> dict[str, list]:
    """
    Create SetFit training data from performance CSV and description JSON.

    Args:
        perf_csv_path: Path to performance CSV file.
        desc_json_path: Path to description JSON file.
        timeout: Timeout value in seconds.
    Returns:
        Dict with keys: "texts", "labels", "paths".
    """
    desc_map = _load_description_map(desc_json_path)
    multi_perf_data = parse_performance_csv(perf_csv_path, timeout)

    texts: list[str] = []
    labels: list[str] = []
    paths: list[str] = []

    for path in multi_perf_data.keys():
        description = desc_map.get(path)
        if not description or not description.strip():
            raise AssertionError(f"Missing description for benchmark: {path}")

        if multi_perf_data.is_none_solved(path):
            continue

        solver_name = multi_perf_data.get_best_solver_for_instance(path)
        if solver_name is None:
            continue

        label = solver_name

        texts.append(description.strip())
        labels.append(label)
        paths.append(path)

    return {"texts": texts, "labels": labels, "paths": paths}


def _load_description_map(desc_json_path: str) -> dict[str, str]:
    json_path = Path(desc_json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    desc_map: dict[str, str] = {}
    for benchmark in benchmarks:
        smtlib_path = benchmark.get("smtlib_path", "")
        if not smtlib_path:
            continue

        description = benchmark.get("description", "")
        if not description or not description.strip():
            raise AssertionError(f"Missing description for benchmark: {smtlib_path}")

        desc_map[smtlib_path] = description.strip()

    return desc_map


def train_setfit_model(
    train_data: dict[str, list],
    test_data: dict[str, list] | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str | None = None,
    num_epochs: int = 1,
    batch_size: int = 16,
) -> SetFitModel:
    if "texts" not in train_data or "labels" not in train_data:
        raise ValueError("Expected train_data to include 'texts' and 'labels' keys.")

    texts = train_data["texts"]
    labels = train_data["labels"]
    if len(texts) != len(labels):
        raise ValueError("texts and labels must be the same length.")

    train_dataset = Dataset.from_dict({"text": texts, "label": labels})
    eval_dataset = None
    if test_data is not None:
        test_texts = test_data.get("texts", [])
        test_labels = test_data.get("labels", [])
        if len(test_texts) != len(test_labels):
            raise ValueError("test texts and labels must be the same length.")
        eval_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

    model = SetFitModel.from_pretrained(model_name)
    training_args = TrainingArguments(
        batch_size=batch_size,
        num_epochs=num_epochs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()

    if output_dir:
        model.save_pretrained(output_dir)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create SetFit data from performance CSV and descriptions JSON."
    )
    parser.add_argument(
        "--perf-csv",
        required=True,
        help="Path to performance CSV.",
    )
    parser.add_argument("--desc-json", required=True, help="Path to descriptions JSON.")
    parser.add_argument(
        "--timeout",
        default=1200.0,
        type=float,
        help="Timeout value used in performance data (seconds).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write SetFit data as JSON.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name for SetFit training.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save the trained SetFit model.",
    )
    parser.add_argument(
        "--test-perf-csv",
        default=None,
        help="Optional performance CSV for test data.",
    )
    parser.add_argument(
        "--num-epochs",
        default=1,
        type=int,
        help="Number of SetFit training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="SetFit training batch size.",
    )

    args = parser.parse_args()

    data = create_setfit_data(args.perf_csv, args.desc_json, args.timeout)
    print(f"Total samples: {len(data['texts'])}")
    print(f"Unique labels: {len(set(data['labels']))}")

    if args.output_json:
        output_path = Path(args.output_json)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote SetFit data to: {output_path}")

    test_data = None
    if args.test_perf_csv:
        test_data = create_setfit_data(args.test_perf_csv, args.desc_json, args.timeout)

    train_setfit_model(
        train_data=data,
        test_data=test_data,
        model_name=args.model_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
