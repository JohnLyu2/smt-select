import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")


from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
import torch

from .performance import parse_performance_json
from .solver_selector import SolverSelector


def create_setfit_data(
    perf_path: str,
    desc_json_path: str,
    timeout: float = 1200.0,
    include_all_solved: bool = False,
    multi_label: bool = False,
) -> dict:
    """
    Create SetFit training data from performance data and description JSON.

    Performance data must be JSON (e.g. data/cp26/raw_data/smtcomp24_performance/LOGIC.json).
    Solver order (for multi_label and for saving) is taken from the performance data.
    Returns solver_id_dict (id -> solver name) so callers can save solver2id for evaluation.

    Args:
        perf_path: Path to performance JSON file.
        desc_json_path: Path to description JSON file.
        timeout: Timeout value in seconds.
        include_all_solved: Whether to include instances solved by all solvers.
        multi_label: Whether to return all solving solvers as labels (multi-hot list).
    Returns:
        Dict with keys: "texts", "labels", "paths", "solver_id_dict" (id -> name).
    """
    desc_map = _load_description_map(desc_json_path)
    multi_perf_data = parse_performance_json(perf_path, timeout)
    solver_id_dict = multi_perf_data.get_solver_id_dict()
    solver_name_to_id = {name: idx for idx, name in solver_id_dict.items()}
    num_solvers = len(solver_id_dict)

    texts: list[str] = []
    labels: list = []
    paths: list[str] = []

    for path in multi_perf_data.keys():
        description = desc_map.get(path)
        if not description or not description.strip():
            raise AssertionError(f"Missing description for benchmark: {path}")

        if multi_perf_data.is_none_solved(path):
            continue
        if not include_all_solved and multi_perf_data.is_all_solved(path):
            continue

        if multi_label:
            solver_names = multi_perf_data.get_solvers_solving_instance(path)
            if not solver_names:
                continue
            label = [0] * num_solvers
            for name in solver_names:
                if name in solver_name_to_id:
                    label[solver_name_to_id[name]] = 1
                else:
                    logging.warning("Solver %r not in performance data, skipping", name)
        else:
            solver_name = multi_perf_data.get_best_solver_for_instance(path)
            if solver_name is None:
                continue
            label = solver_name

        texts.append(description.strip())
        labels.append(label)
        paths.append(path)

    return {
        "texts": texts,
        "labels": labels,
        "paths": paths,
        "solver_id_dict": solver_id_dict,
    }


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


def _load_solver2id_from_model_dir(model_path: str) -> dict[str, int] | None:
    """Load solver2id (name -> id) from model_dir/solver2id.json if present."""
    path = Path(model_path)
    if not path.is_dir():
        return None
    f = path / "solver2id.json"
    if not f.is_file():
        return None
    with open(f, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    return {k: int(v) for k, v in data.items()}


class SetfitSelector(SolverSelector):
    def __init__(self, setfit_model: str, desc_json_path: str) -> None:
        self.setfit_model = setfit_model
        self.desc_json_path = desc_json_path
        self._desc_map = _load_description_map(desc_json_path)
        self._model = self._load_model(setfit_model)
        self._label_to_id = _load_solver2id_from_model_dir(setfit_model)
        if self._label_to_id is None:
            raise ValueError(
                "solver2id.json not found in model dir. Train with scripts/finetune_desc_setfit.py "
                "or add solver2id.json (solver name -> id) to the model directory."
            )

    def algorithm_select(self, instance_path: str | Path) -> int:
        # TODO: This selector does not yet support multi-label models.
        # It assumes a single-label prediction (string or int).
        instance_key = str(instance_path)
        description = self._desc_map.get(instance_key)
        if not description or not description.strip():
            raise ValueError(f"Missing description for benchmark: {instance_key}")

        predicted = self._model.predict([description.strip()])[0]
        if isinstance(predicted, int):
            return predicted
        if isinstance(predicted, str):
            solver_id = self._label_to_id.get(predicted)
            if solver_id is None:
                raise ValueError(f"Unknown solver label: {predicted}")
            return solver_id
        raise TypeError(f"Unsupported prediction type: {type(predicted)}")

    def _load_model(self, setfit_model: str) -> SetFitModel:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            return SetFitModel.from_pretrained(setfit_model, device=device)
        except TypeError:
            model = SetFitModel.from_pretrained(setfit_model)
            try:
                model.to(device)
            except Exception:
                pass
            return model


def train_setfit_model(
    train_data: dict[str, list],
    test_data: dict[str, list] | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_dir: str | None = None,
    num_epochs: int = 1,
    batch_size: int = 16,
    multi_label: bool = False,
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

    logging.info("Training samples: %s", len(train_dataset))
    if eval_dataset is not None:
        logging.info("Eval samples: %s", len(eval_dataset))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)
    logging.info("Base model: %s", model_name)
    logging.info("Training specs: epochs=%s, batch_size=%s, multi_label=%s", num_epochs, batch_size, multi_label)
    
    multi_target_strategy = "one-vs-rest" if multi_label else None
    
    try:
        model = SetFitModel.from_pretrained(
            model_name, 
            device=device, 
            multi_target_strategy=multi_target_strategy
        )
    except TypeError:
        model = SetFitModel.from_pretrained(
            model_name, 
            multi_target_strategy=multi_target_strategy
        )
        try:
            model.to(device)
        except Exception:
            pass

    try:
        training_args = TrainingArguments(
            batch_size=batch_size,
            num_epochs=num_epochs,
            device=device,
        )
    except TypeError:
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
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete.")

    if output_dir:
        model.save_pretrained(output_dir)

    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    parser = argparse.ArgumentParser(
        description="Create SetFit data from performance JSON and descriptions JSON."
    )
    parser.add_argument(
        "--train-perf-json",
        required=True,
        help="Path to training performance JSON (e.g. data/cp26/raw_data/smtcomp24_performance/LOGIC.json).",
    )
    parser.add_argument("--desc-json", required=True, help="Path to descriptions JSON.")
    parser.add_argument(
        "--timeout",
        default=1200.0,
        type=float,
        help="Timeout value used in performance data (seconds).",
    )
    parser.add_argument(
        "--train-data-json",
        default=None,
        help="Optional path to write training data as JSON.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base model name for SetFit training.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory to save the trained SetFit model.",
    )
    parser.add_argument(
        "--test-perf-json",
        default=None,
        help="Optional performance JSON for test data.",
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
    parser.add_argument(
        "--include-all-solved",
        action="store_true",
        help="Include instances solved by all solvers in training data.",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Use multi-label training (all solvers that solve an instance).",
    )

    args = parser.parse_args()

    train_data = create_setfit_data(
        args.train_perf_json,
        args.desc_json,
        args.timeout,
        include_all_solved=args.include_all_solved,
        multi_label=args.multi_label,
    )
    solver_id_dict = train_data["solver_id_dict"]
    logging.info("Total samples: %s", len(train_data["texts"]))
    if args.multi_label:
        logging.info("Multi-label mode enabled. Label vector length: %s", len(solver_id_dict))
    else:
        logging.info("Unique labels: %s", len(set(train_data["labels"])))

    if args.train_data_json:
        output_path = Path(args.train_data_json)
        # Serialize with string keys for JSON
        out_data = {k: v for k, v in train_data.items() if k != "solver_id_dict"}
        out_data["solver_id_dict"] = {str(i): name for i, name in solver_id_dict.items()}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        logging.info("Wrote SetFit data to: %s", output_path)

    test_data = None
    if args.test_perf_json:
        test_data = create_setfit_data(
            args.test_perf_json,
            args.desc_json,
            args.timeout,
            include_all_solved=args.include_all_solved,
            multi_label=args.multi_label,
        )

    train_setfit_model(
        train_data=train_data,
        test_data=test_data,
        model_name=args.model_name,
        output_dir=args.model_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        multi_label=args.multi_label,
    )

    # Save solver name -> id so SetfitSelector can load it at eval time
    model_dir = Path(args.model_dir)
    if model_dir.is_dir():
        solver2id = {name: idx for idx, name in solver_id_dict.items()}
        with open(model_dir / "solver2id.json", "w", encoding="utf-8") as f:
            json.dump(solver2id, f, indent=2)
        logging.info("Wrote solver2id.json to: %s", model_dir / "solver2id.json")


if __name__ == "__main__":
    main()
