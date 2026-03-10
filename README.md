# SMT-Select: Multimodal Learning for SMT Algorithm Selection

SMT-Select is a unified framework for SMT algorithm selection that learns representations from both formula AST and natural-language contextual descriptions (such as the ones in SMT-LIB headers).

We offer the following variants, tailored to different data-availability and computational-resource scenarios:

- `SMT-Select (Graph + Text)`: the full multimodal version that uses GNN and pretrained sentence embedder to represent SMT instances.
- `SMT-Select-Graph`: GNN-based algorithm selector.
- `SMT-Lite`: SVM-based algorithm selector using lightweight syntactic features.
- `SMT-Lite+Text`: SMT-Lite combined with textual embeddings.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Reproduction of CP'26 Submission 

### Data Overview and Preparation

Raw performance data lives in [data/raw_data/smtcomp24_performance/](data/raw_data/smtcomp24_performance/). Five random train-test splits per logic are in [data/train_test_splits/](data/train_test_splits/) (create them with [scripts/split_data_json.py](scripts/split_data_json.py) and the desired seeds). 

\TODO{Saved results}

Download the SMT benchmarks before reproducing results. [scripts/download_smtlib.sh](scripts/download_smtlib.sh) downloads one logic at a time (pass the logic name as the only argument) under `smtlib/`. Example:

```bash
./scripts/download_smtlib.sh QF_BV
```

### `SMT-Select-Lite` 

Precomputed syntactic features for all logic benchmarks, plus extraction times from [Klammerhammer](https://github.com/SMT-LIB/SMT-LIB-db/tree/main/klhm), are in [data/features/syntactic](data/features/syntactic). To run SVM experiments over the five splits for a given logic (e.g. QF_BV):

```bash
python scripts/experiment_svms.py --logic QF_BV --features-dir data/features/syntactic --output-dir data/results/lite
```

Results are saved to `data/results/lite/<logic>/` (e.g. `summary.json`, and per-seed `seed<N>/train_eval.csv`, `seed<N>/test_eval.csv`).

### `SMT-Select-Text`

Descriptions are saved in [data/descriptions/](data/descriptions/). Encode descriptions to feature CSVs (by default using the [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model):

```bash
python scripts/encode_desc_logics.py --output-dir data/features/desc/all-mpnet-base-v2
```

Then run the experiments:

```bash
python scripts/experiment_svms.py --logic QF_BV --features-dir data/features/desc/all-mpnet-base-v2 --output-dir data/results/text/all-mpnet-base-v2
```

Results go to `data/results/text/<model>/<logic>/` (e.g. `summary.json`, per-seed CSVs). Use `--model-name` in `encode_desc_logics` for another encoder; omit `--logic` to run all logics that have desc features and splits.

### `SMT-Select-Lite+Text` 

`SMT-Select-Lite+Text` concatenates syntactic features with description embeddings from benchmark descriptions.

Run the experiments:  


```bash
python scripts/experiment_svms.py --logic QF_BV --features-dir data/features/syntactic data/features/desc/all-mpnet-base-v2 --output-dir data/results/lite+text/all-mpnet-base-v2
```

Results are saved to `data/results/lite+text/<model>/<logic>/`.


