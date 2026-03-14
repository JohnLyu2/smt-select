# SMT-Select: Multimodal Learning for SMT Algorithm Selection

SMT-Select is a unified framework for SMT algorithm selection that learns representations from both formula AST and natural-language contextual descriptions (such as the ones in SMT-LIB headers).

We offer the following variants, tailored to different data-availability and computational-resource scenarios:

- `SMT-Select (Graph + Text)`: the full multimodal version that uses GNN and pretrained sentence encoder to represent SMT instances.
- `SMT-Select-Graph`: GNN-based algorithm selector.
- `SMT-Select-Text`: SVM-based algorithm selector using descripton embeddings only.
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

Download the corresponding SMT-LIB benchmarks before reproducing results. [scripts/download_smtlib.sh](scripts/download_smtlib.sh) downloads one logic at a time (pass the logic name as the only argument) under `smtlib/`. Example:

```bash
./scripts/download_smtlib.sh QF_BV
```

Our experimental results are stored in [data/results](data/results). The reproduction instructions for each variant are provided below. 

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

Results go to `data/results/text/all-mpnet-base-v2/<logic>/` (e.g. `summary.json`, per-seed CSVs). Use `--model-name` in `encode_desc_logics` for another encoder; omit `--logic` to run all logics that have desc features and splits.

### `SMT-Select-Lite+Text` 

`SMT-Select-Lite+Text` concatenates syntactic features with description embeddings from benchmark descriptions.

Run the experiments:  

```bash
python scripts/experiment_svms.py --logic QF_BV --features-dir data/features/syntactic data/features/desc/all-mpnet-base-v2 --output-dir data/results/lite+text/all-mpnet-base-v2
```

Results are saved to `data/results/lite+text/all-mpnet-base-v2/<logic>/`.

### `SMT-Select-Graph`

To run experiments over the five splits for a given logic (e.g. ABV):

```bash
python scripts/experiment_graph.py --logic ABV
```

This automatically uses performance splits from `data/train_test_splits/ABV/`, SMT-LIB benchmarks under `smtlib/non-incremental/`, and writes results to `data/results/graph/ABV/` (per-seed CSVs plus `summary.json`). Trained GIN models for each split are saved under `models/gin_pwc/ABV/seed<N>/`, and you can rerun only evaluation with `--eval-only` (reusing our saved models under `models/gin_pwc/`) to reproduce results without retraining.

For larger datasets, namely QF_BV, QF_LIA, QF_SLIA, and UFNIA, we enable a `--filter` flag that excludes training instances unsolved by all competition solvers or solved by all solvers within 24 seconds.

### `SMT-Select-Graph+Text`

`SMT-Select-Graph+Text` trains multimodal solver-selection layers based on GIN-based graph embeddings and description embeddings.

We store precomputed GIN graph embeddings for each logic under `data/features/graph/`, and the description embeddings are the same as in `SMT-Select-Text` (e.g. `data/features/desc/all-mpnet-base-v2/`). To (re)compute GIN embeddings for a given logic (e.g. ABV) from your trained GIN models:

```bash
python scripts/extract_gin_embeddings.py ABV
```

This command looks for models under `models/gin_pwc/ABV/seed*/`, benchmarks in `data/raw_data/smtcomp24_performance/ABV.json`, and writes seed-wise graph features to `data/features/graph/ABV/seed*/`.

To run the multimodal fusion experiments over the five splits for a logic (e.g. ABV), using precomputed graph and text features:

```bash
python scripts/experiment_fusion.py --logic ABV
```

This will load the corresponding graph and text feature directories, train the fusion model across all seeds, and write results to `data/results/graph+text/all-mpnet-base-v2/ABV/` (per-seed `train_eval.csv`/`test_eval.csv` plus `summary.json`). Trained fusion models are saved under `models/fusion_pwc/ABV/seed<N>/`. 

Our saved models live under `models/fusion_pwc/`, and you can also run evaluation only using the `--eval-only` flag.

### Baselines

We evaluate `MachSMT` at [e524a61](https://github.com/MachSMT/MachSMT/commit/e524a617c1fc79a33c6c797b5c72a8f9650f89e3) and `Sibyl` at [133d33f](https://github.com/will-leeson/sibyl/commit/133d33f1a473eb46d4f6c09a410223907cab6104).