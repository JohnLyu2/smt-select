# SMT-Select: Multimodal Learning for SMT Algorithm Selection

SMT-Select is a unified framework for SMT algorithm selection that learns representations from both formula AST and natural-language contextual descriptions (such as the ones in SMT-LIB headers).

We offer the following variants tailored to different data-availability and computational-resource scenrios:

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

## Reprodcution of CP'26 Submission 

### Data Overview and Preparation

The raw performance data is at [data/raw_data/smtcomp24_performance/](data/raw_data/smtcomp24_performance/). The five random train-test performance splits are at [data/train_test_splits/](data/train_test_splits/) (created with [scripts/split_data_json.py](scripts/split_data_json.py) with the respective seeds). 

\TODO{Saved results}

Before reproducing the results, you need to download the respective SMT benchmarks. [scripts/download_smtlib.sh](scripts/download_smtlib.sh) downloads benchmarks for one logic at a time; pass the logic name as the only argument. Benchmarks are extracted under `smtlib/`. Example:

```bash
./scripts/download_smtlib.sh QF_BV
```




