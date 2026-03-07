# SMT-Select: Multimodal Learning for SMT Algorithm Selection

SMT-Select is a unified framework for SMT algorithm selection that learns representations from both formula AST and natural-language contextual descriptions (such as the ones in SMT-LIB headers).

We offer the following variants tailored to different data-availability and computational-resource scenrios:

- SMT-Select (Graph + Text): the full multimodal version that uses GNN and pretrained sentence embedder to represent SMT instances.
- SMT-Select-Graph: GNN-based algorithm selector.
- SMT-Lite: SVM-based algorithm selector using lightweight syntactic features.
- SMT-Lite+Text: SMT-Lite combined with textual embeddings.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```