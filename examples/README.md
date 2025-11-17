# Examples

This folder contains small runnable examples and experiments that illustrate concepts from Goodfellow, Bengio & Courville's "Deep Learning".

Files
- `toy_mlp.py` — a minimal multilayer perceptron implemented with NumPy (educational, no ML framework required).

Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy
python examples/toy_mlp.py
```

Notes
- The example is intentionally tiny and numeric-stability / performance are not production-quality. It's meant to show core ideas (forward, backward, SGD) in plain Python + NumPy.
