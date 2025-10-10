mnist-unlearning
=================

Small demo project showing training a LeNet model on MNIST, then training with one class removed.

Setup
-----

Create a virtualenv, install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train
-----

Train on full MNIST:

```bash
python train.py --epochs 3
```

Train with class 7 removed:

```bash
python -c "from train import run_training; run_training(epochs=3, remove_label=7)"
```

Serve
-----

Start the FastAPI app with uvicorn:

```bash
uvicorn app:app --reload
```

Then POST to /predict with raw image bytes and optional `model_checkpoint` path.
