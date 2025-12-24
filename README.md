# Text-to-Python Code Generation (Seq2Seq RNN/LSTM/Attention)

This project implements and compares three sequence-to-sequence models that translate English docstrings into Python code using PyTorch:

- Vanilla RNN Seq2Seq
- LSTM Seq2Seq
- LSTM + Bahdanau Attention

The workflow includes training, evaluation (token accuracy, BLEU, exact match), and attention visualization.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run all experiments locally

```bash
python run_all.py --device cpu --epochs 10
```

Outputs:

- `outputs/*_loss.csv` training/validation loss curves
- `outputs/*_metrics.csv` BLEU/token accuracy/exact match
- `checkpoints/*_best.pt` model checkpoints
- `figures/attention_*.png` attention heatmaps (attention model only)

## Train one model

```bash
python -m text2python.train --model rnn --device cpu --epochs 10
python -m text2python.train --model lstm --device cpu --epochs 10
python -m text2python.train --model attention --device cpu --epochs 10
```

## Evaluate a checkpoint

```bash
python -m text2python.eval --checkpoint checkpoints/attention_best.pt --device cpu
```

## Attention visualization (3 examples)

```bash
python -m text2python.attention --checkpoint checkpoints/attention_best.pt --indices 0 1 2 --device cpu
```

## Docker (single script)

```powershell
.\run_docker.ps1 -Device cpu -Epochs 10
```

## Notes

- Dataset: CodeSearchNet Python (`Nan-Do/code-search-net-python`) via Hugging Face.
- Tokenization: whitespace.
- Max lengths: docstring 50 tokens, code 80 tokens.
- Training subsets: 8k train / 1k val / 1k test (adjust via CLI).
