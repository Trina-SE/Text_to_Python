# Src Module Overview

This folder contains the core implementation of the Text-to-Python Seq2Seq models.

## Files and Purpose

- `__init__.py`
  - Declares the package and exposes top-level modules.

- `attention.py`
  - Loads the attention model checkpoint and generates attention heatmap images for selected test examples.

- `config.py`
  - Defines the `Config` dataclass with default hyperparameters and runtime options.

- `data.py`
  - Loads the CodeSearchNet dataset, filters and tokenizes samples, builds fixed-length examples, and creates PyTorch DataLoaders.

- `eval.py`
  - Loads a trained checkpoint and evaluates it on the test set (token accuracy, BLEU, exact match, and length buckets).

- `metrics.py`
  - Implements token accuracy, exact match, and BLEU helpers used during evaluation.

- `models.py`
  - Implements the three model variants: RNN Seq2Seq, LSTM Seq2Seq, and LSTM with Bahdanau Attention.

- `train.py`
  - Trains the selected model, saves the best checkpoint, and logs loss curves.

- `vocab.py`
  - Builds and manages vocabularies for source docstrings and target code sequences.
