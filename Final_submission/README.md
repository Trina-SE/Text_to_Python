# Docstring-to-Code Generation using Seq2Seq Models

Automatic Python code generation from natural language docstrings using three sequence-to-sequence architectures implemented in PyTorch.

## Project Overview

This project implements and compares three neural sequence-to-sequence models that learn to generate Python code from English docstring descriptions:

| Model | Encoder | Decoder | Attention |
|-------|---------|---------|-----------|
| **Vanilla RNN Seq2Seq** | RNN | RNN | None (fixed context) |
| **LSTM Seq2Seq** | LSTM | LSTM | None (fixed context) |
| **LSTM with Attention** | Bidirectional LSTM | LSTM | Bahdanau (additive) |

## Dataset

**CodeSearchNet – Python** ([Hugging Face](https://huggingface.co/datasets/Nan-Do/code-search-net-python))

- ~250,000+ English docstring / Python function pairs from real GitHub repositories
- Training subset: 10,000 examples
- Validation: 1,500 examples | Test: 1,500 examples
- Max docstring length: 50 tokens | Max code length: 80 tokens

## Project Structure

```
Final_submission/
├── src/                            # Shared Python modules
│   ├── config.py                   # Hyperparameters and paths
│   ├── data.py                     # Dataset loading, tokenization, vocabulary
│   ├── models.py                   # All three model architectures
│   ├── train_utils.py              # Training and validation loops
│   └── eval_utils.py               # BLEU, accuracy, generation, error analysis
├── Notebook/                       # Training notebooks (one per model)
│   ├── 01_Vanilla_RNN_Seq2Seq.ipynb
│   ├── 02_LSTM_Seq2Seq.ipynb
│   └── 03_LSTM_with_Attention.ipynb
├── Checkpoints/                    # Saved model checkpoints
├── Attention_Visualizations/       # Attention heatmap analysis
│   └── attention_analysis.ipynb
├── Evaluation/                     # Comparative evaluation
│   └── model_comparison.ipynb
├── main.py                         # Full pipeline orchestrator
├── Dockerfile                      # Docker image definition
├── requirements.txt                # Python dependencies
├── run.sh                          # Linux/Mac run script
├── run.bat                         # Windows run script
└── README.md                       # This file
```

## Quick Start with Docker (Recommended)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed
- (Optional) NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU acceleration

### Run the Full Pipeline

**Linux / macOS:**
```bash
chmod +x run.sh
./run.sh
```

**Windows:**
```cmd
run.bat
```

This single command will:
1. Build the Docker image with all dependencies
2. Download the CodeSearchNet Python dataset
3. Train all three models (Vanilla RNN, LSTM, LSTM+Attention)
4. Evaluate models on the test set
5. Generate attention visualizations
6. Save all outputs to the respective folders

### Run Options

```bash
# Train only a specific model
./run.sh --model rnn
./run.sh --model lstm
./run.sh --model attention

# Change number of epochs
./run.sh --epochs 5

# Only evaluate (requires existing checkpoints)
./run.sh --eval-only
```

## Running Without Docker

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
pip install -r requirements.txt
```

### Train All Models

```bash
python main.py
```

### Run Individual Notebooks

```bash
# Start Jupyter
jupyter notebook

# Then open notebooks from:
#   Notebook/          - Training notebooks
#   Evaluation/        - Model comparison
#   Attention_Visualizations/ - Attention analysis
```

## Model Architectures

### Model 1: Vanilla RNN Seq2Seq (Baseline)
- Standard RNN encoder compresses the entire docstring into a fixed-length context vector
- RNN decoder generates code token-by-token from this context
- Establishes baseline performance; expected to degrade on longer inputs

### Model 2: LSTM Seq2Seq
- LSTM gating mechanisms (forget, input, output gates) improve long-range dependency modeling
- Cell state provides a highway for gradient flow
- Still limited by fixed-length context vector

### Model 3: LSTM with Bahdanau Attention
- Bidirectional LSTM encoder captures both forward and backward context
- Bahdanau (additive) attention computes dynamic context at each decoding step
- Removes the information bottleneck of fixed context vectors
- Enables interpretability through attention weight visualization

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 256 |
| Hidden dimension | 256 |
| Number of layers | 2 |
| Dropout | 0.3 |
| Optimizer | Adam (lr=0.001) |
| Loss function | Cross-entropy (with PAD masking) |
| Teacher forcing ratio | 0.5 |
| Gradient clipping | 1.0 |
| Batch size | 64 |
| Epochs | 15 |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Training/Validation Loss** | Cross-entropy loss to monitor convergence |
| **Token Accuracy** | Percentage of correctly predicted tokens |
| **BLEU Score** | N-gram overlap between generated and reference code |
| **Exact Match Accuracy** | Percentage of completely correct outputs |

## Analysis

The evaluation notebook (`Evaluation/model_comparison.ipynb`) provides:

1. **Training and validation loss curves** for all three models
2. **BLEU score comparison** on the test set
3. **Error analysis** — syntax errors, missing indentation, incorrect operators
4. **Performance vs. docstring length** — how each model degrades with longer inputs

The attention notebook (`Attention_Visualizations/attention_analysis.ipynb`) provides:

1. **Attention heatmaps** for at least three test examples
2. **Alignment analysis** between docstring tokens and generated code tokens
3. **Semantic relevance** — whether attention focuses on meaningful words (e.g., does "maximum" attend to `max()` or `>`)

## Outputs

After running the pipeline:

- **`Checkpoints/`** — Best and final model checkpoints (`.pt` files)
- **`Evaluation/`** — `training_curves.png`, `model_comparison.png`, `bleu_vs_length.png`, `results.json`
- **`Attention_Visualizations/`** — `attention_example_1.png`, `attention_example_2.png`, `attention_example_3.png`
