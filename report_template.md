# Text-to-Python Seq2Seq Report

## Experimental Setup
- Dataset: CodeSearchNet Python
- Train/Val/Test sizes:
- Tokenization: whitespace
- Max lengths: docstring 50, code 80
- Embedding dim:
- Hidden dim:
- Optimizer: Adam
- Loss: Cross-entropy
- Teacher forcing:
- Epochs:

## Results

### Loss Curves
Insert plots from `outputs/*_loss.csv`.

### Quantitative Metrics
Insert results from `outputs/*_metrics.csv`.

| Model | Token Accuracy | BLEU | Exact Match |
| --- | --- | --- | --- |
| RNN |  |  |  |
| LSTM |  |  |  |
| Attention |  |  |  |

### Error Analysis
- Syntax errors:
- Missing indentation:
- Incorrect operators/variables:

### Performance vs Docstring Length
Summarize length-bucket exact match from `outputs/*_metrics.csv`.

## Attention Analysis
Include 3 attention heatmaps from `figures/attention_*.png` and interpret alignment.

## Discussion
- Compare RNN vs LSTM vs Attention.
- Discuss long-range dependency handling and context bottleneck.

## Conclusion
