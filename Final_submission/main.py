"""
Main orchestrator script for training and evaluating all four
Seq2Seq models for docstring-to-code generation.

Usage:
    python main.py                        # Train and evaluate all models
    python main.py --model rnn            # Train only Vanilla RNN
    python main.py --model lstm           # Train only LSTM
    python main.py --model attention      # Train only LSTM with Attention
    python main.py --model transformer    # Train only Transformer
    python main.py --eval-only            # Only run evaluation (models must exist)
    python main.py --extended-length      # Use extended sequence lengths
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    SEED, CHECKPOINT_DIR, EVALUATION_DIR, ATTENTION_VIS_DIR,
    NUM_EPOCHS, HIDDEN_DIM, EMBED_DIM, NUM_LAYERS, DROPOUT,
    MAX_SRC_LEN_EXTENDED, MAX_TRG_LEN_EXTENDED
)
from src.data import load_and_prepare_data
from src.models import (
    build_vanilla_rnn, build_lstm, build_attention_lstm,
    build_transformer, AttentionSeq2Seq
)
from src.train_utils import train_model
from src.eval_utils import (
    evaluate_model_on_test, generate_code, analyze_errors,
    bleu_vs_docstring_length
)


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def plot_training_curves(histories, save_dir):
    """Plot and save training/validation loss and accuracy curves."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    for name, hist in histories.items():
        axes[0].plot(hist["train_losses"], label=f"{name} (train)")
        axes[0].plot(hist["val_losses"], label=f"{name} (val)", linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy curves
    for name, hist in histories.items():
        axes[1].plot(hist["train_accs"], label=f"{name} (train)")
        axes[1].plot(hist["val_accs"], label=f"{name} (val)", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Token Accuracy")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


def plot_bleu_comparison(all_results, save_dir):
    """Plot BLEU score comparison bar chart."""
    os.makedirs(save_dir, exist_ok=True)

    names = list(all_results.keys())
    bleu_scores = [all_results[n]["avg_bleu"] for n in names]
    token_accs = [all_results[n]["token_accuracy"] for n in names]
    exact_matches = [all_results[n]["exact_match_rate"] for n in names]
    ast_rates = [all_results[n].get("ast_valid_rate", 0) for n in names]

    x = np.arange(len(names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 1.5 * width, bleu_scores, width, label="BLEU Score")
    ax.bar(x - 0.5 * width, token_accs, width, label="Token Accuracy")
    ax.bar(x + 0.5 * width, exact_matches, width, label="Exact Match Rate")
    ax.bar(x + 1.5 * width, ast_rates, width, label="AST Valid Rate")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: BLEU, Token Accuracy, Exact Match, AST Valid")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "model_comparison.png"), dpi=150)
    plt.close()
    print(f"Model comparison saved to {save_dir}/model_comparison.png")


def plot_bleu_vs_length(length_bleu_dict, save_dir):
    """Plot BLEU score vs docstring length for each model."""
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, length_bleu in length_bleu_dict.items():
        bins = sorted(length_bleu.keys())
        scores = [length_bleu[b] for b in bins]
        ax.plot(bins, scores, marker="o", label=name)

    ax.set_xlabel("Docstring Length (tokens, binned)")
    ax.set_ylabel("Average BLEU Score")
    ax.set_title("BLEU Score vs. Docstring Length")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "bleu_vs_length.png"), dpi=150)
    plt.close()
    print(f"BLEU vs length saved to {save_dir}/bleu_vs_length.png")


def plot_attention_heatmap(attention_weights, src_tokens, trg_tokens,
                           save_path, title="Attention Heatmap"):
    """Plot attention heatmap for a single example."""
    fig, ax = plt.subplots(figsize=(12, 8))

    trg_len = min(len(trg_tokens), attention_weights.shape[0])
    src_len = min(len(src_tokens), attention_weights.shape[1])

    attn = attention_weights[:trg_len, :src_len]

    sns.heatmap(
        attn, xticklabels=src_tokens[:src_len],
        yticklabels=trg_tokens[:trg_len],
        cmap="YlOrRd", ax=ax, vmin=0, vmax=attn.max()
    )
    ax.set_xlabel("Source (Docstring)")
    ax.set_ylabel("Generated (Code)")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_attention_examples(model, test_loader, src_vocab, trg_vocab,
                                  device, save_dir, num_examples=3):
    """Generate and visualize attention for test examples."""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    count = 0

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)

            for i in range(src.shape[0]):
                if count >= num_examples:
                    return

                src_single = src[i].unsqueeze(0)
                src_tokens = src_vocab.decode(src[i].cpu().tolist())
                ref_tokens = trg_vocab.decode(trg[i].cpu().tolist())

                gen_tokens, attn_weights = generate_code(
                    model, src_single, trg_vocab, device,
                    has_attention=True
                )

                if attn_weights is not None and len(gen_tokens) > 0:
                    save_path = os.path.join(
                        save_dir, f"attention_example_{count + 1}.png"
                    )
                    plot_attention_heatmap(
                        attn_weights, src_tokens, gen_tokens,
                        save_path,
                        title=f"Attention Heatmap - Example {count + 1}"
                    )
                    print(f"\nAttention Example {count + 1}:")
                    print(f"  Docstring: {' '.join(src_tokens[:20])}...")
                    print(f"  Reference: {' '.join(ref_tokens[:20])}...")
                    print(f"  Generated: {' '.join(gen_tokens[:20])}...")
                    print(f"  Saved to: {save_path}")
                    count += 1

    print(f"\nGenerated {count} attention visualizations")


def main():
    parser = argparse.ArgumentParser(
        description="Docstring-to-Code Seq2Seq Training"
    )
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "rnn", "lstm", "attention", "transformer"],
        help="Which model to train"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation on existing checkpoints"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--extended-length", action="store_true",
        help="Use extended sequence lengths (src=100, trg=150)"
    )
    args = parser.parse_args()

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\n" + "=" * 60)
    print("Loading and preparing data...")
    print("=" * 60)
    data_kwargs = {}
    if args.extended_length:
        data_kwargs["max_src_len"] = MAX_SRC_LEN_EXTENDED
        data_kwargs["max_trg_len"] = MAX_TRG_LEN_EXTENDED
        print(f"Using extended lengths: src={MAX_SRC_LEN_EXTENDED}, trg={MAX_TRG_LEN_EXTENDED}")
    train_loader, val_loader, test_loader, src_vocab, trg_vocab = \
        load_and_prepare_data(**data_kwargs)

    src_vocab_size = len(src_vocab)
    trg_vocab_size = len(trg_vocab)

    # Save vocabularies
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save({
        "src_vocab": src_vocab,
        "trg_vocab": trg_vocab,
    }, os.path.join(CHECKPOINT_DIR, "vocabularies.pt"))

    histories = {}
    all_results = {}
    models_to_train = []

    if args.model in ("all", "rnn"):
        models_to_train.append(("Vanilla_RNN", False))
    if args.model in ("all", "lstm"):
        models_to_train.append(("LSTM", False))
    if args.model in ("all", "attention"):
        models_to_train.append(("LSTM_Attention", True))
    if args.model in ("all", "transformer"):
        models_to_train.append(("Transformer", False))

    # ==================== Training ====================
    if not args.eval_only:
        for model_name, has_attention in models_to_train:
            if model_name == "Vanilla_RNN":
                model = build_vanilla_rnn(src_vocab_size, trg_vocab_size,
                                          device)
            elif model_name == "LSTM":
                model = build_lstm(src_vocab_size, trg_vocab_size, device)
            elif model_name == "Transformer":
                model = build_transformer(src_vocab_size, trg_vocab_size,
                                          device)
            else:
                model = build_attention_lstm(src_vocab_size, trg_vocab_size,
                                             device)

            history = train_model(
                model, train_loader, val_loader, device,
                model_name=model_name,
                has_attention=has_attention,
                num_epochs=args.epochs,
            )
            histories[model_name] = history

    # ==================== Evaluation ====================
    print("\n" + "=" * 60)
    print("Evaluating models on test set...")
    print("=" * 60)

    length_bleu_dict = {}

    for model_name, has_attention in models_to_train:
        checkpoint_path = os.path.join(
            CHECKPOINT_DIR, f"{model_name}_best.pt"
        )
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found for {model_name}, skipping eval.")
            continue

        # Rebuild model
        if model_name == "Vanilla_RNN":
            model = build_vanilla_rnn(src_vocab_size, trg_vocab_size, device)
        elif model_name == "LSTM":
            model = build_lstm(src_vocab_size, trg_vocab_size, device)
        elif model_name == "Transformer":
            model = build_transformer(src_vocab_size, trg_vocab_size, device)
        else:
            model = build_attention_lstm(src_vocab_size, trg_vocab_size,
                                         device)

        checkpoint = torch.load(checkpoint_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])

        if model_name not in histories and "history" in checkpoint:
            histories[model_name] = checkpoint["history"]

        print(f"\nEvaluating {model_name}...")
        eval_max_len = MAX_TRG_LEN_EXTENDED + 2 if args.extended_length else 82
        results = evaluate_model_on_test(
            model, test_loader, trg_vocab, device,
            has_attention=has_attention, max_len=eval_max_len
        )
        all_results[model_name] = results

        print(f"  BLEU Score:       {results['avg_bleu']:.4f}")
        print(f"  Token Accuracy:   {results['token_accuracy']:.4f}")
        print(f"  Exact Match Rate: {results['exact_match_rate']:.4f}")
        print(f"  AST Valid Rate:   {results['ast_valid_rate']:.4f}")

        # Error analysis
        errors = analyze_errors(results["samples"])
        print(f"  Error analysis (from {len(results['samples'])} samples):")
        for k, v in errors.items():
            print(f"    {k}: {v}")

        # BLEU vs length
        length_bleu = bleu_vs_docstring_length(
            model, test_loader, src_vocab, trg_vocab, device,
            has_attention=has_attention
        )
        length_bleu_dict[model_name] = length_bleu

        # Print sample predictions
        print(f"\n  Sample predictions ({model_name}):")
        for j, s in enumerate(results["samples"][:3]):
            print(f"    Example {j+1}:")
            print(f"      Ref: {s['reference'][:100]}...")
            print(f"      Gen: {s['generated'][:100]}...")
            print(f"      BLEU: {s['bleu']:.4f}")

    # ==================== Plots ====================
    os.makedirs(EVALUATION_DIR, exist_ok=True)

    if histories:
        plot_training_curves(histories, EVALUATION_DIR)

    if all_results:
        plot_bleu_comparison(all_results, EVALUATION_DIR)

    if length_bleu_dict:
        plot_bleu_vs_length(length_bleu_dict, EVALUATION_DIR)

    # Save numeric results
    serializable_results = {}
    for name, res in all_results.items():
        serializable_results[name] = {
            "avg_bleu": float(res["avg_bleu"]),
            "token_accuracy": float(res["token_accuracy"]),
            "exact_match_rate": float(res["exact_match_rate"]),
            "ast_valid_rate": float(res.get("ast_valid_rate", 0)),
            "num_examples": res["num_examples"],
        }
    with open(os.path.join(EVALUATION_DIR, "results.json"), "w") as f:
        json.dump(serializable_results, f, indent=2)

    # ==================== Attention Visualization ====================
    attn_checkpoint = os.path.join(CHECKPOINT_DIR, "LSTM_Attention_best.pt")
    if os.path.exists(attn_checkpoint):
        print("\n" + "=" * 60)
        print("Generating attention visualizations...")
        print("=" * 60)

        attn_model = build_attention_lstm(
            src_vocab_size, trg_vocab_size, device
        )
        cp = torch.load(attn_checkpoint, map_location=device,
                         weights_only=False)
        attn_model.load_state_dict(cp["model_state_dict"])

        visualize_attention_examples(
            attn_model, test_loader, src_vocab, trg_vocab,
            device, ATTENTION_VIS_DIR, num_examples=3
        )

    print("\n" + "=" * 60)
    print("All done!")
    print(f"Checkpoints:     {CHECKPOINT_DIR}")
    print(f"Evaluation:      {EVALUATION_DIR}")
    print(f"Attention Viz:   {ATTENTION_VIS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
