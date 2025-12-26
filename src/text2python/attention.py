import argparse
import os

import matplotlib.pyplot as plt
import torch

from .config import Config
from .data import load_codesearchnet_splits, simple_tokenize, set_seed
from .models import Seq2SeqAttention
from .vocab import Vocab


def get_device(preferred):
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device)
    config = Config(**payload["config"])
    config.model_type = payload["model_type"]
    src_vocab = Vocab.from_token_to_idx(payload["src_vocab"])
    tgt_vocab = Vocab.from_token_to_idx(payload["tgt_vocab"])
    model = Seq2SeqAttention(
        len(src_vocab),
        len(tgt_vocab),
        config.embed_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return config, model, src_vocab, tgt_vocab


def plot_attention(attn, src_tokens, tgt_tokens, out_path):
    plt.figure(figsize=(max(6, len(src_tokens) * 0.6), max(4, len(tgt_tokens) * 0.4)))
    plt.imshow(attn, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(src_tokens)), src_tokens, rotation=45, ha="right")
    plt.yticks(range(len(tgt_tokens)), tgt_tokens)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--figure-dir", type=str, default=Config.figure_dir)
    args = parser.parse_args()

    device = get_device(args.device)
    config, model, src_vocab, tgt_vocab = load_checkpoint(args.checkpoint, device)
    if config.model_type != "attention":
        raise ValueError("Attention visualization requires an attention model checkpoint.")
    set_seed(config.seed)

    train_examples, val_examples, test_examples = load_codesearchnet_splits(config)

    os.makedirs(args.figure_dir, exist_ok=True)

    for idx in args.indices:
        doc, _ = test_examples[idx]
        src_tokens = simple_tokenize(doc)[: config.max_src_len]
        src_ids = [src_vocab.sos_idx] + src_vocab.encode(src_tokens) + [src_vocab.eos_idx]
        src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        pred_tokens, attn_weights = model.greedy_decode(
            src_tensor, config.max_tgt_len + 1, tgt_vocab.sos_idx, tgt_vocab.pad_idx
        )
        pred_tokens = pred_tokens[0].cpu().tolist()
        pred_text_tokens = tgt_vocab.decode(pred_tokens, stop_at_eos=True)
        attn = attn_weights[0].detach().cpu().numpy()
        attn = attn[: len(pred_text_tokens), : len(src_tokens) + 2]
        src_labels = ["<sos>"] + src_tokens + ["<eos>"]
        out_path = os.path.join(args.figure_dir, f"attention_{idx}.png")
        plot_attention(attn, src_labels, pred_text_tokens, out_path)


if __name__ == "__main__":
    main()
