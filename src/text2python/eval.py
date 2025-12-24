import argparse
import csv
import os

import torch

from .config import Config
from .data import load_codesearchnet_splits, make_loaders, simple_tokenize, set_seed
from .metrics import token_accuracy, exact_match, bleu_score, length_bucket
from .models import Seq2SeqRNN, Seq2SeqLSTM, Seq2SeqAttention
from .vocab import Vocab


MODEL_MAP = {
    "rnn": Seq2SeqRNN,
    "lstm": Seq2SeqLSTM,
    "attention": Seq2SeqAttention,
}


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
    model_cls = MODEL_MAP[config.model_type]
    model = model_cls(
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


def decode_batch(tgt_vocab, batch_tokens):
    decoded = []
    for seq in batch_tokens:
        decoded.append(" ".join(tgt_vocab.decode(seq, stop_at_eos=True)))
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    args = parser.parse_args()

    device = get_device(args.device)
    config, model, src_vocab, tgt_vocab = load_checkpoint(args.checkpoint, device)
    config.output_dir = args.output_dir
    set_seed(config.seed)

    train_examples, val_examples, test_examples = load_codesearchnet_splits(config)

    _, _, test_loader = make_loaders(
        train_examples, val_examples, test_examples, src_vocab, tgt_vocab, config
    )

    all_preds = []
    all_tgts = []
    pred_texts = []
    ref_texts = []
    buckets = [10, 20, 30, 40, 50]
    bucket_scores = {b: {"correct": 0, "total": 0} for b in buckets}
    test_src_lens = [len(simple_tokenize(doc)) for doc, _ in test_examples]
    pos = 0

    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            if config.model_type == "attention":
                pred_tokens, _ = model.greedy_decode(
                    src, config.max_tgt_len + 1, tgt_vocab.sos_idx, tgt_vocab.pad_idx
                )
            else:
                pred_tokens = model.greedy_decode(
                    src, config.max_tgt_len + 1, tgt_vocab.sos_idx
                )
            all_preds.extend(pred_tokens.cpu().tolist())
            all_tgts.extend(tgt[:, 1:].cpu().tolist())
            pred_texts.extend(decode_batch(tgt_vocab, pred_tokens.cpu().tolist()))
            ref_texts.extend(decode_batch(tgt_vocab, tgt[:, 1:].cpu().tolist()))

            for i in range(len(src)):
                src_len = test_src_lens[pos + i]
                b = length_bucket(src_len, buckets)
                pred_seq = tgt_vocab.decode(pred_tokens[i].cpu().tolist(), stop_at_eos=True)
                tgt_seq = tgt_vocab.decode(tgt[i, 1:].cpu().tolist(), stop_at_eos=True)
                if pred_seq == tgt_seq:
                    bucket_scores[b]["correct"] += 1
                bucket_scores[b]["total"] += 1
            pos += len(src)

    tok_acc = token_accuracy(all_preds, all_tgts, tgt_vocab.pad_idx)
    exact = exact_match(all_preds, all_tgts, tgt_vocab.pad_idx, tgt_vocab.eos_idx)
    bleu = bleu_score(pred_texts, ref_texts)

    os.makedirs(config.output_dir, exist_ok=True)
    metrics_path = os.path.join(
        config.output_dir, f"{config.model_type}_metrics.csv"
    )
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["token_accuracy", tok_acc])
        writer.writerow(["exact_match", exact])
        writer.writerow(["bleu", bleu])
        for b in buckets:
            total = bucket_scores[b]["total"]
            acc = bucket_scores[b]["correct"] / total if total else 0.0
            writer.writerow([f"exact_match_len_le_{b}", acc])


if __name__ == "__main__":
    main()
