import argparse
import csv
import os
from dataclasses import asdict

import torch
import torch.nn as nn

from .config import Config
from .data import (
    load_codesearchnet_splits,
    build_vocabs,
    make_loaders,
    set_seed,
)
from .models import Seq2SeqRNN, Seq2SeqLSTM, Seq2SeqAttention


MODEL_MAP = {
    "rnn": Seq2SeqRNN,
    "lstm": Seq2SeqLSTM,
    "attention": Seq2SeqAttention,
}


def get_device(preferred):
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, model, config, src_vocab, tgt_vocab):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_type": config.model_type,
            "config": asdict(config),
            "model_state": model.state_dict(),
            "src_vocab": src_vocab.token_to_idx,
            "tgt_vocab": tgt_vocab.token_to_idx,
        },
        path,
    )


def train_epoch(model, loader, optimizer, criterion, device, config, pad_idx):
    model.train()
    total_loss = 0.0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        optimizer.zero_grad()
        if config.model_type == "attention":
            logits, _ = model(
                src, tgt, config.tgt_sos_idx, config.teacher_forcing, pad_idx
            )
        else:
            logits = model(src, tgt, config.tgt_sos_idx, config.teacher_forcing)
        output_dim = logits.size(-1)
        loss = criterion(logits.reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def eval_epoch(model, loader, criterion, device, config, pad_idx):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            if config.model_type == "attention":
                logits, _ = model(
                    src, tgt, config.tgt_sos_idx, 0.0, pad_idx
                )
            else:
                logits = model(src, tgt, config.tgt_sos_idx, 0.0)
            output_dim = logits.size(-1)
            loss = criterion(logits.reshape(-1, output_dim), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_MAP.keys(), default="rnn")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--train-size", type=int, default=Config.train_size)
    parser.add_argument("--val-size", type=int, default=Config.val_size)
    parser.add_argument("--test-size", type=int, default=Config.test_size)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--embed-dim", type=int, default=Config.embed_dim)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--max-src-len", type=int, default=Config.max_src_len)
    parser.add_argument("--max-tgt-len", type=int, default=Config.max_tgt_len)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--teacher-forcing", type=float, default=Config.teacher_forcing)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--checkpoint-dir", type=str, default=Config.checkpoint_dir)
    args = parser.parse_args()

    config = Config(
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        lr=args.lr,
        epochs=args.epochs,
        teacher_forcing=args.teacher_forcing,
        device=args.device,
        output_dir=args.output_dir,
        checkpoint_dir=args.checkpoint_dir,
    )
    config.model_type = args.model

    set_seed(config.seed)
    device = get_device(config.device)

    train_examples, val_examples, test_examples = load_codesearchnet_splits(config)
    src_vocab, tgt_vocab = build_vocabs(train_examples, config)
    config.tgt_sos_idx = tgt_vocab.sos_idx
    config.tgt_pad_idx = tgt_vocab.pad_idx

    train_loader, val_loader, _ = make_loaders(
        train_examples, val_examples, test_examples, src_vocab, tgt_vocab, config
    )

    model_cls = MODEL_MAP[config.model_type]
    model = model_cls(
        len(src_vocab),
        len(tgt_vocab),
        config.embed_dim,
        config.hidden_dim,
        config.num_layers,
        config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    history_path = os.path.join(config.output_dir, f"{config.model_type}_loss.csv")
    os.makedirs(config.output_dir, exist_ok=True)

    best_val = float("inf")
    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for epoch in range(1, config.epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion, device, config, tgt_vocab.pad_idx
            )
            val_loss = eval_epoch(
                model, val_loader, criterion, device, config, tgt_vocab.pad_idx
            )
            writer.writerow([epoch, train_loss, val_loss])
            print(
                f"epoch {epoch}/{config.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
                flush=True,
            )
            if val_loss < best_val:
                best_val = val_loss
                ckpt_path = os.path.join(
                    config.checkpoint_dir, f"{config.model_type}_best.pt"
                )
                save_checkpoint(ckpt_path, model, config, src_vocab, tgt_vocab)


if __name__ == "__main__":
    main()
