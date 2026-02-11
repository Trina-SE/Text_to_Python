"""
Training and validation utilities for all Seq2Seq models.
"""

import os
import time
import torch
import torch.nn as nn

from .config import (
    LEARNING_RATE, NUM_EPOCHS, CLIP_GRAD, TEACHER_FORCING_RATIO,
    PAD_IDX, CHECKPOINT_DIR
)


def train_epoch(model, dataloader, optimizer, criterion, clip, device,
                teacher_forcing_ratio=TEACHER_FORCING_RATIO,
                has_attention=False):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    total_tokens = 0
    correct_tokens = 0

    for src, trg in dataloader:
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()

        if has_attention:
            output, _ = model(src, trg, teacher_forcing_ratio)
        else:
            output = model(src, trg, teacher_forcing_ratio)

        # output: (batch, trg_len, vocab_size)
        # Flatten for loss: skip first token (SOS)
        output = output[:, 1:].contiguous().view(-1, output.shape[-1])
        trg_flat = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output, trg_flat)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        # Token accuracy
        preds = output.argmax(1)
        mask = trg_flat != PAD_IDX
        correct_tokens += (preds == trg_flat)[mask].sum().item()
        total_tokens += mask.sum().item()

    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy


def evaluate_epoch(model, dataloader, criterion, device,
                   has_attention=False):
    """Evaluate on validation/test set."""
    model.eval()
    epoch_loss = 0
    total_tokens = 0
    correct_tokens = 0

    with torch.no_grad():
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            if has_attention:
                output, _ = model(src, trg, 0)  # no teacher forcing
            else:
                output = model(src, trg, 0)

            output = output[:, 1:].contiguous().view(-1, output.shape[-1])
            trg_flat = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output, trg_flat)
            epoch_loss += loss.item()

            preds = output.argmax(1)
            mask = trg_flat != PAD_IDX
            correct_tokens += (preds == trg_flat)[mask].sum().item()
            total_tokens += mask.sum().item()

    avg_loss = epoch_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, device,
                model_name="model", has_attention=False,
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                clip=CLIP_GRAD,
                teacher_forcing_ratio=TEACHER_FORCING_RATIO):
    """
    Full training loop with checkpointing and loss tracking.

    Returns:
        history dict with train_losses, val_losses, train_accs, val_accs
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history = {
        "train_losses": [],
        "val_losses": [],
        "train_accs": [],
        "val_accs": [],
    }

    best_val_loss = float("inf")
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
    print()

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, clip, device,
            teacher_forcing_ratio, has_attention
        )
        val_loss, val_acc = evaluate_epoch(
            model, val_loader, criterion, device, has_attention
        )

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["train_accs"].append(train_acc)
        history["val_accs"].append(val_acc)

        elapsed = time.time() - start_time

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f"{model_name}_best.pt"
            )
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "history": history,
            }, checkpoint_path)

        print(f"Epoch {epoch:02d}/{num_epochs} | "
              f"Time: {elapsed:.1f}s | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f}"
              f"{' *' if val_loss <= best_val_loss else ''}")

    # Save final checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_final.pt")
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "history": history,
    }, final_path)

    print(f"\nBest validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")

    return history
