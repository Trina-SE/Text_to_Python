import math

import sacrebleu


def token_accuracy(preds, targets, pad_idx):
    correct = 0
    total = 0
    for pred, tgt in zip(preds, targets):
        for p, t in zip(pred, tgt):
            if t == pad_idx:
                continue
            total += 1
            if p == t:
                correct += 1
    return correct / total if total > 0 else 0.0


def exact_match(preds, targets, pad_idx, eos_idx):
    matches = 0
    for pred, tgt in zip(preds, targets):
        pred_seq = _strip_special(pred, pad_idx, eos_idx)
        tgt_seq = _strip_special(tgt, pad_idx, eos_idx)
        if pred_seq == tgt_seq:
            matches += 1
    return matches / len(preds) if preds else 0.0


def bleu_score(pred_texts, ref_texts):
    bleu = sacrebleu.corpus_bleu(pred_texts, [ref_texts])
    return bleu.score


def length_bucket(length, buckets):
    for b in buckets:
        if length <= b:
            return b
    return buckets[-1]


def _strip_special(seq, pad_idx, eos_idx):
    cleaned = []
    for t in seq:
        if t == pad_idx:
            continue
        if t == eos_idx:
            break
        cleaned.append(t)
    return cleaned

