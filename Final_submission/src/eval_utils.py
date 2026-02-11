"""
Evaluation utilities: BLEU score, token accuracy, exact match,
code generation, AST validation, and attention visualization.
"""

import ast
import torch
import numpy as np
from collections import Counter

from .config import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX


def validate_syntax_ast(generated_tokens):
    """
    Check if generated tokens form valid Python syntax using ast.parse().

    Args:
        generated_tokens: list of string tokens

    Returns:
        True if the code parses as valid Python, False otherwise.
    """
    code = " ".join(generated_tokens)
    code = code.replace("NEWLINE", "\n").replace("INDENT", "    ")
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def generate_code(model, src_tensor, trg_vocab, device, max_len=82,
                  has_attention=False):
    """
    Generate code from a source (docstring) tensor using greedy decoding.

    Args:
        model: trained seq2seq model
        src_tensor: (1, src_len) tensor
        trg_vocab: target vocabulary
        device: torch device
        max_len: maximum output length
        has_attention: whether model returns attention weights

    Returns:
        generated_tokens: list of tokens
        attention_weights: (max_len, src_len) or None
    """
    model.eval()
    with torch.no_grad():
        src_tensor = src_tensor.to(device)

        # Transformer autoregressive generation
        if getattr(model, 'is_transformer', False):
            src_key_padding_mask = (src_tensor == PAD_IDX)
            memory = model.encoder(
                src_tensor, src_key_padding_mask=src_key_padding_mask
            )
            generated_indices = [SOS_IDX]

            for _ in range(max_len):
                trg_tensor = torch.tensor(
                    [generated_indices], device=device
                )
                tgt_mask = model._generate_square_subsequent_mask(
                    trg_tensor.size(1), device
                )
                output = model.decoder(
                    trg_tensor, memory, tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask
                )
                # Take prediction at last position
                next_token = output[0, -1].argmax().item()
                if next_token == EOS_IDX:
                    break
                generated_indices.append(next_token)

            # Remove the leading SOS
            generated_tokens = trg_vocab.decode(generated_indices[1:])
            return generated_tokens, None

        if has_attention:
            encoder_outputs, hidden, cell = model.encoder(src_tensor)
        elif hasattr(model.encoder, 'lstm'):
            hidden, cell = model.encoder(src_tensor)
        else:
            hidden = model.encoder(src_tensor)

        input_token = torch.tensor([SOS_IDX], device=device)
        generated_indices = []
        attention_weights = []

        for _ in range(max_len):
            if has_attention:
                prediction, hidden, cell, attn_w = model.decoder(
                    input_token, hidden, cell, encoder_outputs
                )
                attention_weights.append(attn_w.squeeze(0).cpu().numpy())
            elif hasattr(model.decoder, 'lstm'):
                prediction, hidden, cell = model.decoder(
                    input_token, hidden, cell
                )
            else:
                prediction, hidden = model.decoder(input_token, hidden)

            top_token = prediction.argmax(1).item()
            if top_token == EOS_IDX:
                break
            generated_indices.append(top_token)
            input_token = torch.tensor([top_token], device=device)

        generated_tokens = trg_vocab.decode(generated_indices)

        if has_attention and attention_weights:
            attention_weights = np.array(attention_weights)
        else:
            attention_weights = None

        return generated_tokens, attention_weights


def compute_bleu(reference_tokens, hypothesis_tokens, max_n=4):
    """
    Compute BLEU score between reference and hypothesis token lists.
    Uses smoothed BLEU with brevity penalty.
    """
    if len(hypothesis_tokens) == 0:
        return 0.0

    # Compute n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter()
        hyp_ngrams = Counter()

        for i in range(len(reference_tokens) - n + 1):
            ngram = tuple(reference_tokens[i:i + n])
            ref_ngrams[ngram] += 1

        for i in range(len(hypothesis_tokens) - n + 1):
            ngram = tuple(hypothesis_tokens[i:i + n])
            hyp_ngrams[ngram] += 1

        # Clipped counts
        clipped = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            precisions.append(0.0)
        else:
            # Add-1 smoothing for n > 1
            if n > 1 and clipped == 0:
                precisions.append(1.0 / (total + 1))
            else:
                precisions.append(clipped / total)

    # Geometric mean of precisions
    if min(precisions) == 0:
        return 0.0

    log_avg = sum(np.log(p) for p in precisions) / max_n

    # Brevity penalty
    bp = 1.0
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    if hyp_len < ref_len:
        bp = np.exp(1 - ref_len / hyp_len)

    bleu = bp * np.exp(log_avg)
    return bleu


def compute_exact_match(reference_tokens, hypothesis_tokens):
    """Check if generated code exactly matches reference."""
    return reference_tokens == hypothesis_tokens


def evaluate_model_on_test(model, test_loader, trg_vocab, device,
                           has_attention=False, num_examples=None,
                           max_len=82):
    """
    Evaluate a model on the test set.

    Returns:
        results dict with bleu_scores, token_accuracies,
        exact_matches, and sample predictions
    """
    model.eval()
    bleu_scores = []
    exact_matches = []
    ast_valid_count = 0
    total_tokens = 0
    correct_tokens = 0
    samples = []

    with torch.no_grad():
        example_count = 0
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)

            for i in range(src.shape[0]):
                if num_examples and example_count >= num_examples:
                    break

                src_single = src[i].unsqueeze(0)
                trg_single = trg[i]

                # Generate
                gen_tokens, attn = generate_code(
                    model, src_single, trg_vocab, device,
                    max_len=max_len, has_attention=has_attention
                )

                # Reference tokens (excluding PAD, SOS, EOS)
                ref_tokens = trg_vocab.decode(trg_single.cpu().tolist())

                # BLEU
                bleu = compute_bleu(ref_tokens, gen_tokens)
                bleu_scores.append(bleu)

                # Exact match
                em = compute_exact_match(ref_tokens, gen_tokens)
                exact_matches.append(em)

                # AST syntax validation
                if validate_syntax_ast(gen_tokens):
                    ast_valid_count += 1

                # Token accuracy for this example
                min_len = min(len(ref_tokens), len(gen_tokens))
                if min_len > 0:
                    matches = sum(
                        1 for r, g in zip(ref_tokens[:min_len],
                                          gen_tokens[:min_len])
                        if r == g
                    )
                    total_tokens += len(ref_tokens)
                    correct_tokens += matches

                # Save some samples
                if len(samples) < 10:
                    samples.append({
                        "reference": " ".join(ref_tokens),
                        "generated": " ".join(gen_tokens),
                        "bleu": bleu,
                        "exact_match": em,
                        "attention": attn,
                    })

                example_count += 1

            if num_examples and example_count >= num_examples:
                break

    avg_bleu = np.mean(bleu_scores) if bleu_scores else 0
    exact_match_rate = np.mean(exact_matches) if exact_matches else 0
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    ast_valid_rate = ast_valid_count / len(bleu_scores) if bleu_scores else 0

    results = {
        "avg_bleu": avg_bleu,
        "bleu_scores": bleu_scores,
        "exact_match_rate": exact_match_rate,
        "token_accuracy": token_accuracy,
        "ast_valid_rate": ast_valid_rate,
        "num_examples": len(bleu_scores),
        "samples": samples,
    }

    return results


def analyze_errors(samples):
    """
    Analyze common error types in generated code.

    Returns dict with error category counts.
    """
    errors = {
        "syntax_errors": 0,
        "missing_indentation": 0,
        "incorrect_operators": 0,
        "missing_tokens": 0,
        "extra_tokens": 0,
        "total_errors": 0,
    }

    for sample in samples:
        ref = sample["reference"]
        gen = sample["generated"]

        if ref == gen:
            continue

        errors["total_errors"] += 1

        # Check for missing indentation
        if "INDENT" in ref and "INDENT" not in gen:
            errors["missing_indentation"] += 1

        # Check for incorrect operators
        operators = ["+", "-", "*", "/", "==", "!=", ">=", "<=", ">", "<",
                     "and", "or", "not", "in", "is"]
        ref_ops = set(t for t in ref.split() if t in operators)
        gen_ops = set(t for t in gen.split() if t in operators)
        if ref_ops != gen_ops:
            errors["incorrect_operators"] += 1

        # Check for length mismatch (missing/extra tokens)
        ref_len = len(ref.split())
        gen_len = len(gen.split())
        if gen_len < ref_len:
            errors["missing_tokens"] += 1
        elif gen_len > ref_len:
            errors["extra_tokens"] += 1

        # AST-based syntax check
        if not validate_syntax_ast(gen.split()):
            errors["syntax_errors"] += 1

    return errors


def bleu_vs_docstring_length(model, test_loader, src_vocab, trg_vocab,
                              device, has_attention=False):
    """
    Compute BLEU score binned by docstring length.

    Returns dict mapping length_bin -> avg_bleu.
    """
    model.eval()
    length_bleu = {}

    with torch.no_grad():
        for src, trg in test_loader:
            src, trg = src.to(device), trg.to(device)

            for i in range(src.shape[0]):
                src_single = src[i].unsqueeze(0)
                trg_single = trg[i]

                # Count non-pad tokens in source
                src_len = (src[i] != PAD_IDX).sum().item()

                gen_tokens, _ = generate_code(
                    model, src_single, trg_vocab, device,
                    has_attention=has_attention
                )
                ref_tokens = trg_vocab.decode(trg_single.cpu().tolist())

                bleu = compute_bleu(ref_tokens, gen_tokens)

                # Bin by length (groups of 10)
                bin_key = (src_len // 10) * 10
                if bin_key not in length_bleu:
                    length_bleu[bin_key] = []
                length_bleu[bin_key].append(bleu)

    # Average
    avg_length_bleu = {
        k: np.mean(v) for k, v in sorted(length_bleu.items())
    }
    return avg_length_bleu
