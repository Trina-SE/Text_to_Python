"""
Data loading, tokenization, vocabulary building, and dataset creation
for the CodeSearchNet Python dataset.
"""

import re
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter

from .config import (
    DATASET_NAME, NUM_TRAIN_EXAMPLES, NUM_VAL_EXAMPLES, NUM_TEST_EXAMPLES,
    MAX_SRC_LEN, MAX_TRG_LEN, FREQ_THRESHOLD, BATCH_SIZE,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
)

# Special tokens
PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"


def tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    text = text.strip()
    # Replace newlines and tabs with special tokens
    text = text.replace("\n", " NEWLINE ")
    text = text.replace("\t", " INDENT ")
    # Split on whitespace and keep punctuation separate
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+|[^\s]", text)
    return tokens


class Vocabulary:
    """Builds and manages a vocabulary mapping between tokens and indices."""

    def __init__(self, freq_threshold=FREQ_THRESHOLD):
        self.freq_threshold = freq_threshold
        self.itos = {PAD_IDX: PAD_TOKEN, SOS_IDX: SOS_TOKEN,
                     EOS_IDX: EOS_TOKEN, UNK_IDX: UNK_TOKEN}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocabulary(self, token_lists):
        """Build vocabulary from a list of token lists."""
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        idx = len(self.itos)
        for word, count in counter.most_common():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, tokens):
        """Convert tokens to indices."""
        return [self.stoi.get(tok, UNK_IDX) for tok in tokens]

    def decode(self, indices):
        """Convert indices back to tokens."""
        tokens = []
        for idx in indices:
            if idx == EOS_IDX:
                break
            if idx not in (PAD_IDX, SOS_IDX):
                tokens.append(self.itos.get(idx, UNK_TOKEN))
        return tokens

    def __len__(self):
        return len(self.itos)


class CodeDocstringDataset(Dataset):
    """PyTorch Dataset for docstring-code pairs."""

    def __init__(self, docstrings, codes, src_vocab, trg_vocab,
                 max_src_len=MAX_SRC_LEN, max_trg_len=MAX_TRG_LEN):
        self.docstrings = docstrings
        self.codes = codes
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_src_len = max_src_len
        self.max_trg_len = max_trg_len

    def __len__(self):
        return len(self.docstrings)

    def __getitem__(self, idx):
        src_tokens = self.docstrings[idx][:self.max_src_len]
        trg_tokens = self.codes[idx][:self.max_trg_len]

        src_indices = ([SOS_IDX]
                       + self.src_vocab.numericalize(src_tokens)
                       + [EOS_IDX])
        trg_indices = ([SOS_IDX]
                       + self.trg_vocab.numericalize(trg_tokens)
                       + [EOS_IDX])

        return torch.tensor(src_indices, dtype=torch.long), \
               torch.tensor(trg_indices, dtype=torch.long)


def collate_fn(batch):
    """Pad sequences to same length within a batch."""
    src_batch, trg_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True,
                              padding_value=PAD_IDX)
    trg_padded = pad_sequence(trg_batch, batch_first=True,
                              padding_value=PAD_IDX)
    return src_padded, trg_padded


def load_and_prepare_data(num_train=NUM_TRAIN_EXAMPLES,
                          num_val=NUM_VAL_EXAMPLES,
                          num_test=NUM_TEST_EXAMPLES,
                          max_src_len=MAX_SRC_LEN,
                          max_trg_len=MAX_TRG_LEN,
                          batch_size=BATCH_SIZE,
                          freq_threshold=FREQ_THRESHOLD):
    """
    Load CodeSearchNet Python dataset, build vocabularies,
    and return DataLoaders and vocabularies.

    Returns:
        train_loader, val_loader, test_loader, src_vocab, trg_vocab
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset(DATASET_NAME, split="train", trust_remote_code=True)

    # Identify columns
    columns = dataset.column_names
    print(f"Dataset columns: {columns}")

    total_needed = num_train + num_val + num_test

    # Extract docstring and code pairs
    docstrings_raw = []
    codes_raw = []

    for i, example in enumerate(dataset):
        if i >= total_needed:
            break

        # Try different column name conventions
        docstring = None
        code = None

        if "func_documentation_string" in columns:
            docstring = example["func_documentation_string"]
        elif "docstring" in columns:
            docstring = example["docstring"]

        if "func_code_string" in columns:
            code = example["func_code_string"]
        elif "code" in columns:
            code = example["code"]
        elif "whole_func_string" in columns:
            code = example["whole_func_string"]

        if docstring and code and len(docstring.strip()) > 0 and len(code.strip()) > 0:
            docstrings_raw.append(docstring)
            codes_raw.append(code)

    print(f"Loaded {len(docstrings_raw)} valid docstring-code pairs")

    # Tokenize
    print("Tokenizing...")
    docstrings_tokenized = [tokenize(d) for d in docstrings_raw]
    codes_tokenized = [tokenize(c) for c in codes_raw]

    # Filter out pairs where either side is empty after tokenization
    filtered_pairs = [
        (d, c) for d, c in zip(docstrings_tokenized, codes_tokenized)
        if len(d) > 0 and len(c) > 0
    ]
    docstrings_tokenized = [p[0] for p in filtered_pairs]
    codes_tokenized = [p[1] for p in filtered_pairs]

    # Split into train/val/test
    train_docs = docstrings_tokenized[:num_train]
    train_codes = codes_tokenized[:num_train]
    val_docs = docstrings_tokenized[num_train:num_train + num_val]
    val_codes = codes_tokenized[num_train:num_train + num_val]
    test_docs = docstrings_tokenized[num_train + num_val:
                                     num_train + num_val + num_test]
    test_codes = codes_tokenized[num_train + num_val:
                                 num_train + num_val + num_test]

    print(f"Train: {len(train_docs)}, Val: {len(val_docs)}, "
          f"Test: {len(test_docs)}")

    # Build vocabularies (only on training data)
    print("Building vocabularies...")
    src_vocab = Vocabulary(freq_threshold=freq_threshold)
    src_vocab.build_vocabulary(train_docs)
    trg_vocab = Vocabulary(freq_threshold=freq_threshold)
    trg_vocab.build_vocabulary(train_codes)

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(trg_vocab)}")

    # Create datasets
    train_dataset = CodeDocstringDataset(
        train_docs, train_codes, src_vocab, trg_vocab,
        max_src_len, max_trg_len
    )
    val_dataset = CodeDocstringDataset(
        val_docs, val_codes, src_vocab, trg_vocab,
        max_src_len, max_trg_len
    )
    test_dataset = CodeDocstringDataset(
        test_docs, test_codes, src_vocab, trg_vocab,
        max_src_len, max_trg_len
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )

    return train_loader, val_loader, test_loader, src_vocab, trg_vocab
