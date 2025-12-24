import random

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader

from .vocab import Vocab, SOS_TOKEN, EOS_TOKEN


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def simple_tokenize(text):
    return text.strip().split()


class CodeSearchNetDataset(Dataset):
    def __init__(self, examples, src_vocab, tgt_vocab, max_src_len, max_tgt_len):
        self.examples = examples
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        src_text, tgt_text = self.examples[idx]
        src_tokens = simple_tokenize(src_text)[: self.max_src_len]
        tgt_tokens = simple_tokenize(tgt_text)[: self.max_tgt_len]

        src_ids = [self.src_vocab.sos_idx] + self.src_vocab.encode(src_tokens) + [
            self.src_vocab.eos_idx
        ]
        tgt_ids = [self.tgt_vocab.sos_idx] + self.tgt_vocab.encode(tgt_tokens) + [
            self.tgt_vocab.eos_idx
        ]

        return torch.tensor(src_ids), torch.tensor(tgt_ids)


def _filter_example(example):
    doc = example.get("docstring", "") or ""
    code = example.get("code", "") or ""
    if not doc.strip() or not code.strip():
        return False
    return True


def _build_examples(dataset, max_src_len, max_tgt_len):
    examples = []
    for ex in dataset:
        doc = ex["docstring"].replace("\n", " ").strip()
        code = ex["code"].replace("\n", " ").strip()
        src_tokens = simple_tokenize(doc)
        tgt_tokens = simple_tokenize(code)
        if not src_tokens or not tgt_tokens:
            continue
        if len(src_tokens) > max_src_len or len(tgt_tokens) > max_tgt_len:
            continue
        examples.append((doc, code))
    return examples


def _get_split(dataset, names):
    for name in names:
        if name in dataset:
            return dataset[name]
    return None


def load_codesearchnet_splits(config):
    print("Loading dataset...", flush=True)
    dataset = load_dataset("Nan-Do/code-search-net-python")
    train_raw = _get_split(dataset, ["train"])
    if train_raw is None:
        raise ValueError("Dataset is missing a train split.")

    val_raw = _get_split(dataset, ["validation", "valid", "val"])
    test_raw = _get_split(dataset, ["test"])

    if val_raw is None:
        split = train_raw.train_test_split(
            test_size=max(1, config.val_size) / max(1, (config.train_size + config.val_size)),
            seed=config.seed,
        )
        train_raw = split["train"]
        val_raw = split["test"]

    if test_raw is None:
        split = train_raw.train_test_split(
            test_size=max(1, config.test_size) / max(1, (config.train_size + config.test_size)),
            seed=config.seed + 1,
        )
        train_raw = split["train"]
        test_raw = split["test"]

    print("Filtering splits...", flush=True)
    train_raw = train_raw.filter(_filter_example)
    val_raw = val_raw.filter(_filter_example)
    test_raw = test_raw.filter(_filter_example)

    print("Building examples (length filtering + truncation)...", flush=True)
    train_examples = _build_examples(
        train_raw, config.max_src_len, config.max_tgt_len
    )[: config.train_size]
    val_examples = _build_examples(val_raw, config.max_src_len, config.max_tgt_len)[
        : config.val_size
    ]
    test_examples = _build_examples(test_raw, config.max_src_len, config.max_tgt_len)[
        : config.test_size
    ]

    print(
        f"Loaded examples: train={len(train_examples)} val={len(val_examples)} test={len(test_examples)}",
        flush=True,
    )

    return train_examples, val_examples, test_examples


def build_vocabs(train_examples, config):
    src_vocab = Vocab(min_freq=config.min_freq)
    tgt_vocab = Vocab(min_freq=config.min_freq)
    src_tokens = [simple_tokenize(doc) for doc, _ in train_examples]
    tgt_tokens = [simple_tokenize(code) for _, code in train_examples]
    src_vocab.build(src_tokens)
    tgt_vocab.build(tgt_tokens)
    return src_vocab, tgt_vocab


def collate_fn(batch, pad_src, pad_tgt):
    src_batch, tgt_batch = zip(*batch)
    src_len = max([len(s) for s in src_batch])
    tgt_len = max([len(t) for t in tgt_batch])

    src_padded = torch.full((len(batch), src_len), pad_src, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_len), pad_tgt, dtype=torch.long)

    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        src_padded[i, : len(src)] = src
        tgt_padded[i, : len(tgt)] = tgt

    return src_padded, tgt_padded


def make_loaders(train_examples, val_examples, test_examples, src_vocab, tgt_vocab, config):
    train_ds = CodeSearchNetDataset(
        train_examples, src_vocab, tgt_vocab, config.max_src_len, config.max_tgt_len
    )
    val_ds = CodeSearchNetDataset(
        val_examples, src_vocab, tgt_vocab, config.max_src_len, config.max_tgt_len
    )
    test_ds = CodeSearchNetDataset(
        test_examples, src_vocab, tgt_vocab, config.max_src_len, config.max_tgt_len
    )

    collate = lambda batch: collate_fn(batch, src_vocab.pad_idx, tgt_vocab.pad_idx)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, collate_fn=collate
    )

    return train_loader, val_loader, test_loader
