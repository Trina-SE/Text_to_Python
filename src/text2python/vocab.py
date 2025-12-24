from collections import Counter


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


class Vocab:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.token_to_idx = {}
        self.idx_to_token = []

    def build(self, sequences):
        counter = Counter()
        for seq in sequences:
            counter.update(seq)

        special = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]
        self.idx_to_token = list(special)
        self.token_to_idx = {tok: i for i, tok in enumerate(self.idx_to_token)}

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.token_to_idx:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)

    @classmethod
    def from_token_to_idx(cls, token_to_idx):
        vocab = cls(min_freq=1)
        vocab.token_to_idx = dict(token_to_idx)
        vocab.idx_to_token = [""] * len(token_to_idx)
        for token, idx in token_to_idx.items():
            vocab.idx_to_token[idx] = token
        return vocab

    def __len__(self):
        return len(self.idx_to_token)

    def encode(self, tokens):
        return [self.token_to_idx.get(t, self.token_to_idx[UNK_TOKEN]) for t in tokens]

    def decode(self, indices, stop_at_eos=True):
        tokens = []
        for idx in indices:
            tok = self.idx_to_token[idx]
            if stop_at_eos and tok == EOS_TOKEN:
                break
            if tok not in (SOS_TOKEN, PAD_TOKEN):
                tokens.append(tok)
        return tokens

    @property
    def pad_idx(self):
        return self.token_to_idx[PAD_TOKEN]

    @property
    def sos_idx(self):
        return self.token_to_idx[SOS_TOKEN]

    @property
    def eos_idx(self):
        return self.token_to_idx[EOS_TOKEN]
