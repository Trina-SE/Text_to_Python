import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc_out(output.squeeze(1))
        return logits, hidden


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, state):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        output, (hidden, cell) = self.lstm(embedded, state)
        logits = self.fc_out(output.squeeze(1))
        return logits, (hidden, cell)


class EncoderBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return outputs, (hidden, cell)


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.W = nn.Linear(enc_dim, dec_dim, bias=False)
        self.U = nn.Linear(dec_dim, dec_dim, bias=False)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden, mask):
        # enc_outputs: (batch, src_len, enc_dim)
        # dec_hidden: (batch, dec_dim)
        scores = self.v(torch.tanh(self.W(enc_outputs) + self.U(dec_hidden).unsqueeze(1)))
        scores = scores.squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)
        return context, attn


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, enc_dim, dec_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = BahdanauAttention(enc_dim, dec_dim)
        self.lstm = nn.LSTM(
            embed_dim + enc_dim,
            dec_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_out = nn.Linear(dec_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, state, enc_outputs, mask):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        hidden, cell = state
        context, attn = self.attn(enc_outputs, hidden[-1], mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        logits = self.fc_out(output.squeeze(1))
        return logits, (hidden, cell), attn


class Seq2SeqRNN(nn.Module):
    def __init__(
        self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout
    ):
        super().__init__()
        self.encoder = EncoderRNN(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = DecoderRNN(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, tgt, sos_idx, teacher_forcing):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        logits = []
        _, hidden = self.encoder(src)

        input_token = tgt[:, 0] if tgt is not None else torch.full(
            (batch_size,), sos_idx, dtype=torch.long, device=src.device
        )
        for t in range(1, max_len):
            step_logits, hidden = self.decoder(input_token, hidden)
            logits.append(step_logits)
            use_teacher = torch.rand(1).item() < teacher_forcing
            input_token = tgt[:, t] if use_teacher else step_logits.argmax(dim=-1)

        return torch.stack(logits, dim=1)

    def greedy_decode(self, src, max_len, sos_idx):
        batch_size = src.size(0)
        outputs = []
        _, hidden = self.encoder(src)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            step_logits, hidden = self.decoder(input_token, hidden)
            next_token = step_logits.argmax(dim=-1)
            outputs.append(next_token)
            input_token = next_token
        return torch.stack(outputs, dim=1)


class Seq2SeqLSTM(nn.Module):
    def __init__(
        self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout
    ):
        super().__init__()
        self.encoder = EncoderLSTM(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
        self.decoder = DecoderLSTM(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout)

    def forward(self, src, tgt, sos_idx, teacher_forcing):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        logits = []
        _, state = self.encoder(src)

        input_token = tgt[:, 0] if tgt is not None else torch.full(
            (batch_size,), sos_idx, dtype=torch.long, device=src.device
        )
        for t in range(1, max_len):
            step_logits, state = self.decoder(input_token, state)
            logits.append(step_logits)
            use_teacher = torch.rand(1).item() < teacher_forcing
            input_token = tgt[:, t] if use_teacher else step_logits.argmax(dim=-1)

        return torch.stack(logits, dim=1)

    def greedy_decode(self, src, max_len, sos_idx):
        batch_size = src.size(0)
        outputs = []
        _, state = self.encoder(src)
        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            step_logits, state = self.decoder(input_token, state)
            next_token = step_logits.argmax(dim=-1)
            outputs.append(next_token)
            input_token = next_token
        return torch.stack(outputs, dim=1)


class Seq2SeqAttention(nn.Module):
    def __init__(
        self, src_vocab_size, tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout
    ):
        super().__init__()
        self.encoder = EncoderBiLSTM(
            src_vocab_size, embed_dim, hidden_dim, num_layers, dropout
        )
        enc_dim = hidden_dim * 2
        self.reduce_h = nn.Linear(enc_dim, hidden_dim)
        self.reduce_c = nn.Linear(enc_dim, hidden_dim)
        self.decoder = AttentionDecoder(
            tgt_vocab_size, embed_dim, enc_dim, hidden_dim, num_layers, dropout
        )

    def forward(self, src, tgt, sos_idx, teacher_forcing, pad_idx):
        batch_size = src.size(0)
        max_len = tgt.size(1)
        logits = []
        attn_weights = []
        enc_outputs, (hidden, cell) = self.encoder(src)
        hidden = self._bridge_hidden(hidden, self.reduce_h)
        cell = self._bridge_hidden(cell, self.reduce_c)
        state = (hidden, cell)
        mask = (src != pad_idx).int()

        input_token = tgt[:, 0] if tgt is not None else torch.full(
            (batch_size,), sos_idx, dtype=torch.long, device=src.device
        )
        for t in range(1, max_len):
            step_logits, state, attn = self.decoder(input_token, state, enc_outputs, mask)
            logits.append(step_logits)
            attn_weights.append(attn)
            use_teacher = torch.rand(1).item() < teacher_forcing
            input_token = tgt[:, t] if use_teacher else step_logits.argmax(dim=-1)

        return torch.stack(logits, dim=1), torch.stack(attn_weights, dim=1)

    def greedy_decode(self, src, max_len, sos_idx, pad_idx):
        batch_size = src.size(0)
        outputs = []
        attn_weights = []
        enc_outputs, (hidden, cell) = self.encoder(src)
        hidden = self._bridge_hidden(hidden, self.reduce_h)
        cell = self._bridge_hidden(cell, self.reduce_c)
        state = (hidden, cell)
        mask = (src != pad_idx).int()

        input_token = torch.full((batch_size,), sos_idx, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            step_logits, state, attn = self.decoder(input_token, state, enc_outputs, mask)
            next_token = step_logits.argmax(dim=-1)
            outputs.append(next_token)
            attn_weights.append(attn)
            input_token = next_token
        return torch.stack(outputs, dim=1), torch.stack(attn_weights, dim=1)

    def _bridge_hidden(self, hidden, projector):
        # hidden: (num_layers*2, batch, hidden_dim)
        hidden = torch.cat(
            [hidden[0::2], hidden[1::2]], dim=-1
        )
        return torch.tanh(projector(hidden))
