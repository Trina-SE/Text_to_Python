"""
Three Seq2Seq model architectures for docstring-to-code generation:
1. Vanilla RNN Seq2Seq
2. LSTM Seq2Seq
3. LSTM with Bahdanau Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .config import (
    EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT,
    TEACHER_FORCING_RATIO, SOS_IDX, EOS_IDX, PAD_IDX
)

# Model 1: Vanilla RNN Seq2Seq

class RNNEncoder(nn.Module):
    """Vanilla RNN encoder."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers,
                          dropout=dropout if n_layers > 1 else 0,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: (batch, src_len)
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden: (n_layers, batch, hidden_dim) - context vector
        return hidden


class RNNDecoder(nn.Module):
    """Vanilla RNN decoder."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.rnn = nn.RNN(embed_dim, hidden_dim, n_layers,
                          dropout=dropout if n_layers > 1 else 0,
                          batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden):
        # input_token: (batch,) -> (batch, 1)
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden


class VanillaRNNSeq2Seq(nn.Module):
    """Vanilla RNN Seq2Seq model with no attention."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        # Encode
        hidden = self.encoder(src)

        # First input is SOS token
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = prediction

            if random.random() < teacher_forcing_ratio:
                input_token = trg[:, t]
            else:
                input_token = prediction.argmax(1)

        return outputs

# Model 2: LSTM Seq2Seq


class LSTMEncoder(nn.Module):
    """LSTM encoder."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class LSTMDecoder(nn.Module):
    """LSTM decoder."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden, cell


class LSTMSeq2Seq(nn.Module):
    """LSTM Seq2Seq model with no attention."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = prediction

            if random.random() < teacher_forcing_ratio:
                input_token = trg[:, t]
            else:
                input_token = prediction.argmax(1)

        return outputs


# Model 3: LSTM with Bahdanau Attention


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""

    def __init__(self, hidden_dim):
        super().__init__()
        # encoder is bidirectional so enc_dim = hidden_dim * 2
        self.W_s = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.V = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        """
        Args:
            decoder_hidden: (batch, hidden_dim)
            encoder_outputs: (batch, src_len, hidden_dim * 2)
        Returns:
            context: (batch, hidden_dim * 2)
            attention_weights: (batch, src_len)
        """
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state src_len times
        hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # energy: (batch, src_len, hidden_dim)
        energy = torch.tanh(
            self.W_s(hidden_expanded) + self.W_h(encoder_outputs)
        )

        # attention: (batch, src_len, 1) -> (batch, src_len)
        attention = self.V(energy).squeeze(2)
        attention_weights = F.softmax(attention, dim=1)

        # context: (batch, 1, src_len) x (batch, src_len, hidden*2)
        #       -> (batch, 1, hidden*2) -> (batch, hidden*2)
        context = torch.bmm(attention_weights.unsqueeze(1),
                            encoder_outputs).squeeze(1)

        return context, attention_weights


class BiLSTMEncoder(nn.Module):
    """Bidirectional LSTM encoder for the attention model."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        # Project bidirectional hidden/cell to decoder dimensions
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        # encoder_outputs: (batch, src_len, hidden_dim * 2)
        encoder_outputs, (hidden, cell) = self.lstm(embedded)

        # hidden: (n_layers * 2, batch, hidden_dim)
        # Concatenate forward and backward for each layer
        # -> (n_layers, batch, hidden_dim * 2) -> (n_layers, batch, hidden_dim)
        hidden = torch.cat(
            (hidden[0::2], hidden[1::2]), dim=2
        )
        hidden = torch.tanh(self.fc_hidden(hidden))

        cell = torch.cat(
            (cell[0::2], cell[1::2]), dim=2
        )
        cell = torch.tanh(self.fc_cell(cell))

        return encoder_outputs, hidden, cell


class AttentionDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention."""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=NUM_LAYERS,
                 dropout=DROPOUT):
        super().__init__()
        self.attention = BahdanauAttention(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        # Input to LSTM: embedding + context vector
        self.lstm = nn.LSTM(embed_dim + hidden_dim * 2, hidden_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0,
                            batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2 + embed_dim,
                                vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        """
        Args:
            input_token: (batch,)
            hidden: (n_layers, batch, hidden_dim)
            cell: (n_layers, batch, hidden_dim)
            encoder_outputs: (batch, src_len, hidden_dim * 2)
        Returns:
            prediction, hidden, cell, attention_weights
        """
        input_token = input_token.unsqueeze(1)  # (batch, 1)
        embedded = self.dropout(self.embedding(input_token))  # (batch, 1, embed)

        # Use top layer hidden state for attention
        context, attention_weights = self.attention(
            hidden[-1], encoder_outputs
        )

        # Concatenate embedding and context
        context_expanded = context.unsqueeze(1)  # (batch, 1, hidden*2)
        lstm_input = torch.cat([embedded, context_expanded], dim=2)

        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        # Combine output, context, and embedding for prediction
        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        prediction = self.fc_out(
            torch.cat([output, context, embedded], dim=1)
        )

        return prediction, hidden, cell, attention_weights


class AttentionSeq2Seq(nn.Module):
    """LSTM Seq2Seq model with Bahdanau attention."""

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=TEACHER_FORCING_RATIO):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc_out.out_features
        src_len = src.shape[1]

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, trg_len, src_len).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            prediction, hidden, cell, attn_weights = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t] = prediction
            attentions[:, t] = attn_weights

            if random.random() < teacher_forcing_ratio:
                input_token = trg[:, t]
            else:
                input_token = prediction.argmax(1)

        return outputs, attentions



# Model 4: Transformer Seq2Seq


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=500, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder with embedding and positional encoding."""

    def __init__(self, vocab_size, d_model=EMBED_DIM, nhead=8,
                 num_layers=NUM_LAYERS, dim_feedforward=512, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, src, src_key_padding_mask=None):
        # src: (batch, src_len)
        embedded = self.pos_encoding(self.embedding(src) * self.scale)
        return self.transformer_encoder(
            embedded, src_key_padding_mask=src_key_padding_mask
        )


class TransformerDecoder(nn.Module):
    """Transformer decoder with embedding, positional encoding, and output projection."""

    def __init__(self, vocab_size, d_model=EMBED_DIM, nhead=8,
                 num_layers=NUM_LAYERS, dim_feedforward=512, dropout=DROPOUT):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.scale = math.sqrt(d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, trg, memory, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # trg: (batch, trg_len)
        embedded = self.pos_encoding(self.embedding(trg) * self.scale)
        output = self.transformer_decoder(
            embedded, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.fc_out(output)


class TransformerSeq2Seq(nn.Module):
    """Transformer Seq2Seq model for docstring-to-code generation."""

    is_transformer = True

    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    @staticmethod
    def _generate_square_subsequent_mask(sz, device):
        """Generate causal mask: True means masked (ignored)."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=None):
        # teacher_forcing_ratio is accepted but unused (for API compat)
        src_key_padding_mask = (src == PAD_IDX)
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        # Shift target: decoder input is trg[:, :-1], predictions target trg[:, 1:]
        trg_input = trg[:, :-1]
        tgt_key_padding_mask = (trg_input == PAD_IDX)
        tgt_mask = self._generate_square_subsequent_mask(
            trg_input.size(1), self.device
        )

        # (batch, trg_len-1, vocab_size)
        decoder_output = self.decoder(
            trg_input, memory, tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Pad with zeros at position 0 so output[:, 1:] gives the real predictions
        # This matches train_utils which does output[:, 1:] vs trg[:, 1:]
        batch_size = decoder_output.size(0)
        vocab_size = decoder_output.size(2)
        pad_col = torch.zeros(batch_size, 1, vocab_size, device=self.device)
        outputs = torch.cat([pad_col, decoder_output], dim=1)

        return outputs


# Factory functions


def build_vanilla_rnn(src_vocab_size, trg_vocab_size, device):
    """Build Vanilla RNN Seq2Seq model."""
    encoder = RNNEncoder(src_vocab_size)
    decoder = RNNDecoder(trg_vocab_size)
    model = VanillaRNNSeq2Seq(encoder, decoder, device).to(device)
    return model


def build_lstm(src_vocab_size, trg_vocab_size, device):
    """Build LSTM Seq2Seq model."""
    encoder = LSTMEncoder(src_vocab_size)
    decoder = LSTMDecoder(trg_vocab_size)
    model = LSTMSeq2Seq(encoder, decoder, device).to(device)
    return model


def build_attention_lstm(src_vocab_size, trg_vocab_size, device):
    """Build Attention LSTM Seq2Seq model."""
    encoder = BiLSTMEncoder(src_vocab_size)
    decoder = AttentionDecoder(trg_vocab_size)
    model = AttentionSeq2Seq(encoder, decoder, device).to(device)
    return model


def build_transformer(src_vocab_size, trg_vocab_size, device):
    """Build Transformer Seq2Seq model."""
    encoder = TransformerEncoder(src_vocab_size)
    decoder = TransformerDecoder(trg_vocab_size)
    model = TransformerSeq2Seq(encoder, decoder, device).to(device)
    return model
