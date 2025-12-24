from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    model_type: str = "rnn"
    train_size: int = 8000
    val_size: int = 1000
    test_size: int = 1000
    max_src_len: int = 50
    max_tgt_len: int = 80
    min_freq: int = 2
    batch_size: int = 64
    embed_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.2
    lr: float = 1e-3
    epochs: int = 10
    teacher_forcing: float = 0.5
    device: str = "cuda"
    output_dir: str = "outputs"
    checkpoint_dir: str = "checkpoints"
    figure_dir: str = "figures"
