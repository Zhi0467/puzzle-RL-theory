from dataclasses import dataclass, field


@dataclass
class MazeConfig:
    n_nodes: int = 20        # number of nodes per maze
    edge_prob: float = 0.15  # Erdos-Renyi edge probability
    walk_length: int = 40    # length of pre-training random walk


@dataclass
class GPTConfig:
    vocab_size: int = 0      # set after tokenizer is created
    block_size: int = 512    # max context length (sequence length)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 192
    dropout: float = 0.0
    bias: bool = False


@dataclass
class TrainConfig:
    # Dataset sizes
    n_train: int = 20000
    n_val: int = 2000

    # Optimization
    batch_size: int = 64
    max_iters: int = 10000
    learning_rate: float = 1e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_iters: int = 200

    # Optimizer: 'adam' or 'muon'
    optimizer: str = "adam"

    # Training stage: 'pretrain' or 'midtrain'
    stage: str = "pretrain"

    # Checkpointing
    out_dir: str = "checkpoints"
    init_from: str = "scratch"   # 'scratch' or path to a checkpoint file

    # Logging
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 100
