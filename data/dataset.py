"""
Dataset construction for the GPT maze-search experiment.

Pre-training stage  : thinking = random walk from s,  solution = []
Mid-training stage  : thinking = DFS trace s→t,       solution = DFS path s→t
"""
import random

import torch
from torch.utils.data import Dataset

from data.maze_gen import generate_maze, random_walk, dfs_trace
from data.tokenizer import MazeTokenizer


class MazeDataset(Dataset):
    """
    Generates `n_samples` maze sequences on construction and caches them.

    Each item is a (x, y) pair of token-ID tensors of length `block_size`,
    where y = x shifted left by one (standard language-model target).
    Sequences shorter than block_size are right-padded with PAD tokens;
    longer ones are truncated.
    """

    def __init__(
        self,
        n_samples: int,
        n_nodes: int,
        edge_prob: float,
        stage: str,          # 'pretrain' or 'midtrain'
        block_size: int,
        walk_length: int = 40,
        seed: int = 42,
    ):
        assert stage in ("pretrain", "midtrain"), f"Unknown stage: {stage}"
        self.tokenizer  = MazeTokenizer(n_nodes)
        self.block_size = block_size
        self.pad_id     = self.tokenizer.PAD

        rng = random.Random(seed)
        self.sequences: list[list[int]] = []

        for _ in range(n_samples):
            maze_seed = rng.randint(0, 10 ** 9)
            adj, s, t = generate_maze(n_nodes, edge_prob, seed=maze_seed)

            if stage == "pretrain":
                thinking = random_walk(adj, s, length=walk_length, rng=rng)
                solution = []
            else:  # midtrain
                trace, path = dfs_trace(adj, s, t, rng=rng)
                thinking = trace if trace else [s]
                solution = path if path is not None else []

            tokens = self.tokenizer.encode_sequence(adj, s, t, thinking, solution)

            # Keep at most block_size + 1 tokens so we can form (x, y) pairs
            if len(tokens) > block_size + 1:
                tokens = tokens[: block_size + 1]

            self.sequences.append(tokens)

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        x_raw = seq[:-1]
        y_raw = seq[1:]

        pad = self.pad_id
        length = self.block_size

        x = x_raw + [pad] * (length - len(x_raw))
        y = y_raw + [pad] * (length - len(y_raw))

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
