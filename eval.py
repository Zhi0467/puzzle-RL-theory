"""
Evaluation script for the GPT maze-search experiment.

Metrics reported:
  - Cross-entropy loss on the validation set
  - Path validity rate: fraction of generated solutions that form a valid s→t walk
  - Path optimality rate: fraction of valid solutions that are also shortest

Usage:
  python eval.py --checkpoint checkpoints/midtrain_adam/ckpt_best.pt
  python eval.py --checkpoint checkpoints/midtrain_adam/ckpt_best.pt --n_eval 500
"""
import argparse
import random

import torch

from config import GPTConfig, MazeConfig
from data.maze_gen import generate_maze, is_valid_path, bfs_shortest_path
from data.tokenizer import MazeTokenizer
from model.gpt import GPT


# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str) -> tuple[GPT, MazeTokenizer, dict]:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    maze_cfg_dict = state.get("maze_cfg", {})
    gpt_cfg_dict  = state.get("gpt_cfg",  {})

    n_nodes = maze_cfg_dict.get("n_nodes", 20)
    tokenizer = MazeTokenizer(n_nodes)

    gpt_cfg = GPTConfig(
        vocab_size = tokenizer.vocab_size,
        block_size = gpt_cfg_dict.get("block_size", 512),
        n_layer    = gpt_cfg_dict.get("n_layer",    6),
        n_head     = gpt_cfg_dict.get("n_head",     6),
        n_embd     = gpt_cfg_dict.get("n_embd",     192),
        dropout    = 0.0,
        bias       = gpt_cfg_dict.get("bias",       False),
    )
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tokenizer, maze_cfg_dict


@torch.no_grad()
def evaluate_generation(
    model: GPT,
    tokenizer: MazeTokenizer,
    n_nodes: int,
    edge_prob: float,
    n_eval: int,
    max_new_tokens: int,
    device: str,
    temperature: float = 1.0,
    seed: int = 9999,
) -> dict:
    """
    Generate solutions for `n_eval` random mazes and compute:
      - valid_rate    : fraction where generated path is a valid s→t path
      - optimal_rate  : fraction where valid path is also a shortest path
      - avg_path_len  : average length of generated paths (valid ones)
    """
    rng = random.Random(seed)
    n_valid = 0
    n_optimal = 0
    path_lengths = []

    for _ in range(n_eval):
        maze_seed = rng.randint(0, 10 ** 9)
        adj, s, t = generate_maze(n_nodes, edge_prob, seed=maze_seed)

        # Build the graph-only prefix (everything up to and including the first SEP)
        prefix = tokenizer.encode_graph(adj, s, t)
        prefix.append(tokenizer.SEP)
        prompt = torch.tensor([prefix], dtype=torch.long, device=device)

        generated = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            eos_token=tokenizer.EOS,
        )

        gen_tokens = generated[0].tolist()
        _, solution = tokenizer.decode_generated(gen_tokens)

        if is_valid_path(adj, solution, s, t):
            n_valid += 1
            path_lengths.append(len(solution))
            shortest = bfs_shortest_path(adj, s, t)
            if shortest is not None and len(solution) == len(shortest):
                n_optimal += 1

    valid_rate   = n_valid / n_eval
    optimal_rate = n_optimal / n_eval
    avg_len      = sum(path_lengths) / len(path_lengths) if path_lengths else 0.0

    return dict(
        n_eval      = n_eval,
        valid_rate  = valid_rate,
        optimal_rate= optimal_rate,
        avg_path_len= avg_len,
    )


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Evaluate a GPT maze-search checkpoint")
    p.add_argument("--checkpoint",      required=True, help="Path to .pt checkpoint file")
    p.add_argument("--n_eval",          type=int,   default=200,  help="Number of mazes to evaluate")
    p.add_argument("--max_new_tokens",  type=int,   default=200,  help="Max generation length")
    p.add_argument("--temperature",     type=float, default=1.0)
    p.add_argument("--seed",            type=int,   default=9999)
    args = p.parse_args()

    device = (
        "cuda"  if torch.cuda.is_available()           else
        "mps"   if torch.backends.mps.is_available()   else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    model, tokenizer, maze_cfg_dict = load_model(args.checkpoint, device)
    n_nodes   = maze_cfg_dict.get("n_nodes",    20)
    edge_prob = maze_cfg_dict.get("edge_prob",  0.15)

    print(f"Model loaded. n_nodes={n_nodes}, edge_prob={edge_prob}")
    print(f"Evaluating on {args.n_eval} mazes …")

    results = evaluate_generation(
        model, tokenizer,
        n_nodes   = n_nodes,
        edge_prob = edge_prob,
        n_eval    = args.n_eval,
        max_new_tokens = args.max_new_tokens,
        device    = device,
        temperature = args.temperature,
        seed      = args.seed,
    )

    print("\n=== Evaluation Results ===")
    print(f"  Mazes evaluated : {results['n_eval']}")
    print(f"  Valid path rate : {results['valid_rate']:.3f}")
    print(f"  Optimal rate    : {results['optimal_rate']:.3f}")
    print(f"  Avg path length : {results['avg_path_len']:.2f}")


if __name__ == "__main__":
    main()
