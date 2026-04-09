"""
Training script for the GPT maze-search experiment.

Usage:
  # Pre-training with Adam
  python train.py --stage pretrain --optimizer adam

  # Mid-training from a pre-trained checkpoint with Muon
  python train.py --stage midtrain --optimizer muon --init_from checkpoints/pretrain_adam/ckpt_best.pt

  # Quick smoke-test
  python train.py --stage pretrain --max_iters 200 --n_train 500 --n_val 100
"""
import argparse
import math
import os
import time

import torch
from torch.utils.data import DataLoader

from config import GPTConfig, MazeConfig, TrainConfig
from data.dataset import MazeDataset
from data.tokenizer import MazeTokenizer
from model.gpt import GPT
from optimizers import build_optimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_lr(it: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warm-up."""
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    return cfg.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: GPT,
    loader: DataLoader,
    device: str,
    pad_id: int,
    n_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        # Replace PAD in targets with -100 so they are ignored in loss
        y_masked = y.masked_fill(y == pad_id, -100)
        _, loss = model(x, targets=y_masked, ignore_index=-100)
        total_loss += loss.item()
        count += 1
    model.train()
    return total_loss / max(1, count)


def save_checkpoint(model, optimizer, iter_num, val_loss, cfg, gpt_cfg, maze_cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = dict(
        model      = model.state_dict(),
        optimizer  = optimizer.state_dict(),
        iter_num   = iter_num,
        val_loss   = val_loss,
        gpt_cfg    = gpt_cfg.__dict__,
        maze_cfg   = maze_cfg.__dict__,
        train_cfg  = cfg.__dict__,
    )
    torch.save(state, path)
    print(f"  [ckpt] saved → {path}")


def load_checkpoint(path, model, optimizer):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    try:
        optimizer.load_state_dict(state["optimizer"])
    except Exception as e:
        print(f"  [warn] could not load optimizer state: {e}")
    return state.get("iter_num", 0), state.get("val_loss", float("inf"))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: TrainConfig, maze_cfg: MazeConfig, gpt_cfg: GPTConfig):
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    torch.manual_seed(1337)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print(f"Building datasets (stage={cfg.stage}) …")
    tokenizer = MazeTokenizer(maze_cfg.n_nodes)
    gpt_cfg.vocab_size = tokenizer.vocab_size

    train_ds = MazeDataset(
        cfg.n_train, maze_cfg.n_nodes, maze_cfg.edge_prob,
        cfg.stage, gpt_cfg.block_size, maze_cfg.walk_length, seed=42,
    )
    val_ds = MazeDataset(
        cfg.n_val, maze_cfg.n_nodes, maze_cfg.edge_prob,
        cfg.stage, gpt_cfg.block_size, maze_cfg.walk_length, seed=1234,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        pin_memory=(device == "cuda"), num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=(device == "cuda"), num_workers=0,
    )

    train_iter = iter(train_loader)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = GPT(gpt_cfg).to(device)
    print(f"GPT parameters: {model.num_params():,}")

    optimizer = build_optimizer(model, cfg)

    start_iter = 0
    best_val_loss = float("inf")

    out_dir = os.path.join(cfg.out_dir, f"{cfg.stage}_{cfg.optimizer}")
    ckpt_last = os.path.join(out_dir, "ckpt_last.pt")
    ckpt_best = os.path.join(out_dir, "ckpt_best.pt")

    if cfg.init_from != "scratch" and os.path.isfile(cfg.init_from):
        print(f"Resuming from {cfg.init_from} …")
        start_iter, best_val_loss = load_checkpoint(cfg.init_from, model, optimizer)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    model.train()
    t0 = time.time()
    log_rows: list[dict] = []

    for it in range(start_iter, cfg.max_iters):
        # Learning-rate schedule
        lr = get_lr(it, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Fetch next batch (cycle through loader)
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)
        y_masked = y.masked_fill(y == tokenizer.PAD, -100)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y_masked, ignore_index=-100)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if it % cfg.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {it:6d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt*1000:.0f} ms")

        # ------------------------------------------------------------------
        # Evaluation
        # ------------------------------------------------------------------
        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            val_loss = estimate_loss(model, val_loader, device, tokenizer.PAD, cfg.eval_iters)
            train_loss_est = estimate_loss(model, train_loader, device, tokenizer.PAD, cfg.eval_iters)
            print(
                f"  [eval] iter {it:6d} | "
                f"train {train_loss_est:.4f} | val {val_loss:.4f}"
            )
            log_rows.append(dict(iter=it, train_loss=train_loss_est, val_loss=val_loss, lr=lr))
            save_checkpoint(model, optimizer, it, val_loss, cfg, gpt_cfg, maze_cfg, ckpt_last)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, it, val_loss, cfg, gpt_cfg, maze_cfg, ckpt_best)

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------
    import csv
    log_path = os.path.join(out_dir, "training_log.csv")
    os.makedirs(out_dir, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "train_loss", "val_loss", "lr"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"Training log → {log_path}")
    print(f"Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GPT maze-search training")
    p.add_argument("--stage",        default="pretrain",  choices=["pretrain", "midtrain"])
    p.add_argument("--optimizer",    default="adam",      choices=["adam", "muon"])
    p.add_argument("--n_nodes",      type=int,   default=20)
    p.add_argument("--edge_prob",    type=float, default=0.15)
    p.add_argument("--walk_length",  type=int,   default=40)
    p.add_argument("--n_train",      type=int,   default=20000)
    p.add_argument("--n_val",        type=int,   default=2000)
    p.add_argument("--batch_size",   type=int,   default=64)
    p.add_argument("--max_iters",    type=int,   default=10000)
    p.add_argument("--lr",           type=float, default=1e-3, dest="learning_rate")
    p.add_argument("--n_layer",      type=int,   default=6)
    p.add_argument("--n_head",       type=int,   default=6)
    p.add_argument("--n_embd",       type=int,   default=192)
    p.add_argument("--block_size",   type=int,   default=512)
    p.add_argument("--eval_interval",type=int,   default=500)
    p.add_argument("--log_interval", type=int,   default=100)
    p.add_argument("--out_dir",      default="checkpoints")
    p.add_argument("--init_from",    default="scratch")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    maze_cfg = MazeConfig(
        n_nodes    = args.n_nodes,
        edge_prob  = args.edge_prob,
        walk_length= args.walk_length,
    )
    gpt_cfg = GPTConfig(
        block_size = args.block_size,
        n_layer    = args.n_layer,
        n_head     = args.n_head,
        n_embd     = args.n_embd,
    )
    train_cfg = TrainConfig(
        stage          = args.stage,
        optimizer      = args.optimizer,
        n_train        = args.n_train,
        n_val          = args.n_val,
        batch_size     = args.batch_size,
        max_iters      = args.max_iters,
        learning_rate  = args.learning_rate,
        eval_interval  = args.eval_interval,
        log_interval   = args.log_interval,
        out_dir        = args.out_dir,
        init_from      = args.init_from,
    )

    train(train_cfg, maze_cfg, gpt_cfg)
