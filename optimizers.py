"""
Optimizers for the GPT maze-search experiment.

Adam   : standard AdamW (PyTorch built-in), used as the baseline.
Muon   : Momentum + Newton-Schulz Orthogonalization.
         Applied to 2-D weight matrices in the transformer (attention
         projections, MLP weights).  Embeddings and normalisation
         parameters are updated with AdamW instead.

Reference for Muon:
  Kosson et al., "Muon: An optimizer for hidden layers in neural networks",
  https://github.com/KellerJordan/Muon
"""
import math

import torch
from torch.optim import AdamW


# ---------------------------------------------------------------------------
# Newton-Schulz orthogonalisation
# ---------------------------------------------------------------------------

def _ns_orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate the orthogonal polar factor of G via Newton-Schulz iteration.

    The iteration X_{k+1} = a*X_k + (b*A + c*A^2) @ X_k  (A = X_k @ X_k^T)
    converges to the unitary factor when initialised at G / ||G||_F.

    Coefficients (a, b, c) are tuned for ~quintic convergence near the fixed
    point, following the Muon implementation.
    """
    a, b, c = 3.4445, -4.7750, 2.0315

    # Normalise so the largest singular value is ≈ 1
    X = G / (G.norm() + 1e-7)

    # Work on the "tall" orientation so the small dimension is rows
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * (A @ A)) @ X

    if transposed:
        X = X.T

    # Re-scale to match the Frobenius norm of G
    return X * G.norm()


# ---------------------------------------------------------------------------
# Muon optimizer
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    """
    Muon – Nesterov momentum with Newton-Schulz orthogonalization.

    Intended for 2-D weight matrices only.  Use AdamW for embeddings,
    normalisation scales, and biases.

    Args:
        params      : iterable of 2-D weight tensors.
        lr          : learning rate.
        momentum    : Nesterov momentum coefficient (default 0.95).
        nesterov    : whether to apply Nesterov correction (default True).
        ns_steps    : Newton-Schulz iteration count (default 5).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            mu       = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            lr       = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        "Muon is designed for 2-D weight matrices. "
                        "Use AdamW for other parameters."
                    )

                grad  = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)

                # Nesterov lookahead
                effective_grad = grad.add(buf, alpha=mu) if nesterov else buf.clone()

                # Orthogonalise
                update = _ns_orthogonalize(effective_grad, steps=ns_steps)

                p.add_(update, alpha=-lr)

        return loss


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """
    Return an AdamW or Muon+AdamW optimizer according to cfg.optimizer.

    For Muon: weight matrices in attention / MLP layers are handled by Muon;
    everything else (embeddings, RMSNorm scales, lm_head) by AdamW.
    """
    named_params = list(model.named_parameters())

    def _is_matrix(name, param):
        # 2-D weights that are not the embedding table or lm_head
        return (
            param.ndim == 2
            and "wte" not in name
            and "wpe" not in name
            and "lm_head" not in name
        )

    if cfg.optimizer == "adam":
        decay   = [p for n, p in named_params if p.requires_grad and p.ndim >= 2]
        nodecay = [p for n, p in named_params if p.requires_grad and p.ndim < 2]
        param_groups = [
            {"params": decay,   "weight_decay": cfg.weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        return AdamW(param_groups, lr=cfg.learning_rate, betas=(0.9, 0.95))

    elif cfg.optimizer == "muon":
        muon_params  = [p for n, p in named_params if p.requires_grad and _is_matrix(n, p)]
        adam_params  = [p for n, p in named_params if p.requires_grad and not _is_matrix(n, p)]

        # We need two separate optimizers; wrap them in a compound class
        muon_opt = Muon(muon_params, lr=cfg.learning_rate, momentum=0.95)
        adam_opt = AdamW(
            [{"params": adam_params, "weight_decay": 0.0}],
            lr=cfg.learning_rate,
            betas=(0.9, 0.95),
        )
        return _CompoundOptimizer(muon_opt, adam_opt)

    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer!r}")


class _CompoundOptimizer:
    """
    Thin wrapper that forwards step() / zero_grad() to two optimizers.
    Not a torch.optim.Optimizer subclass (no need for parameter groups here).
    """

    def __init__(self, *optimizers):
        self._opts = list(optimizers)

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._opts:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        for opt in self._opts:
            opt.step(closure)

    # Expose param_groups from the first optimizer for LR-scheduler compatibility
    @property
    def param_groups(self):
        groups = []
        for opt in self._opts:
            groups.extend(opt.param_groups)
        return groups

    def state_dict(self):
        return [opt.state_dict() for opt in self._opts]

    def load_state_dict(self, state):
        for opt, s in zip(self._opts, state):
            opt.load_state_dict(s)
