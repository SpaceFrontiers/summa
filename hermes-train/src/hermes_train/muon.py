"""Muon optimizer — MomentUm Orthogonalized by Newton-Schulz.

Single-device reference implementation (Jordan et al.). Intended for 2D weight
matrices only; optimize embeddings, norms, biases and the LM head with AdamW.
"""

from __future__ import annotations

import torch
from torch import Tensor


def zeropower_via_newtonschulz5(g: Tensor, steps: int = 5) -> Tensor:
    """Approximate orthogonalization of ``g`` via quintic Newton-Schulz iteration."""
    assert g.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16() if g.device.type == "cuda" else g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        m = x @ x.mT
        x = a * x + (b * m + c * m @ m) @ x
    if transposed:
        x = x.mT
    return x.to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise ValueError(
                        f"Muon only handles 2D parameters, got shape {tuple(p.shape)}; "
                        "route others to AdamW"
                    )
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - group["momentum"])
                g = g.lerp(buf, group["momentum"]) if group["nesterov"] else buf

                update = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                # Scale so update RMS matches AdamW conventions across shapes
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5

                if group["weight_decay"] > 0.0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(update, alpha=-group["lr"] * scale)

        return loss


def build_optimizers(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    muon_lr: float | None = None,
) -> list[torch.optim.Optimizer]:
    """Split parameters into Muon (2D hidden matrices) and AdamW (the rest).

    Embeddings and the LM head stay on AdamW, per standard Muon practice.
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embedding" not in name and "lm_head" not in name:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizers: list[torch.optim.Optimizer] = []
    if muon_params:
        optimizers.append(
            Muon(muon_params, lr=muon_lr if muon_lr is not None else lr * 20.0)
        )
    if adamw_params:
        optimizers.append(
            torch.optim.AdamW(
                adamw_params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8
            )
        )
    return optimizers
