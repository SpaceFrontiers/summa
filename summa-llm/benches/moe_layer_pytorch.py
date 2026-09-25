#!/usr/bin/env python3
"""PyTorch baselines for the Summa 200M MoE layer geometry."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F


class MoeLayer(torch.nn.Module):
    def __init__(self, hidden: int, intermediate: int, experts: int, top_k: int):
        super().__init__()
        self.hidden = hidden
        self.intermediate = intermediate
        self.experts = experts
        self.top_k = top_k
        self.router = torch.nn.Parameter(torch.empty(hidden, experts))
        self.up = torch.nn.Parameter(torch.empty(experts, 2 * intermediate, hidden))
        self.down = torch.nn.Parameter(torch.empty(experts, hidden, intermediate))
        torch.nn.init.normal_(self.router, std=hidden**-0.5)
        torch.nn.init.normal_(self.up, std=hidden**-0.5)
        torch.nn.init.normal_(self.down, std=intermediate**-0.5)

    def route_details(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", enabled=False):
            logits = x.float() @ self.router.float()
            probabilities = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
        return (
            logits,
            probabilities,
            weights / weights.sum(dim=-1, keepdim=True),
            indices,
        )

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, weights, indices = self.route_details(x)
        return weights, indices

    def _expert(self, x: torch.Tensor, expert: int) -> torch.Tensor:
        projected = F.linear(x, self.up[expert])
        gate, value = projected.split(self.intermediate, dim=-1)
        return F.linear(F.silu(gate) * value, self.down[expert])

    def sparse_loop(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.route(x)
        output = torch.zeros_like(x)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for expert in range(self.experts):
                assignment = indices == expert
                rows = torch.nonzero(assignment, as_tuple=True)[0]
                if rows.numel() == 0:
                    continue
                gates = (weights * assignment).sum(dim=-1).index_select(0, rows)
                values = self._expert(x.index_select(0, rows), expert)
                output = output.index_add(
                    0, rows, values * gates[:, None].to(values.dtype)
                )
        return output

    def masked(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.route(x)
        output = torch.zeros_like(x)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for expert in range(self.experts):
                assignment = indices == expert
                gates = (weights * assignment).sum(dim=-1)
                values = self._expert(x, expert)
                output = output + values * gates[:, None].to(values.dtype)
        return output

    def _grouped_from_routes(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError(
                "this PyTorch build has no torch.nn.functional.grouped_mm"
            )
        token_rows = torch.arange(x.shape[0], device=x.device).repeat_interleave(
            self.top_k
        )
        experts = indices.reshape(-1)
        gates = weights.reshape(-1)
        order = torch.argsort(experts)
        experts = experts.index_select(0, order)
        token_rows = token_rows.index_select(0, order)
        gates = gates.index_select(0, order)
        counts = torch.bincount(experts, minlength=self.experts)
        offsets = counts.cumsum(0).to(torch.int32)
        routed = x.index_select(0, token_rows)
        up = self.up.transpose(1, 2).to(torch.bfloat16)
        down = self.down.transpose(1, 2).to(torch.bfloat16)
        projected = F.grouped_mm(routed, up, offs=offsets)
        gate, value = projected.split(self.intermediate, dim=-1)
        hidden = F.silu(gate) * value
        routed_output = F.grouped_mm(hidden, down, offs=offsets)
        return (
            torch.zeros_like(x).index_add(
                0, token_rows, routed_output * gates[:, None].to(routed_output.dtype)
            ),
            counts,
        )

    def grouped(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self.route(x)
        return self._grouped_from_routes(x, weights, indices)[0]

    def grouped_with_router_losses(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, probabilities, weights, indices = self.route_details(x)
        output, counts = self._grouped_from_routes(x, weights, indices)
        route_fraction = counts.float() / (x.shape[0] * self.top_k)
        balance = (
            self.experts * (probabilities.mean(dim=0) * route_fraction).sum() * 0.01
        )
        router_z = torch.logsumexp(logits, dim=-1).square().mean() * 0.001
        return output, balance + router_z


def measure(
    operation: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    *,
    tokens: int,
    warmup: int,
    iterations: int,
    backward: bool,
    layer: MoeLayer,
    x: torch.Tensor,
) -> dict[str, float | str]:
    def iteration() -> None:
        layer.zero_grad(set_to_none=True)
        x.grad = None
        if backward:
            result = operation()
            if isinstance(result, tuple):
                output, auxiliary = result
                loss = output.float().square().mean() + auxiliary
            else:
                loss = result.float().square().mean()
            loss.backward()
        else:
            with torch.no_grad():
                operation()

    for _ in range(warmup):
        iteration()
        torch.cuda.synchronize()
    elapsed = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        started = time.perf_counter()
        iteration()
        torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - started) * 1_000)
    median_ms = statistics.median(elapsed)
    return {
        "mode": "forward_backward_core" if backward else "forward",
        "median_ms": median_ms,
        "min_ms": min(elapsed),
        "tokens_per_second": tokens / (median_ms / 1_000),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--implementations",
        nargs="+",
        choices=("grouped", "sparse_loop", "masked"),
        default=("grouped", "sparse_loop", "masked"),
    )
    args = parser.parse_args()
    if args.tokens <= 0 or args.iterations <= 0:
        parser.error("--tokens and --iterations must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    layer = MoeLayer(args.hidden, args.intermediate, args.experts, args.top_k).cuda()
    x = torch.randn(
        args.tokens,
        args.hidden,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    implementations = {
        "grouped": layer.grouped,
        "sparse_loop": layer.sparse_loop,
        "masked": layer.masked,
    }
    results = []
    skipped = {}
    for name in args.implementations:

        def operation(name: str = name) -> torch.Tensor:
            return implementations[name](x)

        try:
            forward = measure(
                operation,
                tokens=args.tokens,
                warmup=args.warmup,
                iterations=args.iterations,
                backward=False,
                layer=layer,
                x=x,
            )
            training = measure(
                operation,
                tokens=args.tokens,
                warmup=args.warmup,
                iterations=args.iterations,
                backward=True,
                layer=layer,
                x=x,
            )
            measurements = [forward, training]
            if name == "grouped":

                def operation_with_router_losses() -> tuple[torch.Tensor, torch.Tensor]:
                    return layer.grouped_with_router_losses(x)

                with_router_losses = measure(
                    operation_with_router_losses,
                    tokens=args.tokens,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    backward=True,
                    layer=layer,
                    x=x,
                )
                with_router_losses["mode"] = "forward_backward_with_router_losses"
                measurements.append(with_router_losses)
            results.append(
                {
                    "implementation": f"pytorch_{name}",
                    "measurements": measurements,
                }
            )
        except (RuntimeError, NotImplementedError) as error:
            skipped[name] = str(error).splitlines()[0]

    print(
        json.dumps(
            {
                "framework": f"torch-{torch.__version__}",
                "device": torch.cuda.get_device_name(),
                "dtype": "bfloat16_tensor_cores_fp32_router",
                "hidden_size": args.hidden,
                "intermediate_size": args.intermediate,
                "experts": args.experts,
                "top_k": args.top_k,
                "tokens": args.tokens,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "results": results,
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
