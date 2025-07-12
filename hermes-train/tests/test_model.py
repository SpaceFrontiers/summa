"""Naming-contract and training smoke tests."""

import json

import numpy as np
import pytest
import torch
from hermes_train.config import ModelDef
from hermes_train.model import Transformer, cross_entropy_loss
from hermes_train.muon import build_optimizers

# Matches `hermes-llm export --model tiny` (GQA + gate exercised via overrides below)
TINY_JSON = {
    "name": "tiny",
    "description": "Small model for quick experiments",
    "vocab_size": 128,
    "max_seq_len": 64,
    "hidden_size": 32,
    "num_layers": 2,
    "block": {
        "name": "tiny_block",
        "attention": {
            "name": "tiny_attn",
            "num_heads": 4,
            "num_kv_heads": None,
            "head_dim": None,
            "dropout": 0.0,
            "bias": False,
            "position_encoding": {"Rope": {"theta": 10000.0, "scaling": None}},
            "window_size": None,
            "causal": True,
        },
        "ffn": {
            "name": "tiny_ffn",
            "hidden_dim": 64,
            "activation": "GELU",
            "bias": False,
            "dropout": 0.0,
            "gate": False,
        },
        "norm": {"norm_type": "RmsNorm", "eps": 1e-5},
        "norm_position": "Pre",
        "residual": True,
        "dropout": 0.0,
    },
    "embeddings": {"tie_weights": False, "dropout": 0.0, "scale": None},
    "output": {"bias": False, "norm": None},
}


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(TINY_JSON))
    return ModelDef.from_json(path)


def test_config_computed_properties(config):
    assert config.n_heads == 4
    assert config.n_kv_heads == 4
    assert config.head_dim == 8
    assert config.intermediate_size == 64
    assert not config.use_bias
    assert not config.use_swiglu
    assert config.rope_theta == 10000.0


def test_tensor_naming_contract(config):
    """state_dict keys must match Candle VarMap names exactly."""
    model = Transformer(config)
    keys = set(model.state_dict().keys())
    expected = {"embedding.weight", "final_norm.weight", "lm_head.weight"}
    for i in range(config.num_layers):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            expected.add(f"layers.{i}.attention.{proj}.weight")
        for proj in ("up_proj", "down_proj"):  # gate=False in this config
            expected.add(f"layers.{i}.feed_forward.{proj}.weight")
        expected.add(f"layers.{i}.attn_norm.weight")
        expected.add(f"layers.{i}.ffn_norm.weight")
    assert keys == expected


def test_gqa_gate_swiglu_naming():
    d = json.loads(json.dumps(TINY_JSON))
    d["block"]["attention"]["num_kv_heads"] = 2
    d["block"]["ffn"]["gate"] = True
    d["block"]["ffn"]["activation"] = "SwiGLU"
    config = ModelDef.from_dict(d)
    model = Transformer(config)
    keys = model.state_dict().keys()
    assert "layers.0.feed_forward.gate_proj.weight" in keys
    # GQA shapes: kv projections are num_kv_heads * head_dim
    assert model.state_dict()["layers.0.attention.k_proj.weight"].shape == (16, 32)
    out = model(torch.randint(0, 128, (2, 10)))
    assert out.shape == (2, 10, 128)


def test_forward_and_loss(config):
    model = Transformer(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, config.vocab_size)
    loss = cross_entropy_loss(logits, ids)
    assert loss.isfinite()


def test_train_step_reduces_loss(config):
    torch.manual_seed(0)
    model = Transformer(config)
    optimizers = build_optimizers(model, lr=1e-2)
    ids = torch.randint(0, config.vocab_size, (4, 16))

    losses = []
    for _ in range(30):
        loss = cross_entropy_loss(model(ids), ids)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5, (
        f"loss did not drop: {losses[0]} -> {losses[-1]}"
    )


def test_checkpoint_roundtrip(config, tmp_path):
    from safetensors.torch import load_file, save_file

    model = Transformer(config)
    path = tmp_path / "ckpt.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    model2 = Transformer(config)
    model2.load_state_dict(load_file(str(path)), strict=True)
    ids = torch.randint(0, config.vocab_size, (1, 8))
    model.eval(), model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids), model2(ids))


HYBRID_JSON = {
    **TINY_JSON,
    "name": "hybrid",
    "num_layers": 6,
    "pattern": [
        {
            "name": "mamba_block",
            "attention": TINY_JSON["block"]["attention"],
            "ssm": {
                "name": "h_ssm",
                "state_dim": 16,
                "conv_kernel": 4,
                "expand": 2,
                "dt_rank": None,
            },
            "ffn": {**TINY_JSON["block"]["ffn"], "gate": True, "activation": "SwiGLU"},
            "norm": {"norm_type": "RmsNorm", "eps": 1e-5},
            "norm_position": "Pre",
            "residual": True,
            "dropout": 0.0,
        },
        {
            "name": "mamba_block",
            "attention": TINY_JSON["block"]["attention"],
            "ssm": {
                "name": "h_ssm",
                "state_dim": 16,
                "conv_kernel": 4,
                "expand": 2,
                "dt_rank": None,
            },
            "ffn": {**TINY_JSON["block"]["ffn"], "gate": True, "activation": "SwiGLU"},
            "norm": {"norm_type": "RmsNorm", "eps": 1e-5},
            "norm_position": "Pre",
            "residual": True,
            "dropout": 0.0,
        },
        {
            "name": "attn_block",
            "attention": TINY_JSON["block"]["attention"],
            "ssm": None,
            "ffn": {**TINY_JSON["block"]["ffn"], "gate": True, "activation": "SwiGLU"},
            "norm": {"norm_type": "RmsNorm", "eps": 1e-5},
            "norm_position": "Pre",
            "residual": True,
            "dropout": 0.0,
        },
    ],
}


@pytest.fixture
def hybrid_config():
    return ModelDef.from_dict(json.loads(json.dumps(HYBRID_JSON)))


def test_hybrid_config(hybrid_config):
    assert hybrid_config.pattern is not None
    assert hybrid_config.block_for_layer(0).is_ssm
    assert hybrid_config.block_for_layer(1).is_ssm
    assert not hybrid_config.block_for_layer(2).is_ssm
    assert hybrid_config.block_for_layer(3).is_ssm
    assert not hybrid_config.block_for_layer(5).is_ssm
    ssm = hybrid_config.block_for_layer(0).ssm
    assert hybrid_config.dt_rank(ssm) == 2  # ceil(32/16)


def test_hybrid_tensor_naming(hybrid_config):
    """SSM tensor names must match the Candle side (layers.{i}.ssm.*)."""
    model = Transformer(hybrid_config)
    keys = set(model.state_dict().keys())
    for i in (0, 1, 3, 4):  # SSM layers under [ssm, ssm, attn] cycling
        for t in (
            "in_proj.weight",
            "conv1d.weight",
            "conv1d.bias",
            "x_proj.weight",
            "dt_proj.weight",
            "dt_proj.bias",
            "A_log",
            "D",
            "out_proj.weight",
        ):
            assert f"layers.{i}.ssm.{t}" in keys, f"missing layers.{i}.ssm.{t}"
        assert f"layers.{i}.attention.q_proj.weight" not in keys
    for i in (2, 5):  # attention layers
        assert f"layers.{i}.attention.q_proj.weight" in keys
        assert f"layers.{i}.ssm.in_proj.weight" not in keys

    # Shape checks against the contract: d_inner = 2*32 = 64, N=16, dt_rank=2
    sd = model.state_dict()
    assert sd["layers.0.ssm.in_proj.weight"].shape == (128, 32)
    assert sd["layers.0.ssm.conv1d.weight"].shape == (64, 1, 4)
    assert sd["layers.0.ssm.x_proj.weight"].shape == (2 + 32, 64)
    assert sd["layers.0.ssm.A_log"].shape == (64, 16)


def test_hybrid_forward_and_train_step(hybrid_config):
    torch.manual_seed(0)
    model = Transformer(hybrid_config)
    ids = torch.randint(0, hybrid_config.vocab_size, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, hybrid_config.vocab_size)
    assert logits.isfinite().all()

    optimizers = build_optimizers(model, lr=5e-3)
    losses = []
    for _ in range(25):
        loss = cross_entropy_loss(model(ids), ids)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers:
            opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7, (
        f"loss did not drop: {losses[0]} -> {losses[-1]}"
    )


def test_tied_embeddings(hybrid_config, tmp_path):
    from safetensors.torch import load_file, save_file

    hybrid_config.embeddings.tie_weights = True
    model = Transformer(hybrid_config)
    keys = model.state_dict().keys()
    assert not any(k.startswith("lm_head") for k in keys)

    ids = torch.randint(0, hybrid_config.vocab_size, (1, 8))
    logits = model(ids)
    assert logits.shape == (1, 8, hybrid_config.vocab_size)

    # Roundtrip without lm_head tensor
    path = tmp_path / "tied.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    model2 = Transformer(hybrid_config)
    model2.load_state_dict(load_file(str(path)), strict=True)
    model.eval(), model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids), model2(ids))

    # Gradients flow to the shared matrix from both ends
    loss = cross_entropy_loss(model(ids), ids)
    loss.backward()
    assert model.embedding.weight.grad is not None


def test_fused_kernel_dispatch(hybrid_config):
    """Off-CUDA the reference scan must be used; fused path needs mamba-ssm."""
    from hermes_train import model as model_mod
    from hermes_train.model import MambaMixer

    mixer = MambaMixer(hybrid_config, hybrid_config.block_for_layer(0).ssm)
    x = torch.randn(1, 8, hybrid_config.hidden_size)
    # CPU input → forward must route to the reference scan regardless of
    # whether mamba-ssm is importable
    ref = mixer._forward_reference(x)
    out = mixer(x)
    assert torch.equal(out, ref) or torch.allclose(out, ref)
    if model_mod._selective_scan_fn is None:
        assert not x.is_cuda  # fused path unreachable without kernels


def test_hybrid_checkpoint_roundtrip(hybrid_config, tmp_path):
    from safetensors.torch import load_file, save_file

    model = Transformer(hybrid_config)
    path = tmp_path / "hybrid.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    model2 = Transformer(hybrid_config)
    model2.load_state_dict(load_file(str(path)), strict=True)
    ids = torch.randint(0, hybrid_config.vocab_size, (1, 8))
    model.eval(), model2.eval()
    with torch.no_grad():
        assert torch.allclose(model(ids), model2(ids))


def test_dataloader_resume_and_sharding(config):
    from hermes_train.data import DataLoader, Dataset

    tokens = np.arange(1000, dtype=np.uint32)
    ds = Dataset(tokens, seq_len=16)

    # Rank sharding: two ranks see disjoint batches covering the same epoch
    loaders = [DataLoader(ds, 8, rank=r, world_size=2) for r in range(2)]
    for loader in loaders:
        loader.reset(seed=1)
    seen = [list(map(tuple, loader)) for loader in loaders]
    assert len(seen[0]) == len(seen[1]) == loaders[0].num_batches()
    assert not set(seen[0]) & set(seen[1])

    # Position resume: skipping ahead reproduces the tail
    loader = DataLoader(ds, 8)
    loader.reset(seed=2)
    full = list(map(tuple, loader))
    loader.reset(seed=2)
    loader.set_position(3 * 8)
    tail = list(map(tuple, loader))
    assert tail == full[3:]
