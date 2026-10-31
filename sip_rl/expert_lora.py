"""LoRA on the pi0.5 action expert of RLinf's openpi model (attention + MLP), everything else frozen.

RLinf's Gemma keeps one entry per expert in each layer: index 0 is the VLM (PaliGemma), index 1 the action expert.
The attention projections are nn.Linear; the MLP (FeedForward) uses raw parameter tensors, which PEFT cannot adapt,
so both get small explicit LoRA wrappers here. Frozen weights keep their parameter names, so pretrained checkpoints
load as before.

Precision: the base model runs in bf16, but LoRA matrices live in float32 ``LoRAAdapter`` modules. An Adam step of
~1e-5 on a bf16 weight of magnitude ~0.03 is below bf16 resolution and would be lost; float32 master weights keep
it. FSDP needs one dtype per wrap unit, so ``LoRAAdapter`` is added to the model's FSDP wrap classes
(sip_rl/rlinf_ext.py) and becomes its own float32 unit inside the bf16 transformer blocks.

Trainable after ``apply_expert_lora``: LoRA on expert attention (q, k, v, o) and MLP (gating x2, linear), the value
head, and the small action/time projections. Frozen: SigLIP, the VLM, all base expert weights (incl. adaRMS
modulation). This mirrors openpi's ``gemma_300m_lora`` (attn + ffn LoRA) with the VLM frozen as in piRL.
"""

from __future__ import annotations

import math
import re

import torch
from torch import nn

from rlinf.models.embodiment.openpi.modules.gemma import FeedForward, gelu_glu

EXTRA_TRAINABLE = ("value_head", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
                   "action_time_mlp_in", "action_time_mlp_out", "state_proj")


class LoRAAdapter(nn.Module):
    """delta(x) = scale * x @ A @ B with float32 A [in, r], B [r, out]; B = 0 at init (no change to the model)."""

    def __init__(self, n_in: int, n_out: int, rank: int, alpha: float, device=None):
        super().__init__()
        self.scale = alpha / rank
        a = torch.empty(n_in, rank, dtype=torch.float32, device=device)
        nn.init.kaiming_uniform_(a.T, a=math.sqrt(5))  # as nn.Linear / PEFT default for A
        self.A = nn.Parameter(a)
        self.B = nn.Parameter(torch.zeros(rank, n_out, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.A.to(x.dtype)) @ self.B.to(x.dtype) * self.scale

    def weight_delta(self) -> torch.Tensor:
        """[in, out] update to add to a weight used as x @ W."""
        return self.A.float() @ self.B.float() * self.scale


class LoRALinear(nn.Linear):
    """nn.Linear (frozen) + LoRA. Keeps ``weight`` under the same name."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None, device="meta")
        self.weight = base.weight
        self.bias = base.bias
        self.lora = LoRAAdapter(base.in_features, base.out_features, rank, alpha, device=base.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.lora(x)


class LoRAFeedForward(FeedForward):
    """Gemma GeGLU MLP (frozen) with LoRA on both gating matrices and the output matrix."""

    def __init__(self, base: FeedForward, rank: int, alpha: float):
        nn.Module.__init__(self)
        self.features, self.hidden_dim = base.features, base.hidden_dim
        self.w_gating = base.w_gating
        self.w_linear = base.w_linear
        dev = base.w_linear.device
        self.lora_g0 = LoRAAdapter(self.features, self.hidden_dim, rank, alpha, device=dev)
        self.lora_g1 = LoRAAdapter(self.features, self.hidden_dim, rank, alpha, device=dev)
        self.lora_l = LoRAAdapter(self.hidden_dim, self.features, rank, alpha, device=dev)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        ff_gate = torch.matmul(x, self.w_gating[0].to(dtype)) + self.lora_g0(x)
        ff1 = torch.matmul(x, self.w_gating[1].to(dtype)) + self.lora_g1(x)
        act = gelu_glu(ff_gate, ff1)
        return torch.matmul(act, self.w_linear.to(dtype)) + self.lora_l(act)


def apply_expert_lora(model: nn.Module, rank: int, alpha: float | None = None, expert_id: int = 1) -> dict:
    """Wrap the action expert's attention and MLP with LoRA and set requires_grad (see module docstring)."""
    alpha = float(alpha if alpha is not None else rank)
    for p in model.parameters():
        p.requires_grad = False
    n_lin = n_ffn = 0
    for block in model.llm.layers:
        attn = block.attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            plist = getattr(attn, name)
            if plist[expert_id] is not None:
                plist[expert_id] = LoRALinear(plist[expert_id], rank, alpha)
                n_lin += 1
        block.mlps[expert_id] = LoRAFeedForward(block.mlps[expert_id], rank, alpha)
        n_ffn += 1
    for name, p in model.named_parameters():
        if ".lora" in name or name.split(".")[0] in EXTRA_TRAINABLE:
            p.requires_grad = True
    if hasattr(model, "_mark_fsdp_wrap_names"):  # new modules need RLinf's FSDP wrap-name markers too
        model._mark_fsdp_wrap_names()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if ".lora" in n)
    return {"lora_linear": n_lin, "lora_ffn": n_ffn, "trainable": n_train, "lora_params": n_lora,
            "total": sum(p.numel() for p in model.parameters())}


_ADAPTER_KEY = re.compile(r"(.*)\.(lora|lora_g0|lora_g1|lora_l)\.(A|B)$")


def merge_expert_lora(state_dict: dict[str, torch.Tensor], rank: int, alpha: float | None = None) -> dict:
    """Fold LoRA deltas into the base weights; returns a state dict loadable by the plain (un-wrapped) model."""
    scale = float(alpha if alpha is not None else rank) / rank
    sd = {k: v for k, v in state_dict.items() if not _ADAPTER_KEY.match(k)}

    def delta(prefix, which):
        return state_dict[f"{prefix}.{which}.A"].float() @ state_dict[f"{prefix}.{which}.B"].float() * scale

    prefixes = {(m.group(1), m.group(2)) for k in state_dict if (m := _ADAPTER_KEY.match(k))}
    for pre, which in prefixes:
        if which == "lora":  # Linear: weight is [out, in], forward is x @ W^T
            w = sd[f"{pre}.weight"]
            sd[f"{pre}.weight"] = (w.float() + delta(pre, "lora").T).to(w.dtype)
        elif which == "lora_l":
            wl = sd[f"{pre}.w_linear"]
            sd[f"{pre}.w_linear"] = (wl.float() + delta(pre, "lora_l")).to(wl.dtype)
        elif which == "lora_g0":  # handle both gating matrices once
            g = sd[f"{pre}.w_gating"]
            gf = g.float().clone()
            gf[0] += delta(pre, "lora_g0")
            gf[1] += delta(pre, "lora_g1")
            sd[f"{pre}.w_gating"] = gf.to(g.dtype)
    return sd
