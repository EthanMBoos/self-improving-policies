"""RLinf extension module (export RLINF_EXT_MODULE=sip_rl.rlinf_ext). Every RLinf worker imports it and calls
register() at startup, so all model copies (actor, rollout, eval) get the same architecture.

Wraps the built-in "openpi" model builder: when the model config has ``expert_lora_rank``, LoRA is applied to the
action expert after the pretrained weights are loaded (see sip_rl/expert_lora.py).
"""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    import rlinf.models as rlinf_models

    original = rlinf_models._MODEL_REGISTRY["openpi"]
    if getattr(original, "_sip_expert_lora", False):
        return

    def build_openpi_with_expert_lora(cfg, torch_dtype):
        model = original(cfg, torch_dtype)
        rank = cfg.get("expert_lora_rank", None)
        if rank:
            from sip_rl.expert_lora import apply_expert_lora

            stats = apply_expert_lora(model, rank=int(rank), alpha=cfg.get("expert_lora_alpha", None))
            logger.warning(f"[sip_rl] expert LoRA applied: {stats}")
        ckpt = cfg.get("trainable_ckpt", None)
        if ckpt:  # evaluation of a training snapshot: LoRA + small heads on top of the base checkpoint
            import torch

            state = torch.load(ckpt, map_location="cpu")
            own = dict(model.named_parameters())
            skipped = [k for k in state if k not in own]
            if any(not k.startswith("value_head.") for k in skipped):
                raise KeyError(f"snapshot keys not in model: {[k for k in skipped if not k.startswith('value_head.')][:5]}")
            with torch.no_grad():
                for k, v in state.items():
                    if k in own:
                        own[k].copy_(v.to(own[k].dtype))
            logger.warning(f"[sip_rl] loaded {len(state) - len(skipped)} trainable tensors from {ckpt}")
        return model

    build_openpi_with_expert_lora._sip_expert_lora = True
    rlinf_models.register_model("openpi", build_openpi_with_expert_lora, category="embodied", force=True)

    # FSDP wraps each listed class as its own unit; LoRAAdapter (float32) must not be flattened with bf16 blocks.
    from rlinf.models.embodiment.openpi.pi0 import Pi0

    base_prop = Pi0._no_split_modules
    if not getattr(base_prop.fget, "_sip_lora", False):
        prop = property(lambda self: list(base_prop.fget(self) or []) + ["LoRAAdapter"])
        prop.fget._sip_lora = True
        Pi0._no_split_modules = prop
