"""Loading a frozen backbone. No weight updates happen anywhere in Jobe v1.

The device is chosen once, here, so the rest of the runtime never asks. The
machine this was built on has a CUDA GPU but a CPU-only torch build, so the
fallback path is not hypothetical - it is the default until a CUDA wheel for
this Python is installed.
"""

from __future__ import annotations

import os

from dataclasses import dataclass


@dataclass(frozen=True)
class LoadedModel:
    """A frozen backbone, its tokenizer, and what it was loaded as."""

    model: object
    tokenizer: object
    name: str
    revision: str | None
    device: str
    dtype: str
    attn_implementation: str | None = None
    adapter: str | None = None      # a merged LoRA adapter directory, when one was applied


def pick_device(requested: str | None = None) -> str:
    """Resolve the device to run on. `auto` prefers CUDA, then MPS, then CPU."""
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_adapter(model, lora_dir: str) -> int:
    """Load a TheLab LoRA adapter onto `model` and merge it into the weights.

    Returns the number of merged modules. The wrappers stay in place but are
    inert once merged, so the readout, the prefix cache and the harness adapter
    all see an ordinary model with the trained deltas folded in.
    """
    try:
        from thelab.decisions.lora import load_lora, merge_lora
    except ImportError as e:  # pragma: no cover
        raise ImportError("applying a LoRA adapter needs TheLab: pip install -e <TheLab checkout>") from e
    load_lora(model, lora_dir)
    return merge_lora(model)


def load(
    name: str,
    *,
    revision: str | None = None,
    device: str | None = "auto",
    dtype: str | None = None,
    attn_implementation: str | None = "eager",
    lora_dir: str | None = None,
) -> LoadedModel:
    """Load a causal LM in eval mode with gradients off.

    Args:
        name: a HuggingFace model id or local path.
        revision: pin a specific commit. Strongly recommended - a moving
            revision silently changes every score you have recorded.
        device: "auto", "cuda", "mps" or "cpu".
        dtype: torch dtype name; defaults to bfloat16 on GPU, float32 on CPU
            (CPU bfloat16 is slow and often unsupported).
        attn_implementation: defaults to ``"eager"``, which is **measured**, not
            assumed: on an RTX 3080 with a 130-token prompt at batch 1, eager ran
            39.1 ms against SDPA's 50.4 ms. A decision prompt is short and
            un-batched, so SDPA's kernel overhead is not repaid. Pass ``"sdpa"``
            for long evidence, or ``None`` to let transformers choose.
        lora_dir: a LoRA adapter saved by TheLab's training loop
            (``lora.pt`` + ``lora.json``). Applied and merged into the frozen
            weights, so every scoring path sees one plain model. Defaults to
            the ``JOBE_LORA_DIR`` environment variable, which is how the
            benchmark adapter and the gate pick an adapter up without a code
            change. Needs TheLab installed only when actually used.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = pick_device(device)
    if dtype is None:
        dtype = "float32" if resolved == "cpu" else "bfloat16"
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    model = AutoModelForCausalLM.from_pretrained(
        name, revision=revision, dtype=torch_dtype, **extra
    )
    model.to(resolved)
    lora_dir = lora_dir if lora_dir is not None else os.environ.get("JOBE_LORA_DIR") or None
    if lora_dir:
        apply_adapter(model, lora_dir)
    model.eval()
    model.requires_grad_(False)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        revision=revision,
        device=resolved,
        dtype=dtype,
        attn_implementation=getattr(model.config, "_attn_implementation", attn_implementation),
        adapter=lora_dir,
    )
