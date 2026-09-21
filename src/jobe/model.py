"""Loading a frozen backbone. No weight updates happen anywhere in Jobe v1.

The device is chosen once, here, so the rest of the runtime never asks. The
machine this was built on has a CUDA GPU but a CPU-only torch build, so the
fallback path is not hypothetical - it is the default until a CUDA wheel for
this Python is installed.
"""

from __future__ import annotations

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


def load(
    name: str,
    *,
    revision: str | None = None,
    device: str | None = "auto",
    dtype: str | None = None,
) -> LoadedModel:
    """Load a causal LM in eval mode with gradients off.

    Args:
        name: a HuggingFace model id or local path.
        revision: pin a specific commit. Strongly recommended - a moving
            revision silently changes every score you have recorded.
        device: "auto", "cuda", "mps" or "cpu".
        dtype: torch dtype name; defaults to bfloat16 on GPU, float32 on CPU
            (CPU bfloat16 is slow and often unsupported).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved = pick_device(device)
    if dtype is None:
        dtype = "float32" if resolved == "cpu" else "bfloat16"
    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch_dtype)
    model.to(resolved)
    model.eval()
    model.requires_grad_(False)

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        revision=revision,
        device=resolved,
        dtype=dtype,
    )
