"""
Prompt composition utilities:  x_tilde = clip_range(x + eps * h(x))

Kept OUTSIDE the Coordinator: the trainer composes the prompted image, and
the black-box CLIP only ever sees x_tilde.
"""

import torch

from blackvip.constants import CLIP_PIXEL_MAX, CLIP_PIXEL_MIN, NUM_CHANNELS


def _bounds(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel min/max tensors on x's device/dtype, shaped to broadcast against x."""
    shape = (1, NUM_CHANNELS, 1, 1) if x.dim() == 4 else (NUM_CHANNELS, 1, 1)
    lo = torch.tensor(CLIP_PIXEL_MIN, device=x.device, dtype=x.dtype).view(shape)
    hi = torch.tensor(CLIP_PIXEL_MAX, device=x.device, dtype=x.dtype).view(shape)
    return lo, hi


def clip_range(x: torch.Tensor) -> torch.Tensor:
    """Clamp a CLIP-normalized image to the valid RGB range [0, 1], per channel.

    Accepts (N, 3, H, W) or (3, H, W).
    """
    if x.dim() not in (3, 4):
        raise ValueError(f"clip_range expects a 3-D or 4-D tensor, got {x.dim()}-D")
    channel_dim = 1 if x.dim() == 4 else 0
    if x.shape[channel_dim] != NUM_CHANNELS:
        raise ValueError(
            f"clip_range expects {NUM_CHANNELS} channels, got shape {tuple(x.shape)}"
        )
    lo, hi = _bounds(x)
    return torch.maximum(torch.minimum(x, hi), lo)


def apply_prompt(x: torch.Tensor, prompt: torch.Tensor, eps: float) -> torch.Tensor:
    """Return the prompted image clip_range(x + eps * prompt)."""
    if x.shape != prompt.shape:
        raise ValueError(
            f"image and prompt shapes differ: {tuple(x.shape)} vs {tuple(prompt.shape)}"
        )
    return clip_range(x + eps * prompt)
