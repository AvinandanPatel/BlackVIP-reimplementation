"""
input: CLIP-normalized image, (N, C, H, W) -> (N, 3, 224, 224)
output: z_x (N, z_dim)
"""

# --- importing necessary packages
import torch
import torch.nn as nn

from blackvip.constants import IMAGE_SIZE, NUM_CHANNELS

from transformers import ViTForImageClassification
from blackvip.constants import ENCODER_Z_DIMS, HF_ENCODER_IDS


# --- Frozen Pre Trained Encoder: Skeleton of each Frozen Encoder
# FrozenEncoder is a baseclass

class FrozenEncoder(nn.Module):
    trainable = False

    def __init__(self):
        # set placeholder name, z_dim
        super().__init__()
        self.name: str = ""
        self.z_dim: int = 0

    def _freeze(self):
        """
        Turn-off the gradients for every parameter and switch to eval mode.
        """
        # no grads + eval mode
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        

    def train(self, mode = True): 
        """ 
        Always stays in eval mode 
        
        Call the parent's train with False: super().train(False). 
        The parent version sets self.training = False 
        and then recurses into every child (conv, bn, and later the whole ViT).
        Don't set self.training = False by hand; 
        that would only change the top-level module and leave bn in training mode.
        """
        return super().train(False)

    def forward(self, x):
        # check the input -> no_grad -> _extract -> check output -> float
        self._check_input(x)
        with torch.no_grad():
            z = self._extract(x)
        self._check_output(z, x.shape[0])
        return z.float()

    def _check_input(self, x):
        """ Check the input """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"[{self.name}] expects a torch.Tensor, got {type(x).__name__}")
        if x.dim() != 4:
            raise ValueError(f"[{self.name}] expects a 4-D batch (N, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != NUM_CHANNELS:
            raise ValueError(f"[{self.name}] expects {NUM_CHANNELS} channels at dim 1, got shape {tuple(x.shape)}")
        if x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(f"[{self.name}] expects {IMAGE_SIZE}x{IMAGE_SIZE} images, got shape {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"[{self.name}] expects a float tensor (CLIP-normalized), got {x.dtype}")


    def _extract(self, x):
        # raise NoImplementationError
        raise NotImplementedError(f"{type(self).__name__} must implement _extract()")
    
    def _check_output(self, z: torch.Tensor, n: int) -> None:
        """ Features must be (N, z_dim)"""
        expected = (n, self.z_dim)
        if tuple(z.shape) != expected:
            raise ValueError(f"[{self.name}] _extract must return {expected}, got {tuple(z.shape)}")


# -----------------------------------------------------------------
# For the frozen encoder of Coordinator, 
# we use ImageNet pre-trained vit-mae-basecheckpoint

# Implementation of a dummy encoder for testing

class MAEEncoder(FrozenEncoder):
    """
    Frozen MAE ViT-B/16
    """

    def __init__(self):
        super().__init__()
        self.name = "vit-mae-base"
        self.z_dim = ENCODER_Z_DIMS[self.name]

        model, info = ViTForImageClassification.from_pretrained(
            HF_ENCODER_IDS[self.name], output_loading_info = True
        )

        self.backbone = model.vit
        self._freeze()

    def _extract(self, x):
        out = self.backbone(x, output_hidden_states = True)
        return out.hidden_states[-1][:, 0, :]




