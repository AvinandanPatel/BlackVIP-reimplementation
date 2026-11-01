"""
Configuration for the BlackVIP Coordinator.

Pure Python (no torch import) so it can be validated and serialized anywhere.
Every Coordinator component is built from a `CoordinatorConfig`.
"""

from dataclasses import asdict, dataclass, fields
from typing import Any

from blackvip.constants import DEFAULT_SRC_DIM, ENCODER_Z_DIMS, SEED_AREA

# Encoders known to the Coordinator: the pre-trained HF ones + a trainable scratch CNN.
VALID_ENCODERS = tuple(ENCODER_Z_DIMS) + ("scratch",)
VALID_ACTS = ("gelu", "relu")
# "train": decoder BN uses batch stats and updates running stats on every query (reference).
# "eval" : decoder BN always uses running stats (ablation).
VALID_BN_MODES = ("train", "eval")


@dataclass
class CoordinatorConfig:
    encoder: str = "vit-mae-base"            # one of VALID_ENCODERS
    src_dim: int = DEFAULT_SRC_DIM           # len(concat(z, trigger)); must be divisible by 49
    scratch_out_dim: int = 0                 # only for encoder="scratch" (64 / 128 / 256)
    renormalize_input: bool = False          # CLIP-norm -> ImageNet-norm before the encoder (ablation)
    trigger_init: tuple[float, float] = (0.0, 0.1)  # uniform init range of the prompt trigger
    act: str = "gelu"                        # decoder activation
    p_eps: float = 1.0                       # prompt intensity epsilon (paper: in [0, 1])
    bn_mode: str = "train"                   # one of VALID_BN_MODES
    device: str = "cuda"

    # -- derived values ------------------------------------------------------
    @property
    def z_dim(self) -> int:
        """Encoder output size. 0 for scratch (it outputs a 3-D map, no trigger)."""
        return 0 if self.encoder == "scratch" else ENCODER_Z_DIMS[self.encoder]

    @property
    def trigger_dim(self) -> int:
        """Length of the learnable prompt trigger vector (800 by default)."""
        return 0 if self.encoder == "scratch" else self.src_dim - self.z_dim

    @property
    def seed_channels(self) -> int:
        """Channels of the (C, 7, 7) map fed to the decoder (32 by default)."""
        return self.scratch_out_dim if self.encoder == "scratch" else self.src_dim // SEED_AREA

    # -- validation ----------------------------------------------------------
    def validate(self) -> "CoordinatorConfig":
        """Raise ValueError on an invalid combination. Returns self for chaining."""
        if self.encoder not in VALID_ENCODERS:
            raise ValueError(f"Unknown encoder '{self.encoder}'. Valid: {VALID_ENCODERS}")

        if self.encoder == "scratch":
            if self.scratch_out_dim <= 0:
                raise ValueError("encoder='scratch' requires scratch_out_dim > 0 (e.g. 64/128/256)")
        else:
            if self.src_dim % SEED_AREA != 0:
                raise ValueError(
                    f"src_dim={self.src_dim} must be divisible by {SEED_AREA} "
                    f"(it is reshaped to (C, 7, 7))"
                )
            if self.src_dim <= self.z_dim:
                raise ValueError(
                    f"src_dim={self.src_dim} must be > z_dim={self.z_dim} of '{self.encoder}' "
                    f"to leave room for the prompt trigger"
                )

        low, high = self.trigger_init
        if not low < high:
            raise ValueError(f"trigger_init must satisfy low < high, got {self.trigger_init}")
        if self.act not in VALID_ACTS:
            raise ValueError(f"Unknown act '{self.act}'. Valid: {VALID_ACTS}")
        if not 0.0 <= self.p_eps <= 1.0:
            raise ValueError(f"p_eps must be in [0, 1], got {self.p_eps}")
        if self.bn_mode not in VALID_BN_MODES:
            raise ValueError(f"Unknown bn_mode '{self.bn_mode}'. Valid: {VALID_BN_MODES}")
        return self

    # -- serialization (for checkpoints / logs) ------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trigger_init"] = list(self.trigger_init)  # JSON/YAML friendly
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CoordinatorConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown CoordinatorConfig keys: {sorted(unknown)}")
        d = dict(d)
        if "trigger_init" in d:
            d["trigger_init"] = tuple(d["trigger_init"])  # lists come back from JSON/YAML
        return cls(**d)
