"""
Global constants for the BlackVIP implementation.
"""

# CLIP image normalization
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Valid pixel range [0, 1] expressed in CLIP-normalized space per channel
CLIP_PIXEL_MIN = tuple((0.0 - m) / s for m, s in zip(CLIP_MEAN, CLIP_STD))
CLIP_PIXEL_MAX = tuple((1.0 - m) / s for m, s in zip(CLIP_MEAN, CLIP_STD))

# ImageNet normalization (what HF ViT/MAE image processors expect).
# Only used by the optional `renormalize_input` flag of the encoders.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Image / decoder geometry ---
IMAGE_SIZE = 224
NUM_CHANNELS: int = 3
SEED_HW = 7 # Decoder starts from 7 x 7
SEED_AREA = SEED_HW * SEED_HW # 49; src_dim must be divisible by this
NUM_UPSAMPLE_STAGES = 5 # 7 -> 14 -> 28 -> 56 -> 112 -> 224

# --- Coordinator encoder checkpoints (Hugging Face) ---
HF_ENCODER_IDS: dict[str, str] = {
    "vit-mae-base": "facebook/vit-mae-base",               # default, SSL (MAE)
    "vit-base": "google/vit-base-patch16-224-in21k",       # supervised (Table 6)
    "dino-resnet-50": "Ramos-Ramos/dino-resnet-50",        # SSL CNN (Table 5)
}
ENCODER_Z_DIMS: dict[str, int] = {
    "vit-mae-base": 768,
    "vit-base": 768,
    "dino-resnet-50": 2048,
}

# --- Coordinator defaults (paper Supp. A.4) ---
DEFAULT_SRC_DIM: int = 1568    # 768 (z) + 800 (prompt trigger) = 32 * 7 * 7
