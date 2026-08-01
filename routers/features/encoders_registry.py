#!/usr/bin/env python3
"""
Encoder registry - records available text/vision encoders and their output dimensions.

Used for:
1. Quick lookup of encoder output dimensions
2. Validating encoder compatibility
3. Generating ablation experiment configurations
"""

# ==================== Text encoders ====================

AVAILABLE_TEXT_ENCODERS = {
    "intfloat/e5-small-v2": {
        "dimension": 384,
        "type": "dense",
        "description": "E5 small v2 model",
        "max_length": 512,
    },
    "intfloat/e5-base-v2": {
        "dimension": 768,
        "type": "dense",
        "description": "E5 base v2 model",
        "max_length": 512,
    },
    "intfloat/e5-large-v2": {
        "dimension": 1024,
        "type": "dense",
        "description": "E5 large v2 model",
        "max_length": 512,
    },
    "BAAI/bge-m3": {
        "dimension": 1024,  # Note: bge-m3 supports multi-vector retrieval; main vector dim is 1024
        "type": "dense",
        "description": "BGE-M3 multilingual dense model",
        "max_length": 8192,
    },
    "thenlper/gte-large": {
        "dimension": 1024,
        "type": "dense",
        "description": "GTE large model",
        "max_length": 8192,
    },
}

# ==================== Vision encoders ====================

AVAILABLE_VISION_ENCODERS = {
    "facebook/dinov2-small": {
        "dimension": 384,
        "type": "vision_transformer",
        "description": "DINOv2 small",
        "image_size": 224,
        "patch_size": 14,
    },
    "facebook/dinov2-base": {
        "dimension": 768,
        "type": "vision_transformer",
        "description": "DINOv2 base",
        "image_size": 224,
        "patch_size": 14,
    },
    "facebook/dinov2-large": {
        "dimension": 1024,
        "type": "vision_transformer",
        "description": "DINOv2 large",
        "image_size": 224,
        "patch_size": 14,
    },
    "openai/clip-vit-large-patch14": {
        "dimension": 768,
        "type": "vision_transformer",
        "description": "CLIP ViT-L/14",
        "image_size": 224,
        "patch_size": 14,
    },
    "google/siglip-large-patch16-384": {
        "dimension": 1024,
        "type": "vision_transformer",
        "description": "SigLIP large 384px",
        "image_size": 384,
        "patch_size": 16,
    },
}

# ==================== Utilities ====================

def get_text_encoder_dimension(model_name: str) -> int:
    """Get output dimension of a text encoder."""
    if model_name not in AVAILABLE_TEXT_ENCODERS:
        raise ValueError(
            f"Unknown text encoder: {model_name}\n"
            f"Available encoders: {list(AVAILABLE_TEXT_ENCODERS.keys())}"
        )
    return AVAILABLE_TEXT_ENCODERS[model_name]["dimension"]


def get_vision_encoder_dimension(model_name: str) -> int:
    """Get output dimension of a vision encoder."""
    if model_name not in AVAILABLE_VISION_ENCODERS:
        raise ValueError(
            f"Unknown vision encoder: {model_name}\n"
            f"Available encoders: {list(AVAILABLE_VISION_ENCODERS.keys())}"
        )
    return AVAILABLE_VISION_ENCODERS[model_name]["dimension"]


def list_available_encoders():
    """List all available encoders."""
    print("=" * 80)
    print("Available text encoders:")
    print("=" * 80)
    for name, info in AVAILABLE_TEXT_ENCODERS.items():
        print(f"  {name:50s} dim={info['dimension']:4d}  {info['description']}")
    
    print("\n" + "=" * 80)
    print("Available vision encoders:")
    print("=" * 80)
    for name, info in AVAILABLE_VISION_ENCODERS.items():
        print(f"  {name:50s} dim={info['dimension']:4d}  {info['description']}")


if __name__ == "__main__":
    list_available_encoders()
