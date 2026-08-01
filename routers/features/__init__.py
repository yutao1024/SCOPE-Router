"""
Feature extraction module - image and text embedding extraction.
Decoupled from router training; supports multiple encoder choices.

Example usage:
    from routers.features import TextEncoder, VisionEncoder, FeatureExtractor
    
    # Extract text features
    text_encoder = TextEncoder(model_name="BAAI/bge-m3")
    text_embeddings = text_encoder.extract(texts)
    
    # Extract vision features
    vision_encoder = VisionEncoder(model_name="openai/clip-vit-base-patch32")
    vision_embeddings = vision_encoder.extract(image_paths)
    
    # Unified extraction interface
    extractor = FeatureExtractor(
        text_encoder="BAAI/bge-m3",
        vision_encoder="openai/clip-vit-base-patch32"
    )
    features_df = extractor.extract_from_samples(samples)
"""

from routers.features.text_encoder import TextEncoder
from routers.features.vision_encoder import VisionEncoder
from routers.features.extractor import FeatureExtractor, extract_all_features

# Note: FeatureFusion has been moved into router implementations, but fusion.py is kept for reference.
# from routers.features.fusion import FeatureFusion, AVAILABLE_FUSION_METHODS
from routers.features.encoders_registry import (
    AVAILABLE_TEXT_ENCODERS,
    AVAILABLE_VISION_ENCODERS,
    get_text_encoder_dimension,
    get_vision_encoder_dimension,
    list_available_encoders
)

__all__ = [
    'TextEncoder',
    'VisionEncoder',
    'FeatureExtractor',
    'extract_all_features',
    'AVAILABLE_TEXT_ENCODERS',
    'AVAILABLE_VISION_ENCODERS',
    'get_text_encoder_dimension',
    'get_vision_encoder_dimension',
    'list_available_encoders',
]

