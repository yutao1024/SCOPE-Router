#!/usr/bin/env python3
"""
Unified feature extractor - wraps text and vision encoders and provides a single interface.

Fully decoupled from router training; supports:
1. Ablations with different text/vision encoders
2. Batched extraction (and optional caching)
3. Saving text and vision features separately (fusion happens during router training)

Note: feature fusion has been moved into router implementations, as it may require learnable parameters.
This module only pre-extracts and stores embeddings.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

from routers.features.text_encoder import TextEncoder
from routers.features.vision_encoder import VisionEncoder
from routers.features.encoders_registry import (
    get_text_encoder_dimension,
    get_vision_encoder_dimension
)


class FeatureExtractor:
    """
    Unified feature extractor.

    Wraps a text encoder and a vision encoder to extract and save features separately.
    Feature fusion is handled in router implementations (may involve learnable parameters).
    """
    
    def __init__(
        self,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True
    ):
        """
        Args:
            text_encoder: Text encoder name
            vision_encoder: Vision encoder name
            device: Device ("cuda" or "cpu")
            batch_size: Batch size
            normalize: Whether to normalize embeddings
        """
        # Initialize encoders
        self.text_encoder = TextEncoder(
            model_name=text_encoder,
            device=device,
            batch_size=batch_size,
            normalize=normalize
        )
        
        self.vision_encoder = VisionEncoder(
            model_name=vision_encoder,
            device=device,
            batch_size=batch_size,
            normalize=normalize
        )
        
        # Metadata
        self.text_encoder_name = text_encoder
        self.vision_encoder_name = vision_encoder
        self.device = device or self.text_encoder.device
    
    def extract_text(self, texts: List[str]) -> np.ndarray:
        """Extract text features."""
        return self.text_encoder.extract(texts)
    
    def extract_vision(self, images: List[Any]) -> np.ndarray:
        """Extract vision features."""
        return self.vision_encoder.extract(images)
    
    def extract_from_samples(
        self,
        samples: List[Dict[str, Any]],
        text_key: str = "prompt",
        asset_key: str = "assets",
        vision_pooling: str = "mean",
        vision_batch_size: int = 32
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract text and vision features separately from a list of samples.
        
        Args:
            samples: [{'sample_id': str, 'prompt': str, 'assets': [...], ...}, ...]
            text_key: Text field name
            asset_key: Asset field name
            vision_pooling: Multi-image pooling strategy
        
        Returns:
            Dict with keys: 'text_df', 'vision_df'
            Each DataFrame contains: ['sample_id', 'embedding']
        """
        sample_ids = [s['sample_id'] for s in samples]
        
        # Extract text features
        texts = []
        for sample in samples:
            # Try multiple possible text field names
            text = sample.get(text_key) or sample.get('question') or sample.get('text') or sample.get('prompt') or ''
            texts.append(text)
        
        print(f"  Extracting text features: {len(texts)} samples...")
        print(f"    Non-empty texts: {sum(1 for t in texts if t)}")
        text_embeddings = self.text_encoder.extract(texts)
        
        text_df = pd.DataFrame({
            'sample_id': sample_ids,
            'embedding': list(text_embeddings)
        })
        
        # Extract vision features
        print(f"  Extracting vision features: {len(samples)} samples...")
        vision_embeddings = self.vision_encoder.extract_from_samples(
            samples,
            asset_key=asset_key,
            pooling=vision_pooling,
            batch_size=vision_batch_size
        )
        
        vision_df = pd.DataFrame({
            'sample_id': sample_ids,
            'embedding': list(vision_embeddings)
        })
        
        return {
            'text_df': text_df,
            'vision_df': vision_df
        }
    
    def save_features(
        self,
        text_df: pd.DataFrame,
        vision_df: pd.DataFrame,
        output_dir: Path,
        prefix: Optional[str] = None
    ):
        """
        Save text and vision features to disk (separately).
        
        Args:
            text_df: Text feature DataFrame (columns: ['sample_id', 'embedding'])
            vision_df: Vision feature DataFrame (columns: ['sample_id', 'embedding'])
            output_dir: Output directory
            prefix: Optional filename prefix
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Text feature output directory
        text_dir = output_dir / "text"
        text_dir.mkdir(parents=True, exist_ok=True)
        
        # Vision feature output directory
        vision_dir = output_dir / "vision"
        vision_dir.mkdir(parents=True, exist_ok=True)
        
        # Build filenames
        if prefix is None:
            text_name = self.text_encoder_name.split('/')[-1]
            vision_name = self.vision_encoder_name.split('/')[-1]
        else:
            text_name = self.text_encoder_name.split('/')[-1]
            vision_name = self.vision_encoder_name.split('/')[-1]
        
        text_file = text_dir / f"{text_name}.parquet"
        vision_file = vision_dir / f"{vision_name}.parquet"
        
        # Save features
        text_df.to_parquet(text_file, index=False)
        vision_df.to_parquet(vision_file, index=False)
        
        # Save metadata
        text_metadata = {
            "encoder": self.text_encoder_name,
            "dimension": self.text_encoder.dimension,
            "num_samples": len(text_df),
            "device": self.device
        }
        
        vision_metadata = {
            "encoder": self.vision_encoder_name,
            "dimension": self.vision_encoder.dimension,
            "num_samples": len(vision_df),
            "device": self.device
        }
        
        text_metadata_file = text_file.with_suffix('.json')
        vision_metadata_file = vision_file.with_suffix('.json')
        
        with open(text_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(text_metadata, f, indent=2, ensure_ascii=False)
        
        with open(vision_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(vision_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ Text features saved: {text_file}")
        print(f"     Dim: {self.text_encoder.dimension}, samples: {len(text_df)}")
        print(f"  ✓ Vision features saved: {vision_file}")
        print(f"     Dim: {self.vision_encoder.dimension}, samples: {len(vision_df)}")
    
    @classmethod
    def load_text_features(cls, feature_file: Path) -> tuple:
        """
        Load text features from disk.
        
        Args:
            feature_file: Path to text feature parquet file
        
        Returns:
            (features_df, metadata) tuple
        """
        feature_file = Path(feature_file)
        
        # Load features
        features_df = pd.read_parquet(feature_file)
        
        # Load metadata
        metadata_file = feature_file.with_suffix('.json')
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return features_df, metadata
    
    @classmethod
    def load_vision_features(cls, feature_file: Path) -> tuple:
        """
        Load vision features from disk.
        
        Args:
            feature_file: Path to vision feature parquet file
        
        Returns:
            (features_df, metadata) tuple
        """
        feature_file = Path(feature_file)
        
        # Load features
        features_df = pd.read_parquet(feature_file)
        
        # Load metadata
        metadata_file = feature_file.with_suffix('.json')
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        return features_df, metadata
    
    def __repr__(self):
        return (
            f"FeatureExtractor(\n"
            f"  text={self.text_encoder_name} (dim={self.text_encoder.dimension}),\n"
            f"  vision={self.vision_encoder_name} (dim={self.vision_encoder.dimension}),\n"
            f"  device={self.device}\n"
            f")"
        )


def extract_all_features(
    dataset_dir: Path,
    samples: List[Dict[str, Any]],
    text_encoder: str = "BAAI/bge-m3",
    vision_encoder: str = "facebook/dinov2-base",
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    vision_batch_size: int = 32
) -> Dict[str, pd.DataFrame]:
    """
    Convenience helper: extract text and vision features for all samples and save them.
    
    Args:
        dataset_dir: Dataset root directory
        samples: Sample list
        text_encoder: Text encoder
        vision_encoder: Vision encoder
        output_dir: Output dir (default: dataset_dir/EMBEDDINGS)
        device: Device
        vision_batch_size: Vision feature batch size
    
    Returns:
        Dict with keys: 'text_df', 'vision_df'
    """
    if output_dir is None:
        output_dir = dataset_dir / "EMBEDDINGS"
    
    # Create extractor
    extractor = FeatureExtractor(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        device=device
    )
    
    # Extract features
    print("\n📊 Extracting features...")
    print(f"  Text encoder: {text_encoder}")
    print(f"  Vision encoder: {vision_encoder}")
    print(f"  Samples: {len(samples)}")
    print(f"  Vision batch size: {vision_batch_size}")
    
    result = extractor.extract_from_samples(samples, vision_batch_size=vision_batch_size)
    
    # Save
    extractor.save_features(result['text_df'], result['vision_df'], output_dir)
    
    return result

