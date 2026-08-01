#!/usr/bin/env python3
"""
Text encoder - supports multiple text embedding models

Supported encoders:
- BGE family (BAAI/bge-*)
- E5 family (intfloat/e5-*)
- Sentence-BERT family (sentence-transformers/*)
- OpenAI API (text-embedding-*)
"""

import warnings
from typing import List, Optional, Union
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn("sentence-transformers not installed; some text encoders are unavailable")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from routers.features.encoders_registry import (
    AVAILABLE_TEXT_ENCODERS,
    get_text_encoder_dimension
)


class TextEncoder:
    """
    Text encoder - a unified interface for text embedding extraction.

    Supports multiple encoder backends and handles different APIs automatically.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ):
        """
        Args:
            model_name: Encoder model name (see AVAILABLE_TEXT_ENCODERS)
            device: Device ("cuda" or "cpu"); auto-select if None
            batch_size: Batch size for batched extraction
            normalize: Whether to apply L2 normalization
            show_progress: Whether to show progress bar
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.show_progress = show_progress
        
        # Validate model
        if model_name not in AVAILABLE_TEXT_ENCODERS:
            raise ValueError(
                f"Unknown text encoder: {model_name}\n"
                f"Available encoders: {list(AVAILABLE_TEXT_ENCODERS.keys())[:5]}..."
            )
        
        self.model_info = AVAILABLE_TEXT_ENCODERS[model_name]
        self.requires_api = self.model_info.get("requires_api", False)
        
        # Select device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        
        # Initialize model
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the encoder model."""
        if self.requires_api:
            # API-based models (e.g., OpenAI) require API keys
            if "openai" in self.model_name.lower():
                if not HAS_OPENAI:
                    raise ImportError("Please install the openai package: pip install openai")
                print(f"⚠️  {self.model_name} requires an OpenAI API key; set OPENAI_API_KEY in your environment.")
                # The API key is read during extract()
            else:
                raise NotImplementedError(f"API model {self.model_name} is not implemented yet")
        else:
            # Local models (sentence-transformers)
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "Please install: pip install sentence-transformers\n"
                    f"Model {self.model_name} requires the sentence-transformers library"
                )
            
            print(f"📥 Loading text encoder: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            print(f"  ✓ Device: {self.device}, output dim: {self.model_info['dimension']}")
    
    @property
    def dimension(self) -> int:
        """Return the encoder output dimension."""
        return self.model_info["dimension"]
    
    def extract(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Extract text embeddings.
        
        Args:
            texts: A single text string or a list of texts
        
        Returns:
            embeddings: (N, D) numpy array, where N=1 or len(texts)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.requires_api:
            # API call (e.g., OpenAI)
            return self._extract_api(texts)
        else:
            # Local model
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            return embeddings
    
    def _extract_api(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings via an API (e.g., OpenAI)."""
        if "openai" in self.model_name.lower():
            # Requires OPENAI_API_KEY environment variable
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set")
            
            try:
                import openai
                client = openai.OpenAI(api_key=api_key)
                
                # Batched API calls
                embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i+self.batch_size]
                    response = client.embeddings.create(
                        model=self.model_name,
                        input=batch
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    embeddings.extend(batch_embeddings)
                
                embeddings = np.array(embeddings)
                if self.normalize:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    embeddings = embeddings / (norms + 1e-8)
                
                return embeddings
            except Exception as e:
                raise RuntimeError(f"OpenAI API call failed: {e}")
        else:
            raise NotImplementedError(f"API model {self.model_name} is not implemented yet")
    
    def extract_from_samples(self, samples: List[dict], text_key: str = "prompt") -> np.ndarray:
        """
        Extract text features from a list of samples.
        
        Args:
            samples: [{'sample_id': str, 'prompt': str, ...}, ...]
            text_key: Text field name (default: 'prompt')
        
        Returns:
            embeddings: (N, D) numpy array
        """
        texts = [sample[text_key] for sample in samples]
        return self.extract(texts)
    
    def __repr__(self):
        return f"TextEncoder(model='{self.model_name}', dim={self.dimension}, device='{self.device}')"


# Export available encoder list
AVAILABLE_TEXT_ENCODERS = AVAILABLE_TEXT_ENCODERS  # imported from encoders_registry

