#!/usr/bin/env python3
"""
Vision encoder - supports multiple image embedding models

Supported encoders:
- CLIP family (openai/clip-*)
- SigLIP family (google/siglip-*)
- DINOv2 family (facebook/dinov2-*)
- ConvNeXt family (facebook/convnext-*)

Supported image sources:
- Local file paths
- PIL Image objects
- Base64-encoded images stored in TSV files
"""

import warnings
from typing import List, Optional, Union, Any, Dict
from pathlib import Path
import numpy as np
import base64
from io import BytesIO

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    from transformers import AutoModel, AutoProcessor
    try:
        from PIL import Image
        HAS_PIL = True
    except ImportError:
        Image = None
        HAS_PIL = False
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    Image = None
    HAS_PIL = False
    warnings.warn("transformers/PIL not installed; vision encoder is unavailable")

from routers.features.encoders_registry import (
    AVAILABLE_VISION_ENCODERS,
    get_vision_encoder_dimension
)


class VisionEncoder:
    """
    Vision encoder - a unified interface for image embedding extraction.

    Supports multiple encoder backends and handles different model architectures automatically.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 16,
        normalize: bool = True
    ):
        """
        Args:
            model_name: Encoder model name (see AVAILABLE_VISION_ENCODERS)
            device: Device ("cuda" or "cpu"); auto-select if None
            batch_size: Batch size for batched extraction
            normalize: Whether to apply L2 normalization
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        
        # Validate model
        if model_name not in AVAILABLE_VISION_ENCODERS:
            raise ValueError(
                f"Unknown vision encoder: {model_name}\n"
                f"Available encoders: {list(AVAILABLE_VISION_ENCODERS.keys())[:5]}..."
            )
        
        self.model_info = AVAILABLE_VISION_ENCODERS[model_name]
        
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
        self.processor = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the encoder model."""
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Please install: pip install transformers pillow torch\n"
                f"Model {self.model_name} requires the transformers library"
            )
        
        print(f"📥 Loading vision encoder: {self.model_name}")
        
        # Choose loading method based on model type
        if "clip" in self.model_name.lower():
            # CLIP model
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            
        elif "siglip" in self.model_name.lower():
            # SigLIP model (CLIP-like)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model.eval()
            
        elif "dinov2" in self.model_name.lower():
            # DINOv2 model
            self.model = AutoModel.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model.eval()
            
        elif "convnext" in self.model_name.lower():
            # ConvNeXt model
            from transformers import ConvNextImageProcessor, ConvNextModel
            self.model = ConvNextModel.from_pretrained(self.model_name)
            self.processor = ConvNextImageProcessor.from_pretrained(self.model_name)
            self.model.eval()
            
        else:
            # Generic AutoModel loading
            self.model = AutoModel.from_pretrained(self.model_name)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model.eval()
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print(f"  ✓ Device: {self.device}, output dim: {self.model_info['dimension']}")
    
    @property
    def dimension(self) -> int:
        """Return the encoder output dimension."""
        return self.model_info["dimension"]

    def _features_from_outputs(self, outputs) -> Any:
        """Normalize model outputs from CLIP/SigLIP/DINO-style models to a feature tensor."""
        if hasattr(outputs, "cpu"):
            return outputs
        if hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
            return outputs.image_embeds
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            return outputs.pooler_output
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state[:, 0, :]
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return self._features_from_outputs(outputs[0])
        raise ValueError(f"Failed to extract tensor features from output type {type(outputs).__name__}")

    def _forward_features(self, inputs: Dict[str, Any]) -> Any:
        """Run the model and return an image feature tensor across transformers variants."""
        with torch.no_grad():
            if ("clip" in self.model_name.lower() or "siglip" in self.model_name.lower()) and hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
                return self._features_from_outputs(features)

            outputs = self.model(**inputs)
            return self._features_from_outputs(outputs)
    
    def _load_image_from_tsv(self, asset: Dict) -> Any:
        """
        Load a base64-encoded image from a TSV file (fully compatible with VLMEvalKit).
        
        Args:
            asset: {'type': 'image_tsv', 'tsv_file': str, 'index': int}
        
        Returns:
            PIL Image object
        """
        import pandas as pd
        
        tsv_file = asset['tsv_file']
        index = asset['index']
        
        # VLMEvalKit approach: load full TSV then access rows via iloc
        df = pd.read_csv(tsv_file, sep='\t')
        
        if index >= len(df):
            raise IndexError(f"index {index} is out of range for TSV rows: {len(df)}")
        
        if 'image' not in df.columns:
            raise ValueError(f"TSV file {tsv_file} does not contain an 'image' column")
        
        # VLMEvalKit compatibility: resolve references in the image column (space-saving mechanism)
        # The image column can be base64 image data or a reference to another sample's index
        if 'index' in df.columns:
            df['image'] = df['image'].astype(str)
            image_map = {str(idx): img for idx, img in zip(df['index'], df['image'])}
            
            # Resolve references: if image data length <= 64, treat it as a reference index
            for k in list(image_map.keys()):
                if len(image_map[k]) <= 64:
                    ref_idx = image_map[k]  # referenced index
                    if ref_idx in image_map and len(image_map[ref_idx]) > 64:
                        image_map[k] = image_map[ref_idx]  # replace with real base64
            
            # Use resolved image data
            row = df.iloc[index]
            row_index = str(row['index'])
            img_base64 = image_map.get(row_index, row['image'])
        else:
            # No index column; use image data directly
            row = df.iloc[index]
            img_base64 = row['image']
        
        # Decode base64 image
        img_data = base64.b64decode(img_base64)
        image = Image.open(BytesIO(img_data)).convert('RGB')
        
        return image
    
    def _extract_from_pil_image(self, image: Any) -> np.ndarray:
        """Extract features from a PIL Image object."""
        # Preprocess image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        features = self._forward_features(inputs)
        
        # Convert to numpy and normalize
        embedding = features.cpu().numpy()[0]
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def _extract_single(self, image_path: Union[str, Path, Any]) -> np.ndarray:
        """Extract an embedding for a single image."""
        # Load image
        if isinstance(image_path, (str, Path)):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                raise ValueError(f"Failed to load image {image_path}: {e}")
        elif Image is not None and isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_path)}")
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move to device
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        features = self._forward_features(inputs)
        features = features.cpu().numpy()[0]
        
        # Normalize
        if self.normalize:
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
        
        return features
    
    def extract(self, images: Union[str, Path, Any, List[Union[str, Path, Any]]]) -> np.ndarray:
        """
        Extract image embeddings.
        
        Args:
            images: A single image (path/PIL Image) or a list of images
        
        Returns:
            embeddings: (N, D) numpy array, where N=1 or len(images)
        """
        if not isinstance(images, list):
            images = [images]
        
        embeddings = []
        for image in images:
            emb = self._extract_single(image)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def _extract_batch_from_images(self, images: List[Any], batch_size: int = 32) -> np.ndarray:
        """
        Batch-extract image features (improves GPU utilization).
        
        Args:
            images: List of PIL Image objects
            batch_size: Batch size
        
        Returns:
            embeddings: (N, D) numpy array
        """
        from tqdm import tqdm
        
        if len(images) == 0:
            return np.array([])
        
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(images) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(images), batch_size), total=num_batches, desc="      batched inference", unit="batch"):
            batch_images = images[i:i + batch_size]
            
            # Batch preprocessing
            inputs = self.processor(images=batch_images, return_tensors="pt")
            
            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            features = self._forward_features(inputs)
            
            # Convert to numpy and release GPU memory
            batch_embeddings = features.cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
            # Cleanup intermediates and GPU cache (avoid memory accumulation)
            del inputs, features, batch_embeddings
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
        
        # Concatenate all batches
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Normalize
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)  # avoid division by zero
            embeddings = embeddings / norms
        
        return embeddings
    
    def extract_from_samples(
        self,
        samples: List[dict],
        asset_key: str = "assets",
        pooling: str = "mean",
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Extract vision features from a list of samples (memory-optimized batch-storage version).
        
        Strategy: process TSV-by-TSV → run inference → batch-store TSV-level features → merge at the end.
        
        Args:
            samples: [{'sample_id': str, 'assets': [...], ...}, ...]
            asset_key: Asset field name (default: 'assets')
            pooling: Multi-image pooling strategy ("mean", "max", "first")
            batch_size: Batch size (default: 32)
        
        Returns:
            embeddings: (N, D) numpy array
        """
        from tqdm import tqdm
        import pandas as pd
        from collections import defaultdict
        import gc
        import tempfile
        import os
        import psutil
        
        def get_memory_mb():
            """Get current process memory usage (MB)."""
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        
        print(f"    Optimized batch-storage strategy (batch_size={batch_size}, avoid RAM blowup)...")
        print(f"    Initial memory: {get_memory_mb():.1f} MB")
        
        # Create a temporary directory to store intermediate features
        temp_dir = tempfile.mkdtemp(prefix="vision_features_")
        print(f"    Temp dir: {temp_dir}")
        
        # Group samples by TSV file
        tsv_to_samples = defaultdict(list)  # {tsv_file: [(sample_idx, asset), ...]}
        non_tsv_samples = []  # [(sample_idx, asset), ...]
        
        for sample_idx, sample in enumerate(samples):
            assets = sample.get(asset_key, [])
            for asset in assets:
                if isinstance(asset, dict) and asset.get('type') == 'image_tsv':
                    tsv_file = asset['tsv_file']
                    tsv_to_samples[tsv_file].append((sample_idx, asset))
                else:
                    non_tsv_samples.append((sample_idx, asset))
        
        # New strategy: buffer features in-memory and periodically persist TSV-level batches
        # Avoid frequent load/save of many small files
        sample_features_dict = {}  # {sample_idx: [features...]}
        tsv_processed = 0
        
        # Key optimization: process TSV-by-TSV (load → infer → buffer → periodically persist)
        for tsv_idx, (tsv_file, tsv_samples) in enumerate(tsv_to_samples.items()):
            mem_before = get_memory_mb()
            tsv_name = Path(tsv_file).name
            print(f"\n    [{tsv_idx+1}/{len(tsv_to_samples)}] Processing {tsv_name}")
            print(f"      Memory (before): {mem_before:.1f} MB")
            
            try:
                # 1. Load TSV file
                df_tsv = pd.read_csv(tsv_file, sep='\t')
                mem_after_tsv = get_memory_mb()
                print(f"      After TSV load: {mem_after_tsv:.1f} MB (+{mem_after_tsv-mem_before:.1f})")
                
                # VLMEvalKit compatibility: resolve references in the image column
                if 'image' in df_tsv.columns and 'index' in df_tsv.columns:
                    df_tsv['image'] = df_tsv['image'].astype(str)
                    image_map = {str(idx): img for idx, img in zip(df_tsv['index'], df_tsv['image'])}
                    
                    for k in list(image_map.keys()):
                        if len(image_map[k]) <= 64:
                            ref_idx = image_map[k]
                            if ref_idx in image_map and len(image_map[ref_idx]) > 64:
                                image_map[k] = image_map[ref_idx]
                    
                    for idx_val, img_data in image_map.items():
                        mask = df_tsv['index'].astype(str) == idx_val
                        df_tsv.loc[mask, 'image'] = img_data
                
                # 2. Load+infer in chunks (avoid loading too many images at once)
                # Key optimization: load only CHUNK_SIZE images per chunk
                CHUNK_SIZE = 1000  # images per chunk
                tsv_sample_list = list(tsv_samples)
                total_images = len(tsv_sample_list)
                print(f"      Total {total_images} samples, {(total_images + CHUNK_SIZE - 1) // CHUNK_SIZE} chunks")
                
                for chunk_start in range(0, total_images, CHUNK_SIZE):
                    chunk_end = min(chunk_start + CHUNK_SIZE, total_images)
                    chunk_samples = tsv_sample_list[chunk_start:chunk_end]
                    
                    # Load images for this chunk
                    chunk_images = []
                    chunk_sample_map = []  # [(sample_idx, img_idx_in_chunk), ...]
                    
                    for sample_idx, asset in chunk_samples:
                        try:
                            index = asset['index']
                            if index < len(df_tsv):
                                row = df_tsv.iloc[index]
                                img_base64 = row['image']
                                img_data = base64.b64decode(img_base64)
                                image = Image.open(BytesIO(img_data)).convert('RGB')
                                
                                chunk_sample_map.append((sample_idx, len(chunk_images)))
                                chunk_images.append(image)
                        except Exception:
                            pass
                    
                    if len(chunk_images) == 0:
                        continue
                    
                    # Run inference for this chunk
                    chunk_features = self._extract_batch_from_images(chunk_images, batch_size)
                    
                    # Move to CPU immediately and free GPU memory
                    if hasattr(chunk_features, 'cpu'):
                        chunk_features = chunk_features.cpu().numpy()
                    
                    # Accumulate features
                    for sample_idx, img_idx in chunk_sample_map:
                        feature = chunk_features[img_idx]
                        if sample_idx not in sample_features_dict:
                            sample_features_dict[sample_idx] = []
                        sample_features_dict[sample_idx].append(feature)
                    
                    # Release chunk resources immediately
                    del chunk_images, chunk_features, chunk_sample_map
                    gc.collect()
                    
                    if self.device.startswith('cuda'):
                        import torch
                        torch.cuda.empty_cache()
                
                mem_after_infer = get_memory_mb()
                print(f"      Inference done: {mem_after_infer:.1f} MB (+{mem_after_infer-mem_after_tsv:.1f})")
                
                # 3. Batch persist: save every 2 TSVs or when reaching 5000 samples
                tsv_processed += 1
                if tsv_processed % 2 == 0 or len(sample_features_dict) > 5000:
                    print(f"      Persisting features for {len(sample_features_dict)} samples...")
                    
                    # Save to a temporary file
                    batch_file = os.path.join(temp_dir, f"batch_{tsv_idx}.npz")
                    save_dict = {}
                    for sid, feats in sample_features_dict.items():
                        if len(feats) == 1:
                            save_dict[f"s{sid}"] = feats[0]
                        else:
                            save_dict[f"s{sid}"] = np.stack(feats)
                    
                    np.savez_compressed(batch_file, **save_dict)
                    
                    # Clear the buffer
                    sample_features_dict.clear()
                    
                    mem_after_save = get_memory_mb()
                    print(f"      Memory after save: {mem_after_save:.1f} MB")
                
                # 4. Release TSV DataFrame and other resources immediately
                del df_tsv
                
                # Force garbage collection (repeat to ensure cleanup)
                for _ in range(3):
                    gc.collect()
                
                # Clear GPU cache
                if self.device.startswith('cuda'):
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # ensure GPU ops finished
                
                mem_after_clean = get_memory_mb()
                print(f"      Memory after cleanup: {mem_after_clean:.1f} MB (freed {mem_after_infer-mem_after_clean:.1f})")
                    
            except Exception as e:
                warnings.warn(f"Failed to process TSV file {tsv_file}: {e}")
        
        # Persist remaining features
        if len(sample_features_dict) > 0:
            print(f"\n    Persisting remaining features for {len(sample_features_dict)} samples...")
            batch_file = os.path.join(temp_dir, f"batch_final.npz")
            save_dict = {}
            for sid, feats in sample_features_dict.items():
                if len(feats) == 1:
                    save_dict[f"s{sid}"] = feats[0]
                else:
                    save_dict[f"s{sid}"] = np.stack(feats)
            np.savez_compressed(batch_file, **save_dict)
            sample_features_dict.clear()
        
        # Handle non-TSV samples (regular image files) - currently skipped, focusing on TSV
        # if len(non_tsv_samples) > 0:
        #     ... (can be added later)
        
        # Finally: load from batch files and apply pooling
        print("\n    Loading from batch files and applying pooling...")
        print(f"    Memory (before load): {get_memory_mb():.1f} MB")
        
        # Collect data from all batch files first
        all_features = {}  # {sample_idx: [features...]}
        batch_files = sorted([f for f in os.listdir(temp_dir) if f.startswith('batch_')])
        
        print(f"    Loading {len(batch_files)} batch files...")
        for batch_file in batch_files:
            batch_path = os.path.join(temp_dir, batch_file)
            data = np.load(batch_path)
            
            for key in data.keys():
                sample_idx = int(key[1:])  # strip leading 's'
                features = data[key]
                
                if sample_idx not in all_features:
                    all_features[sample_idx] = []
                
                if features.ndim == 1:
                    all_features[sample_idx].append(features)
                else:
                    # Already stacked features
                    all_features[sample_idx].extend(list(features))
            
            data.close()
            del data
        
        mem_after_load = get_memory_mb()
        print(f"    Load complete: {mem_after_load:.1f} MB")
        
        # Apply pooling
        embeddings = []
        for sample_idx in range(len(samples)):
            if sample_idx not in all_features or len(all_features[sample_idx]) == 0:
                # No features -> use zero vector
                embedding = np.zeros(self.dimension)
            else:
                features_list = all_features[sample_idx]
                
                if len(features_list) == 1:
                    # Single feature
                    embedding = features_list[0]
                else:
                    # Multiple features -> apply pooling
                    features = np.stack(features_list)
                    if pooling == "mean":
                        embedding = np.mean(features, axis=0)
                    elif pooling == "max":
                        embedding = np.max(features, axis=0)
                    elif pooling == "first":
                        embedding = features[0]
                    else:
                        embedding = np.mean(features, axis=0)
                    
                    # Re-normalize after pooling
                    if self.normalize:
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
            
            embeddings.append(embedding)
        
        print(f"    Pooling complete: {get_memory_mb():.1f} MB")
        
        # Cleanup temporary files
        print("    Cleaning up temporary files...")
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            warnings.warn(f"Failed to clean temporary directory: {e}")
        
        print(f"    Final memory: {get_memory_mb():.1f} MB")
        return np.array(embeddings)
    
    def __repr__(self):
        return f"VisionEncoder(model='{self.model_name}', dim={self.dimension}, device='{self.device}')"


# Export available encoder list
AVAILABLE_VISION_ENCODERS = AVAILABLE_VISION_ENCODERS  # imported from encoders_registry
