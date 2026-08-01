#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cosine-LXMERT Router
 RouterDC ， LXMERT -

：
1.  E(x)  Transformer (only)  LXMERT (+)
2.  meta ()  text + image
3.  VLC Router （ text  image）
4.  RouterDC 
"""

import numpy as np
import pickle
import sys
from typing import Optional, Dict, Any, Tuple, List
from tqdm.auto import tqdm
from pathlib import Path
import warnings

# ---  sklearn ---
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

# ---  PyTorch ---
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    from transformers import (
        AutoTokenizer,
        LxmertModel, LxmertConfig,
        ViTModel, ViTConfig,
        get_linear_schedule_with_warmup
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning：PyTorch  Transformers 。Cosine-LXMERT Router 。")

from routers.common import RouterBase


# --- PyTorch  ---

if HAS_TORCH:
    # --- - (Cosine-LXMERT) ---
    class CosineCLSRDataset(Dataset):
        """
         Cosine-LXMERT -
        : images, texts (tokenized), scores (Y), cluster_label
        """
        def __init__(self, images_list: list, texts: list, S_scores: np.ndarray, 
                     cluster_labels: np.ndarray, tokenizer, max_length=128,
                     image_transform=None, image_base_dir=None):
            self.images_list = images_list  # List of PIL Images, paths, or assets
            self.texts = texts
            self.S_scores = S_scores
            self.cluster_labels = cluster_labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.image_transform = image_transform
            self.image_base_dir = Path(image_base_dir) if image_base_dir else Path('.')
            self._tsv_cache = {}
        
        def __len__(self):
            return len(self.texts)
        
        def _load_image_from_tsv(self, asset: dict) -> Image.Image:
            """TSVload（VLC）"""
            import pandas as pd
            import base64
            from io import BytesIO
            
            tsv_file = asset['tsv_file']
            index = asset['index']
            
            cache_key = tsv_file
            if cache_key not in self._tsv_cache:
                self._tsv_cache[cache_key] = pd.read_csv(tsv_file, sep='\t')
            
            df = self._tsv_cache[cache_key]
            
            if index >= len(df):
                raise IndexError(f"index {index} TSV {len(df)}")
            
            if 'image' not in df.columns:
                raise ValueError(f"TSV'image'")
            
            # NOTE: (translated from Chinese)
            if 'index' in df.columns:
                df['image'] = df['image'].astype(str)
                image_map = {str(idx): img for idx, img in zip(df['index'], df['image'])}
                
                for k in list(image_map.keys()):
                    if len(image_map[k]) <= 64:
                        ref_idx = image_map[k]
                        if ref_idx in image_map and len(image_map[ref_idx]) > 64:
                            image_map[k] = image_map[ref_idx]
                
                row = df.iloc[index]
                row_index = str(row['index'])
                img_base64 = image_map.get(row_index, row['image'])
            else:
                row = df.iloc[index]
                img_base64 = row['image']
            
            # base64
            img_data = base64.b64decode(img_base64)
            image = Image.open(BytesIO(img_data)).convert('RGB')
            return image
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            image_source = self.images_list[idx]
            
            # （VLC）
            try:
                if isinstance(image_source, Image.Image):
                    # PIL Image
                    image = image_source
                elif isinstance(image_source, str):
                    # NOTE: (translated from Chinese)
                    img_path = Path(image_source)
                    if not img_path.is_absolute():
                        img_path = self.image_base_dir / img_path
                    
                    if img_path.exists():
                        image = Image.open(img_path).convert('RGB')
                    else:
                        # Fallback: 
                        image = Image.new('RGB', (224, 224), color='black')
                        
                elif isinstance(image_source, list) and len(image_source) > 0:
                    # Assets（）
                    asset = image_source[0]
                    asset_type = asset.get('type', '')
                    
                    if asset_type == 'image_tsv':
                        image = self._load_image_from_tsv(asset)
                    elif asset_type == 'image' or not asset_type:
                        # 'path''uri'
                        img_path_str = asset.get('path') or asset.get('uri')
                        if img_path_str:
                            img_path = Path(img_path_str)
                            if not img_path.is_absolute():
                                img_path = self.image_base_dir / img_path
                            
                            if img_path.exists():
                                image = Image.open(img_path).convert('RGB')
                            else:
                                # Fallback: 
                                image = Image.new('RGB', (224, 224), color='black')
                        else:
                            image = Image.new('RGB', (224, 224), color='black')
                    else:
                        image = Image.new('RGB', (224, 224), color='black')
                else:
                    # assets：
                    image = Image.new('RGB', (224, 224), color='black')
                    
            except Exception as e:
                # fallback
                import traceback
                if self.tokenizer:  # verbose check
                    print(f"Warning: load (idx={idx}): {e}")
                    print(f"  image_source type: {type(image_source)}")
                    if isinstance(image_source, list) and len(image_source) > 0:
                        print(f"  asset: {image_source[0]}")
                image = Image.new('RGB', (224, 224), color='black')
            
            # Transform image
            if self.image_transform:
                image = self.image_transform(image)
            
            # Get labels
            scores = self.S_scores[idx]
            cluster_label = self.cluster_labels[idx]
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'image': image,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'scores': torch.tensor(scores, dtype=torch.float32),
                'cluster_label': torch.tensor(cluster_label, dtype=torch.long)
            }

    # --- - ---
    class VLPredictDataset(Dataset):
        """-（）"""
        def __init__(self, images_list, texts, tokenizer, max_length=128, 
                     image_transform=None, image_base_dir=None):
            self.images_list = images_list
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.image_transform = image_transform
            self.image_base_dir = Path(image_base_dir) if image_base_dir else Path('.')
            self._tsv_cache = {}
        
        def __len__(self):
            return len(self.texts)
        
        def _load_image_from_tsv(self, asset: dict) -> Image.Image:
            """TSVload"""
            import pandas as pd
            import base64
            from io import BytesIO
            
            tsv_file = asset['tsv_file']
            index = asset['index']
            
            cache_key = tsv_file
            if cache_key not in self._tsv_cache:
                self._tsv_cache[cache_key] = pd.read_csv(tsv_file, sep='\t')
            
            df = self._tsv_cache[cache_key]
            
            if index >= len(df):
                raise IndexError(f"index {index} TSV {len(df)}")
            
            if 'image' not in df.columns:
                raise ValueError(f"TSV'image'")
            
            # NOTE: (translated from Chinese)
            if 'index' in df.columns:
                df['image'] = df['image'].astype(str)
                image_map = {str(idx): img for idx, img in zip(df['index'], df['image'])}
                
                for k in list(image_map.keys()):
                    if len(image_map[k]) <= 64:
                        ref_idx = image_map[k]
                        if ref_idx in image_map and len(image_map[ref_idx]) > 64:
                            image_map[k] = image_map[ref_idx]
                
                row = df.iloc[index]
                row_index = str(row['index'])
                img_base64 = image_map.get(row_index, row['image'])
            else:
                row = df.iloc[index]
                img_base64 = row['image']
            
            # base64
            img_data = base64.b64decode(img_base64)
            image = Image.open(BytesIO(img_data)).convert('RGB')
            return image
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            image_source = self.images_list[idx]
            
            # NOTE: (translated from Chinese)
            try:
                if isinstance(image_source, Image.Image):
                    image = image_source
                elif isinstance(image_source, str):
                    img_path = Path(image_source)
                    if not img_path.is_absolute():
                        img_path = self.image_base_dir / img_path
                    
                    if img_path.exists():
                        image = Image.open(img_path).convert('RGB')
                    else:
                        image = Image.new('RGB', (224, 224), color='black')
                        
                elif isinstance(image_source, list) and len(image_source) > 0:
                    asset = image_source[0]
                    asset_type = asset.get('type', '')
                    
                    if asset_type == 'image_tsv':
                        image = self._load_image_from_tsv(asset)
                    elif asset_type == 'image' or not asset_type:
                        img_path_str = asset.get('path') or asset.get('uri')
                        if img_path_str:
                            img_path = Path(img_path_str)
                            if not img_path.is_absolute():
                                img_path = self.image_base_dir / img_path
                            
                            if img_path.exists():
                                image = Image.open(img_path).convert('RGB')
                            else:
                                image = Image.new('RGB', (224, 224), color='black')
                        else:
                            image = Image.new('RGB', (224, 224), color='black')
                    else:
                        image = Image.new('RGB', (224, 224), color='black')
                else:
                    image = Image.new('RGB', (224, 224), color='black')
                    
            except Exception as e:
                image = Image.new('RGB', (224, 224), color='black')
            
            # Transform image
            if self.image_transform:
                image = self.image_transform(image)
            
            # Tokenize text
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'image': image,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }

    # --- Cosine-LXMERT  ---
    class CosineCLSRModel(nn.Module):
        """
        Cosine-LXMERT Router 
        E(x) = LXMERT(image, text) 
        k_t =  LLM 
        """
        def __init__(self, num_classes: int):
            super().__init__()
            
            # Vision Encoder: ViT
            self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.vision_hidden_size = self.vision_encoder.config.hidden_size  # 768
            
            # LXMERT Model
            self.lxmert = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
            self.lxmert_hidden_size = self.lxmert.config.hidden_size  # 768
            
            # Visual projection: ViT (768) -> LXMERT visual_feat_dim (2048)
            self.visual_projection = nn.Linear(self.vision_hidden_size, 2048)
            
            #  LLM  k_t
            #  LXMERT 
            self.llm_embeddings = nn.Parameter(torch.randn(num_classes, self.lxmert_hidden_size))
            
            # Logit scale (temperature)
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        def get_llm_embeddings_norm(self):
            return nn.functional.normalize(self.llm_embeddings, p=2, dim=1)

        def get_query_embeddings(self, images: torch.Tensor, input_ids: torch.Tensor, 
                                attention_mask: torch.Tensor) -> torch.Tensor:
            """
             E(image, text)  LXMERT
            
            Args:
                images: (batch, 3, 224, 224)
                input_ids: (batch, seq_len)
                attention_mask: (batch, seq_len)
            
            Returns:
                query_embed: (batch, hidden_size) L2-normalized
            """
            batch_size = images.size(0)
            device = images.device
            
            # 1. 
            vision_outputs = self.vision_encoder(pixel_values=images)
            vision_features = vision_outputs.last_hidden_state  # (batch, num_patches, 768)
            
            # Project to LXMERT's expected dimension (2048)
            vision_features = self.visual_projection(vision_features)  # (batch, num_patches, 2048)
            
            # 2. LXMERT expects visual_pos (bounding box positions)
            # Use dummy positions for image patches
            num_patches = vision_features.size(1)
            visual_pos = torch.zeros(
                batch_size, num_patches, 4,
                dtype=torch.float,
                device=device
            )
            
            # 3. Forward through LXMERT
            lxmert_outputs = self.lxmert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                visual_feats=vision_features,
                visual_pos=visual_pos
            )
            
            # Use pooled output (cross-modal representation)
            query_embed = lxmert_outputs.pooled_output  # (batch, hidden_size)
            
            # L2 normalize
            query_embed = nn.functional.normalize(query_embed, p=2, dim=1)
            return query_embed

        def forward(self, images: torch.Tensor, input_ids: torch.Tensor, 
                   attention_mask: torch.Tensor, return_features: bool = False) -> torch.Tensor:
            """
             E(image, text)  k_t 
            
            Returns:
                logits: (batch, num_classes) 
            """
            query_embed = self.get_query_embeddings(images, input_ids, attention_mask)  # (N, D)
            llm_embed = self.get_llm_embeddings_norm()  # (K, D)
            
            # NOTE: (translated from Chinese)
            tau = torch.exp(self.logit_scale)
            logits = query_embed @ llm_embed.T * tau  # (N, K)
            
            if return_features:
                return logits, query_embed
            return logits

else:
    CosineCLSRModel = None
    CosineCLSRDataset = None
    VLPredictDataset = None


# ---  ---

class CosineCLSRRouter(RouterBase):
    """
    Cosine-LXMERT Router
     RouterDC ， LXMERT -
    
    ：
    1.  LXMERT  (image, text) -> embedding
    2.  LLM embeddings
    3. ：Sample-LLM + Sample-Sample
    4.  text  image 
    """
    def __init__(self, 
                 # --- RouterDC  ---
                 K_plus: int = 3,
                 K_minus: int = 3,
                 lambda_sample_sample: float = 1.0,
                 num_clusters: int = 5,
                 H_out_group: int = 3,
                 loss_type: str = "routerdc",
                 sample_sample_temperature: float = 0.1,
                 
                 # --- LXMERT  ---
                 max_length: int = 128,
                 
                 # --- Soft labels (cost-aware targets) ---
                 use_soft_labels: bool = False,
                 train_lambda: float = 0.0,
                 cost_scale: float = 100.0,
                 
                 # ---  ---
                 max_epochs: int = 5,
                 learning_rate: float = 2e-5,
                 batch_size: int = 16,
                 warmup_ratio: float = 0.1,
                 weight_decay: float = 0.01,
                 grad_accumulation_steps: int = 1,
                 
                 # ---  ---
                 enable_monitoring: bool = True,
                 patience: int = 2,
                 monitor_metric: str = 'rank_score',
                 
                 # ---  ---
                 device: str = 'cuda',
                 verbose: int = 1
                ):
        
        if not HAS_TORCH:
            raise ImportError("Cosine-LXMERT Router  PyTorch  Transformers。")
        if not HAS_SKLEARN:
            raise ImportError("Cosine-LXMERT Router  scikit-learn。")
            
        super().__init__()
        
        # RouterDC 
        self.K_plus = K_plus
        self.K_minus = K_minus
        self.lambda_sample_sample = lambda_sample_sample
        self.num_clusters = num_clusters
        self.H_out_group = H_out_group
        if loss_type not in {"routerdc", "siglip_relevance"}:
            raise ValueError(f"loss_type must be routerdc or siglip_relevance; got {loss_type}")
        self.loss_type = loss_type
        self.sample_sample_temperature = float(sample_sample_temperature)
        
        # LXMERT 
        self.max_length = max_length
        
        # Soft labels (cost-aware targets)
        self.use_soft_labels = use_soft_labels
        self.train_lambda = float(train_lambda)
        self.cost_scale = float(cost_scale)
        
        # NOTE: (translated from Chinese)
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.grad_accumulation_steps = grad_accumulation_steps
        
        # NOTE: (translated from Chinese)
        self.enable_monitoring = enable_monitoring
        self.patience = patience
        self.monitor_metric = monitor_metric
        # Rank score configuration (used for monitoring on dev)
        self._rank_score_cmin = None
        self._rank_score_cmax = None
        self._rank_score_beta = 0.1
        
        # NOTE: (translated from Chinese)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        # NOTE: (translated from Chinese)
        self.model = None
        self.tokenizer = None
        self.image_transform = None
        self.scaler = None
        self.kmeans = None
        self.model_mapping = None
        
        if self.verbose > 0:
            print(f"Cosine-LXMERT Router initialized (device={self.device})")
    
    def _extract_texts_and_images_from_meta(self, meta) -> Tuple[List[str], List]:
        """
        
        
        Args:
            meta: DataFrame with 'text' and 'assets' columns
        
        Returns:
            texts: List of strings
            images: List of PIL Images or paths
        """
        texts = []
        images = []
        
        for idx, row in meta.iterrows():
            # Extract text
            text = row.get('text', row.get('question', row.get('prompt', '')))
            if not text:
                text = "No text available"
            texts.append(str(text))
            
            # Extract image (first asset)
            assets = row.get('assets', [])
            if assets and len(assets) > 0:
                # Assume assets[0] is an image path or PIL Image
                image = assets[0]
                images.append(image)
            else:
                # Create a blank image if no asset
                images.append(Image.new('RGB', (224, 224), color='white'))
        
        return texts, images
    
    def fit(self, Y, C, meta, 
            Y_dev=None, C_dev=None, meta_dev=None,
            model_mapping=None, costs=None, 
            monitor_output_dir=None, **kwargs):
        """
         Cosine-LXMERT Router
        
        Args:
            Y:  (N, K)
            C:  (N, K)
            meta:  DataFrame（ 'text'  'assets' ）
            Y_dev, C_dev, meta_dev: Dev（）
            model_mapping: 
            costs: （）
            monitor_output_dir: 
        """
        if self.verbose > 0:
            print("\n" + "="*80)
            print("🚀 Cosine-LXMERT Router Training")
            print("="*80)
        
        # 1. 
        if self.verbose > 0:
            print("\n📥 ...")
        
        texts_train, images_train = self._extract_texts_and_images_from_meta(meta)
        
        if self.verbose > 0:
            print(f"  Train set: {len(texts_train)} samples")
            print(f"  samples: {texts_train[0][:100]}...")
        
        # 2. （）
        has_correct = (Y.sum(axis=1) > 0)
        valid_indices = np.where(has_correct)[0]
        
        if len(valid_indices) < len(Y):
            if self.verbose > 0:
                print(f"  Filtering training samples: {len(Y)} -> {len(valid_indices)} "
                      f"(removed {len(Y) - len(valid_indices)} no correct model)")
            Y = Y[valid_indices]
            C = C[valid_indices]
            texts_train = [texts_train[i] for i in valid_indices]
            images_train = [images_train[i] for i in valid_indices]
        
        # 3. 
        num_classes = Y.shape[1]
        
        if self.verbose > 0:
            print(f"\n🏗️   Cosine-LXMERT ...")
            print(f"  : {num_classes}")
            print(f"  : {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = CosineCLSRModel(num_classes=num_classes).to(self.device)
        
        # Image transform (ViT preprocessing)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # 4. 
        if model_mapping is None:
            model_mapping = {i: f'model_{i}' for i in range(num_classes)}
        self.model_mapping = model_mapping

        # Rank score bounds for dev monitoring (optional but recommended)
        try:
            cmin = kwargs.get("cmin", None)
            cmax = kwargs.get("cmax", None)
            beta = kwargs.get("rank_score_beta", kwargs.get("beta", 0.1))
            if cmin is None or cmax is None:
                cmin = float(np.min(C_dev)) if C_dev is not None else float(np.min(C))
                cmax = float(np.max(C_dev)) if C_dev is not None else float(np.max(C))
            self._rank_score_cmin = float(cmin)
            self._rank_score_cmax = float(cmax)
            self._rank_score_beta = float(beta)
        except Exception:
            self._rank_score_cmin = None
            self._rank_score_cmax = None
            self._rank_score_beta = 0.1
        
        # 5.  S_scores ()
        if self.verbose > 0:
            if self.use_soft_labels:
                print(f"\n📊  soft targets as S_scores (λ={self.train_lambda})...")
            else:
                print("\n📊  S_scores (uniform over correct models)...")

        if self.loss_type == "siglip_relevance":
            from routers.utils.soft_targets import build_relevance_targets
            S_scores = build_relevance_targets(
                Y, C,
                lam=self.train_lambda,
                fallback='zeros',
                cost_scale=self.cost_scale
            ).astype(np.float32)
        elif self.use_soft_labels:
            from routers.utils.soft_targets import build_soft_targets
            S_scores = build_soft_targets(
                Y, C,
                lam=self.train_lambda,
                scheme='exp',
                fallback='cheapest',
                cost_scale=self.cost_scale
            ).astype(np.float32)
        else:
            # Uniform over correct models
            S_scores = Y.astype(np.float32)
            row_sums = S_scores.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # 
            S_scores = S_scores / row_sums
        
        # 6. （ Sample-Sample ）
        if self.verbose > 0:
            print(f"\n🔍 K-Means  (k={self.num_clusters})...")
        
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        cluster_labels = self.kmeans.fit_predict(S_scores)
        
        if self.verbose > 0:
            unique, counts = np.unique(cluster_labels, return_counts=True)
            print(f"  : {dict(zip(unique, counts))}")
        
        # 7. 
        train_dataset = CosineCLSRDataset(
            images_list=images_train,
            texts=texts_train,
            S_scores=S_scores,
            cluster_labels=cluster_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            image_transform=self.image_transform,
            image_base_dir='.'  # 
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # 8.  dev （）
        dev_loader = None
        if self.enable_monitoring and Y_dev is not None and meta_dev is not None:
            if self.verbose > 0:
                print(f"\n📥  Dev ...")
            
            texts_dev, images_dev = self._extract_texts_and_images_from_meta(meta_dev)
            # IMPORTANT:
            # - Do NOT filter dev samples here.
            # - train_and_eval evaluates on the full dev split; filtering here causes
            #   dev metrics during monitoring to be inconsistent with post-training evaluation.
            # - For samples with no correct model, soft_targets has a safe fallback;
            #   evaluation accuracy uses Y_dev directly and remains well-defined (0 for those samples).
            if self.verbose > 0:
                num_no_correct = int(np.sum(Y_dev.sum(axis=1) == 0))
                if num_no_correct > 0:
                    print(f"  Dev no correct model: {num_no_correct}/{len(Y_dev)} (evaluation)")
            
            # Dev S_scores 
            if self.loss_type == "siglip_relevance":
                from routers.utils.soft_targets import build_relevance_targets
                S_scores_dev = build_relevance_targets(
                    Y_dev, C_dev,
                    lam=self.train_lambda,
                    fallback='zeros',
                    cost_scale=self.cost_scale
                ).astype(np.float32)
            elif self.use_soft_labels:
                from routers.utils.soft_targets import build_soft_targets
                S_scores_dev = build_soft_targets(
                    Y_dev, C_dev,
                    lam=self.train_lambda,
                    scheme='exp',
                    fallback='cheapest',
                    cost_scale=self.cost_scale
                ).astype(np.float32)
            else:
                S_scores_dev = Y_dev.astype(np.float32)
                row_sums_dev = S_scores_dev.sum(axis=1, keepdims=True)
                row_sums_dev[row_sums_dev == 0] = 1
                S_scores_dev = S_scores_dev / row_sums_dev
            
            cluster_labels_dev = self.kmeans.predict(S_scores_dev)
            
            dev_dataset = CosineCLSRDataset(
                images_list=images_dev,
                texts=texts_dev,
                S_scores=S_scores_dev,
                cluster_labels=cluster_labels_dev,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                image_transform=self.image_transform,
                image_base_dir='.'  # 
            )
            
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            if self.verbose > 0:
                print(f"  Dev : {len(texts_dev)} samples")
        
        # 9. 
        self._train(
            train_loader=train_loader,
            dev_loader=dev_loader,
            Y_dev=Y_dev if self.enable_monitoring else None,
            C_dev=C_dev if self.enable_monitoring else None,
            monitor_output_dir=monitor_output_dir
        )
        
        if self.verbose > 0:
            print(f"\n✅ Cosine-LXMERT Router Training complete")
    
    def _train(self, train_loader, dev_loader=None, Y_dev=None, C_dev=None, 
              monitor_output_dir=None):
        """training"""
        # NOTE: (translated from Chinese)
        num_training_steps = len(train_loader) * self.max_epochs // self.grad_accumulation_steps
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, 
                               weight_decay=self.weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # NOTE: (translated from Chinese)
        use_monitoring = (self.enable_monitoring and dev_loader is not None and 
                         Y_dev is not None and C_dev is not None)
        
        if use_monitoring:
            from routers.utils.training_monitor import TrainingMonitor
            if monitor_output_dir is None:
                import tempfile
                monitor_output_dir = Path(tempfile.mkdtemp(prefix='cosinecls_'))
            
            monitor = TrainingMonitor(
                output_dir=monitor_output_dir,
                metric=self.monitor_metric,
                patience=self.patience,
                maximize=True if self.monitor_metric in ['accuracy', 'rank_score'] else False,
                save_checkpoints=True,
                verbose=self.verbose
            )
            monitor.start_training()
        else:
            monitor = None
        
        # NOTE: (translated from Chinese)
        if self.verbose > 0:
            print(f"\n🎓 training...")
            print(f"  samples: {len(train_loader.dataset)}, : {len(train_loader)}")
            print(f"  Max epochs: {self.max_epochs}, Batch size: {self.batch_size}")
            print(f"  Learning rate: {self.learning_rate}, : {num_warmup_steps}")
        
        best_metric = -float('inf') if use_monitoring else None
        best_epoch = 0
        
        for epoch in range(1, self.max_epochs + 1):
            # NOTE: (translated from Chinese)
            self.model.train()
            total_loss = 0.0
            total_loss_sl = 0.0
            total_loss_ss = 0.0
            num_batches = 0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.max_epochs}", 
                       leave=False, ncols=100) if self.verbose > 0 else train_loader
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                scores = batch['scores'].to(self.device)
                cluster_labels = batch['cluster_label'].to(self.device)
                
                # NOTE: (translated from Chinese)
                if self.loss_type == "siglip_relevance":
                    logits, query_features = self.model(images, input_ids, attention_mask, return_features=True)
                else:
                    logits = self.model(images, input_ids, attention_mask)
                    query_features = None
                
                # NOTE: (translated from Chinese)
                if self.loss_type == "siglip_relevance":
                    loss_sl = nn.functional.binary_cross_entropy_with_logits(
                        logits,
                        scores.float().clamp(0.0, 1.0),
                    )
                    loss_ss = self._compute_soft_target_sample_sample_loss(query_features, scores)
                else:
                    loss_sl, loss_ss = self._compute_dual_contrastive_loss(
                        logits, scores, cluster_labels
                    )
                loss = loss_sl + self.lambda_sample_sample * loss_ss
                loss = loss / self.grad_accumulation_steps
                
                # NOTE: (translated from Chinese)
                loss.backward()
                
                # NOTE: (translated from Chinese)
                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.grad_accumulation_steps
                total_loss_sl += loss_sl.item()
                total_loss_ss += loss_ss.item()
                num_batches += 1
                
                if self.verbose > 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item() * self.grad_accumulation_steps:.4f}',
                        'sl': f'{loss_sl.item():.4f}',
                        'ss': f'{loss_ss.item():.4f}'
                    })
            
            avg_train_loss = total_loss / num_batches
            avg_loss_sl = total_loss_sl / num_batches
            avg_loss_ss = total_loss_ss / num_batches
            
            # NOTE: (translated from Chinese)
            if use_monitoring:
                #  train 
                train_acc, train_cost = self._evaluate_accuracy_cost(
                    train_loader, Y_data=None, C_data=None
                )
                train_metrics = {
                    'loss': float(avg_train_loss),
                    'loss_sl': float(avg_loss_sl),
                    'loss_ss': float(avg_loss_ss),
                    'accuracy': float(train_acc),
                    'avg_cost': float(train_cost)
                }
                
                #  dev 
                dev_acc, dev_cost = self._evaluate_accuracy_cost(
                    dev_loader, Y_data=Y_dev, C_data=C_dev
                )
                dev_metrics = {
                    'accuracy': float(dev_acc),
                    'avg_cost': float(dev_cost)
                }

                # Add Rank Score (dev) for model selection
                try:
                    from routers.utils.rank_score import rank_score
                    cmin = self._rank_score_cmin
                    cmax = self._rank_score_cmax
                    beta = self._rank_score_beta
                    if cmin is None or cmax is None:
                        cmin = float(np.min(C_dev))
                        cmax = float(np.max(C_dev))
                    dev_metrics['rank_score'] = float(rank_score(float(dev_acc), float(dev_cost), float(cmin), float(cmax), beta=float(beta)))
                    dev_metrics['rank_score_beta'] = float(beta)
                except Exception:
                    pass
                
                # For display only (checkpointing handled by TrainingMonitor)
                current_metric = dev_metrics.get(self.monitor_metric, None)
                is_best = False
                try:
                    if current_metric is not None:
                        is_best = monitor.is_better(float(current_metric), float(monitor.best_metric_value))
                except Exception:
                    is_best = False
                
                # NOTE: (translated from Chinese)
                should_stop = monitor.log_epoch(epoch, train_metrics, dev_metrics, self)
                
                if self.verbose > 0:
                    status = "🌟 [BEST]" if is_best else "         "
                    print(f"  {status} Epoch {epoch:3d}/{self.max_epochs}: "
                          f"loss={avg_train_loss:.4f} (sl={avg_loss_sl:.4f}, ss={avg_loss_ss:.4f}), "
                          f"train_acc={train_acc:.4f}, "
                          f"dev_acc={dev_acc:.4f}, "
                          f"cost=${dev_cost:.6f}, "
                          f"RS={dev_metrics.get('rank_score', float('nan')):.4f}")
                
                if should_stop:
                    if self.verbose > 0:
                        print(f"\n⏹️  early stopping！ epoch: {best_epoch} (dev_acc={best_metric:.4f})")
                    break
            else:
                if self.verbose > 0:
                    print(f"  Epoch {epoch:3d}/{self.max_epochs}: "
                          f"loss={avg_train_loss:.4f} (sl={avg_loss_sl:.4f}, ss={avg_loss_ss:.4f})")
        
        # NOTE: (translated from Chinese)
        if use_monitoring:
            monitor.end_training(epoch)
            monitor.plot_training_curves()
            
            # NOTE: (translated from Chinese)
            if monitor.best_model_path and Path(monitor.best_model_path).exists():
                if self.verbose > 0:
                    print(f"\n  loadbest model (epoch {monitor.best_epoch})...")
                try:
                    loaded_router = CosineCLSRRouter.load(monitor.best_model_path, device=self.device)
                    # Ensure we restore the exact weights/components used for inference
                    self.model = loaded_router.model.to(self.device) if loaded_router.model is not None else self.model
                    self.model.eval()
                    self.tokenizer = loaded_router.tokenizer
                    self.image_transform = loaded_router.image_transform
                    # Also restore these for completeness / future saving
                    self.kmeans = loaded_router.kmeans
                    self.model_mapping = loaded_router.model_mapping
                    if self.verbose > 0:
                        print(f"  ✓ best modelload (dev {self.monitor_metric}={monitor.best_metric_value:.6f})")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"  ⚠️  loadbest model: {e}")
            
            if self.verbose > 0:
                print(f"\n  📊 training:")
                print(f"     Output directory: {monitor_output_dir}")
                print(f"      epoch: {monitor.best_epoch}")
                print(f"      metric: {monitor.best_metric_value:.4f}")
    
    def _compute_dual_contrastive_loss(self, logits, scores, cluster_labels):
        """
        ：Sample-LLM + Sample-Sample
        
        Args:
            logits: (N, K) 
            scores: (N, K) 
            cluster_labels: (N,) 
        
        Returns:
            loss_sl: Sample-LLM 
            loss_ss: Sample-Sample 
        """
        N, K = logits.shape
        device = logits.device
        
        # 1. Sample-LLM 
        #  top-K_plus  top-K_minus 
        _, top_indices = torch.topk(scores, self.K_plus, dim=1)
        _, bottom_indices = torch.topk(scores, self.K_minus, dim=1, largest=False)
        
        # mask
        pos_mask = torch.zeros_like(logits, dtype=torch.bool)
        neg_mask = torch.zeros_like(logits, dtype=torch.bool)
        
        for i in range(N):
            pos_mask[i, top_indices[i]] = True
            neg_mask[i, bottom_indices[i]] = True
        
        # InfoNCE loss
        exp_logits = torch.exp(logits)
        pos_sum = (exp_logits * pos_mask).sum(dim=1)
        all_sum = exp_logits.sum(dim=1)
        
        loss_sl = -torch.log(pos_sum / (all_sum + 1e-8)).mean()
        
        # 2. Sample-Sample 
        # ，
        # （ query embeddings）
        # ： logits  L2 
        
        # mask
        cluster_mask = (cluster_labels.unsqueeze(0) == cluster_labels.unsqueeze(1))  # (N, N)
        cluster_mask.fill_diagonal_(False)  # 
        
        # （ logits ）
        logits_norm = nn.functional.normalize(logits, p=2, dim=1)
        sample_sim = logits_norm @ logits_norm.T  # (N, N)
        
        #  top-H_out_group 
        # NOTE: (translated from Chinese)
        neg_cluster_mask = ~cluster_mask
        
        # InfoNCE for samples
        exp_sim = torch.exp(sample_sim)
        pos_sim_sum = (exp_sim * cluster_mask).sum(dim=1)
        all_sim_sum = exp_sim.sum(dim=1)
        
        # NOTE: (translated from Chinese)
        has_pos = cluster_mask.sum(dim=1) > 0
        if has_pos.sum() > 0:
            loss_ss = -torch.log(pos_sim_sum[has_pos] / (all_sim_sum[has_pos] + 1e-8)).mean()
        else:
            loss_ss = torch.tensor(0.0, device=device)
        
        return loss_sl, loss_ss

    def _compute_soft_target_sample_sample_loss(self, query_features, scores):
        if self.lambda_sample_sample <= 0 or query_features is None or query_features.shape[0] <= 1:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        q = nn.functional.normalize(query_features, p=2, dim=1)
        target = scores.float().clamp_min(0.0)
        target_norm = target / target.sum(dim=1, keepdim=True).clamp_min(1e-8)
        sim_target = target_norm @ target_norm.T
        sim_target.fill_diagonal_(0.0)
        row_sum = sim_target.sum(dim=1, keepdim=True)
        valid = row_sum.squeeze(1) > 1e-8
        if valid.sum() == 0:
            return q.new_tensor(0.0)
        sim_target = sim_target / row_sum.clamp_min(1e-8)
        sample_logits = (q @ q.T) / max(float(self.sample_sample_temperature), 1e-6)
        sample_logits = sample_logits.masked_fill(
            torch.eye(sample_logits.shape[0], dtype=torch.bool, device=sample_logits.device),
            -torch.finfo(sample_logits.dtype).max,
        )
        log_prob = nn.functional.log_softmax(sample_logits, dim=1)
        return -(sim_target[valid] * log_prob[valid]).sum(dim=1).mean()
    
    def _evaluate_accuracy_cost(self, data_loader, Y_data=None, C_data=None):
        """evaluationAccuracyAvg cost"""
        self.model.eval()
        
        all_preds = []
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(images, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
        
        all_preds = np.array(all_preds)
        
        # NOTE: (translated from Chinese)
        # - if Y_data is provided: use ground-truth correctness (Y > 0) like other routers
        # - else: fallback to "match the best model under S_scores" (legacy behavior)
        if Y_data is not None:
            correct = Y_data[np.arange(len(all_preds)), all_preds]
            accuracy = float(np.mean(correct))
        else:
            scores = data_loader.dataset.S_scores
            best_models = np.argmax(scores, axis=1)
            accuracy = float((all_preds == best_models).mean())
        
        # NOTE: (translated from Chinese)
        avg_cost = 0.0
        if C_data is not None:
            costs = C_data[np.arange(len(all_preds)), all_preds]
            avg_cost = costs.mean()
        
        return accuracy, avg_cost
    
    def predict(self, meta, X=None, batch_size=None, num_workers=4, show_progress=False, **kwargs) -> np.ndarray:
        """
        
        
        Args:
            meta:  DataFrame（ 'text'  'assets' ）
            X: （）
            batch_size: batch size（batch_size4，）
            num_workers: （4，0）
            show_progress: 
        
        Returns:
             (N)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("trainingload")
        
        # batch_size
        if batch_size is None:
            batch_size = self.batch_size * 4
        
        # NOTE: (translated from Chinese)
        texts, images = self._extract_texts_and_images_from_meta(meta)
        
        # NOTE: (translated from Chinese)
        dataset = VLPredictDataset(
            images_list=images,
            texts=texts,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            image_transform=self.image_transform,
            image_base_dir='.'  # 
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == 'cuda')  # pin_memoryGPU
        )
        
        # NOTE: (translated from Chinese)
        if show_progress:
            try:
                from tqdm import tqdm
                data_loader = tqdm(data_loader, desc="", unit="batch")
            except ImportError:
                pass
        
        # NOTE: (translated from Chinese)
        self.model.eval()
        all_preds = []
        
        with torch.no_grad():
            for batch in data_loader:
                images_batch = batch['image'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                logits = self.model(images_batch, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
        
        return np.array(all_preds)
    
    def save(self, path: str):
        """save"""
        state = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'tokenizer': self.tokenizer,
            'image_transform': self.image_transform,
            'kmeans': self.kmeans,
            'model_mapping': self.model_mapping,
            'hyperparameters': {
                'K_plus': self.K_plus,
                'K_minus': self.K_minus,
                'lambda_sample_sample': self.lambda_sample_sample,
                'num_clusters': self.num_clusters,
                'H_out_group': self.H_out_group,
                'loss_type': self.loss_type,
                'sample_sample_temperature': self.sample_sample_temperature,
                'use_soft_labels': self.use_soft_labels,
                'train_lambda': self.train_lambda,
                'cost_scale': self.cost_scale,
                'max_length': self.max_length,
                'max_epochs': self.max_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'device': self.device
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        if self.verbose > 0:
            print(f"✓ Model saved: {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cuda'):
        """load"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        #  router
        hyperparams = state['hyperparameters']
        router = cls(
            K_plus=hyperparams['K_plus'],
            K_minus=hyperparams['K_minus'],
            lambda_sample_sample=hyperparams['lambda_sample_sample'],
            num_clusters=hyperparams['num_clusters'],
            H_out_group=hyperparams['H_out_group'],
            loss_type=hyperparams.get('loss_type', 'routerdc'),
            sample_sample_temperature=hyperparams.get('sample_sample_temperature', 0.1),
            use_soft_labels=hyperparams.get('use_soft_labels', False),
            train_lambda=hyperparams.get('train_lambda', 0.0),
            cost_scale=hyperparams.get('cost_scale', 100.0),
            max_length=hyperparams['max_length'],
            max_epochs=hyperparams['max_epochs'],
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            device=device,
            verbose=0
        )
        
        # NOTE: (translated from Chinese)
        router.tokenizer = state['tokenizer']
        router.image_transform = state['image_transform']
        router.kmeans = state['kmeans']
        router.model_mapping = state['model_mapping']
        
        # NOTE: (translated from Chinese)
        if state['model_state_dict']:
            num_classes = len(router.model_mapping)
            router.model = CosineCLSRModel(num_classes=num_classes).to(device)
            router.model.load_state_dict(state['model_state_dict'])
        
        return router
