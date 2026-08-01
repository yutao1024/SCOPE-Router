#!/usr/bin/env python3
"""
Zooter Router: LXMERT

 Zooter  Transformer backbone  LXMERT 

"""

import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Literal
import warnings

from routers.common import RouterBase

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        LxmertModel,
        ViTModel,
        get_linear_schedule_with_warmup
    )
    from torchvision import transforms
    from PIL import Image
    import base64
    from io import BytesIO
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None

try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    KMeans = None
    HAS_SKLEARN = False


if HAS_TORCH:
    class LXMERTClassifier(nn.Module):
        """
        LXMERT （）
        """
        def __init__(self, num_classes: int, dropout: float = 0.1, freeze_backbone: bool = False):
            super().__init__()
            
            # Vision Encoder: ViT
            self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.vision_hidden_size = self.vision_encoder.config.hidden_size  # 768
            
            # LXMERT Model
            self.lxmert = LxmertModel.from_pretrained('unc-nlp/lxmert-base-uncased')
            self.lxmert_hidden_size = self.lxmert.config.hidden_size  # 768
            
            # Visual projection: ViT (768) -> LXMERT visual_feat_dim (2048)
            self.visual_projection = nn.Linear(self.vision_hidden_size, 2048)
            
            # Dropout and classifier
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.lxmert_hidden_size, num_classes)
            
            # ：backbone
            if freeze_backbone:
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
                for param in self.lxmert.parameters():
                    param.requires_grad = False
        
        def forward(self, images, input_ids, attention_mask, return_features: bool = False):
            """
            Args:
                images: (batch_size, 3, 224, 224)
                input_ids: (batch_size, seq_len)
                attention_mask: (batch_size, seq_len)
            
            Returns:
                logits: (batch_size, num_classes)
            """
            batch_size = images.size(0)
            device = images.device
            
            # 1. Extract visual features using ViT
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
            pooled_output = lxmert_outputs.pooled_output  # (batch, hidden_size)
            features = self.dropout(pooled_output)
            logits = self.classifier(features)
            
            if return_features:
                return logits, features
            return logits
    
    
    class MultiModalDataset(Dataset):
        """（ + ）"""
        def __init__(self, images_list, texts, labels, tokenizer, 
                     max_length=512, is_soft_label=False, 
                     image_transform=None, image_base_dir='.',
                     cluster_labels=None):
            """
            Args:
                images_list: （PIL Image, ,  asset ）
                texts: 
                labels:  (N,)   (N, K) 
                tokenizer: Transformer tokenizer
                max_length: 
                is_soft_label: 
                image_transform: 
                image_base_dir: 
            """
            self.images_list = images_list
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.is_soft_label = is_soft_label
            self.image_transform = image_transform
            self.image_base_dir = Path(image_base_dir)
            self.cluster_labels = cluster_labels
        
        def __len__(self):
            return len(self.texts)
        
        def _load_image_from_tsv(self, asset):
            """ TSV load"""
            import pandas as pd
            
            tsv_file = asset.get('tsv_file') or asset.get('path') or asset.get('uri')
            lineno = int(asset.get('index', asset.get('lineno', 0)))
            
            #  TSV 
            df = pd.read_csv(tsv_file, sep='\t')
            if lineno >= len(df):
                return Image.new('RGB', (224, 224), color='black')
            
            if 'image' in df.columns:
                if 'index' in df.columns:
                    df['image'] = df['image'].astype(str)
                    image_map = {str(idx): img for idx, img in zip(df['index'], df['image'])}
                    for k in list(image_map.keys()):
                        if len(image_map[k]) <= 64:
                            ref_idx = image_map[k]
                            if ref_idx in image_map and len(image_map[ref_idx]) > 64:
                                image_map[k] = image_map[ref_idx]
                    row = df.iloc[lineno]
                    img_str = image_map.get(str(row['index']), row['image'])
                else:
                    img_str = df.iloc[lineno]['image']
            else:
                # Legacy headerless TSV fallback.
                df = pd.read_csv(tsv_file, sep='\t', header=None)
                if lineno >= len(df):
                    return Image.new('RGB', (224, 224), color='black')
                img_str = df.iloc[lineno, 1]

            img_bytes = base64.b64decode(img_str)
            image = Image.open(BytesIO(img_bytes)).convert('RGB')
            return image
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            image_source = self.images_list[idx]
            
            # （ VLC  routerdc ）
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
            
            item = {
                'image': image,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.float32 if self.is_soft_label else torch.long)
            }
            if self.cluster_labels is not None:
                item['cluster_label'] = torch.tensor(self.cluster_labels[idx], dtype=torch.long)
            return item


class ZooterRouter(RouterBase):
    """
    Zooter-LXMERT Router: LXMERT
    
    ：
    1.  LXMERT  backbone
    2. 
    3. （、dev）
    """
    
    def __init__(
        self,
        pretrained_model: str = 'unc-nlp/lxmert-base-uncased',
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_soft_labels: bool = False,
        train_lambda: float = 0.0,
        loss_type: Literal['softmax', 'siglip_relevance'] = 'softmax',
        sample_sample_weight: float = 0.0,
        sample_sample_temperature: float = 0.1,
        num_clusters: int = 5,
        cost_scale: float = 100.0,
        max_epochs: int = 5,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        enable_monitoring: bool = False,
        patience: int = 2,
        monitor_metric: Literal['rank_score', 'accuracy', 'avg_cost'] = 'rank_score',
        device: str = 'cuda',
        verbose: int = 1
    ):
        super().__init__()
        
        if not HAS_TORCH:
            raise ImportError("PyTorch  Transformers ")
        
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        self.use_soft_labels = use_soft_labels
        self.train_lambda = train_lambda
        if loss_type not in {'softmax', 'siglip_relevance'}:
            raise ValueError(f"loss_type must be softmax or siglip_relevance; got {loss_type}")
        self.loss_type = loss_type
        self.sample_sample_weight = float(sample_sample_weight)
        self.sample_sample_temperature = float(sample_sample_temperature)
        self.num_clusters = int(num_clusters)
        self.cost_scale = float(cost_scale)
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.enable_monitoring = enable_monitoring
        self.patience = patience
        self.monitor_metric = monitor_metric
        # Rank score configuration (used for monitoring on dev)
        self._rank_score_cmin = None
        self._rank_score_cmax = None
        self._rank_score_beta = 0.1
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.verbose = verbose
        
        self.model = None
        self.tokenizer = None
        self.image_transform = None
        self.model_mapping = None
    
    def fit(self, Y, C, meta, X=None, Y_dev=None, C_dev=None, meta_dev=None, 
            model_mapping=None, costs=None, monitor_output_dir=None, **kwargs):
        """
         Zooter-LXMERT Router
        
        Args:
            Y:  (N, K)
            C:  (N, K)
            meta:  DataFrame（ 'text'  'assets' ）
            Y_dev, C_dev, meta_dev: Dev （）
            model_mapping: 
            costs: （）
            monitor_output_dir: 
        """
        from tqdm.auto import tqdm
        
        if self.verbose > 0:
            print("\n" + "="*80)
            print("🚀 Zooter-LXMERT Router training")
            print("="*80)
            if self.loss_type == "siglip_relevance":
                print(f"training loss: SigLIP-style dense relevance (λ={self.train_lambda})")
            if self.use_soft_labels:
                print(f"training: Soft label (λ={self.train_lambda})")
            else:
                print(f"training: Hard label")
            if self.sample_sample_weight > 0:
                print(
                    f"sample-sample: weight={self.sample_sample_weight:g}, "
                    f"temp={self.sample_sample_temperature:g}, clusters={self.num_clusters}"
                )
        
        # 1. 
        texts_train, images_train = self._extract_texts_and_images_from_meta(meta)
        
        # 2. 
        uses_dense_targets = self.use_soft_labels or self.loss_type == "siglip_relevance"
        if self.loss_type == "siglip_relevance":
            from routers.utils.soft_targets import build_relevance_targets
            T_train = build_relevance_targets(
                Y, C,
                lam=self.train_lambda,
                cost_scale=self.cost_scale,
                fallback='zeros',
            ).astype(np.float32)
            if self.verbose > 0:
                print(f"✓ Dense relevance targets (λ={self.train_lambda})")
        elif self.use_soft_labels:
            try:
                from routers.utils.soft_targets import build_soft_targets
                T_train = build_soft_targets(
                    Y, C,
                    lam=self.train_lambda,
                    scheme='exp',
                    fallback='cheapest',
                    cost_scale=self.cost_scale
                )
                if self.verbose > 0:
                    print(f"✓ Soft label (λ={self.train_lambda})")
            except ImportError:
                if self.verbose > 0:
                    print("⚠️  Warning: Failed to import soft_targets，Hard label")
                T_train = self._derive_hard_labels(Y, C)
                self.use_soft_labels = False
        else:
            T_train = self._derive_hard_labels(Y, C)
        
        # 3. 
        num_classes = Y.shape[1]
        
        if self.verbose > 0:
            print(f"\n🏗️   Zooter-LXMERT ...")
            print(f"  : {num_classes}")
            print(f"  : {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = LXMERTClassifier(
            num_classes=num_classes,
            dropout=self.dropout,
            freeze_backbone=self.freeze_backbone
        ).to(self.device)
        
        # Image transform (ViT preprocessing)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        
        cluster_labels = None
        if self.sample_sample_weight > 0:
            if not HAS_SKLEARN:
                raise ImportError("sample-sample loss requires scikit-learn for KMeans")
            scores_for_clusters = T_train.astype(np.float32) if uses_dense_targets else Y.astype(np.float32)
            if not uses_dense_targets:
                row_sums = scores_for_clusters.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1.0
                scores_for_clusters = scores_for_clusters / row_sums
            k = min(self.num_clusters, len(scores_for_clusters))
            self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(scores_for_clusters)

        # 5. 
        train_dataset = MultiModalDataset(
            images_list=images_train,
            texts=texts_train,
            labels=T_train,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            is_soft_label=uses_dense_targets,
            image_transform=self.image_transform,
            image_base_dir='.',
            cluster_labels=cluster_labels
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        #  dev （）
        dev_loader = None
        if self.enable_monitoring and Y_dev is not None:
            texts_dev, images_dev = self._extract_texts_and_images_from_meta(meta_dev)
            
            if self.loss_type == "siglip_relevance":
                from routers.utils.soft_targets import build_relevance_targets
                T_dev = build_relevance_targets(
                    Y_dev, C_dev,
                    lam=self.train_lambda,
                    cost_scale=self.cost_scale,
                    fallback='zeros',
                ).astype(np.float32)
            elif self.use_soft_labels:
                try:
                    from routers.utils.soft_targets import build_soft_targets
                    T_dev = build_soft_targets(Y_dev, C_dev, lam=self.train_lambda,
                                               scheme='exp', fallback='cheapest', cost_scale=self.cost_scale)
                except:
                    T_dev = self._derive_hard_labels(Y_dev, C_dev)
            else:
                T_dev = self._derive_hard_labels(Y_dev, C_dev)
            
            dev_dataset = MultiModalDataset(
                images_list=images_dev,
                texts=texts_dev,
                labels=T_dev,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                is_soft_label=uses_dense_targets,
                image_transform=self.image_transform,
                image_base_dir='.'
            )
            
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0
            )
        
        # 6. 
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(train_loader) * self.max_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 7. 
        if self.loss_type == "siglip_relevance":
            criterion = nn.BCEWithLogitsLoss()
        elif self.use_soft_labels:
            criterion = nn.KLDivLoss(reduction='batchmean')
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 8. 
        monitor = None
        if self.enable_monitoring:
            from routers.utils.training_monitor import TrainingMonitor
            monitor = TrainingMonitor(
                patience=self.patience,
                metric=self.monitor_metric,
                output_dir=monitor_output_dir,
                maximize=True if self.monitor_metric in ['accuracy', 'rank_score'] else False,
                verbose=self.verbose
            )
            monitor.start_training()
        
        # 9. 
        if self.verbose > 0:
            print(f"\n🎓 training...")
            print(f"  Epochs: {self.max_epochs}")
            print(f"  Batch size: {self.batch_size}")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Warmup steps: {warmup_steps}/{total_steps}")
        
        for epoch in range(1, self.max_epochs + 1):
            # NOTE: (translated from Chinese)
            self.model.train()
            total_loss = 0.0
            total_loss_main = 0.0
            total_loss_ss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.max_epochs}", 
                       leave=False, ncols=100) if self.verbose > 0 else train_loader
            
            for batch in pbar:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # NOTE: (translated from Chinese)
                if self.sample_sample_weight > 0:
                    logits, query_features = self.model(images, input_ids, attention_mask, return_features=True)
                else:
                    logits = self.model(images, input_ids, attention_mask)
                    query_features = None
                
                # NOTE: (translated from Chinese)
                if self.loss_type == "siglip_relevance":
                    main_loss = criterion(logits, labels.float().clamp(0.0, 1.0))
                elif self.use_soft_labels:
                    log_probs = torch.log_softmax(logits, dim=-1)
                    main_loss = criterion(log_probs, labels)
                else:
                    main_loss = criterion(logits, labels)
                ss_loss = self._sample_sample_loss(query_features, labels) if self.sample_sample_weight > 0 else logits.new_tensor(0.0)
                loss = main_loss + self.sample_sample_weight * ss_loss
                
                # NOTE: (translated from Chinese)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                total_loss_main += main_loss.item()
                total_loss_ss += ss_loss.item()
                num_batches += 1
                
                if self.verbose > 0:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'main': f'{main_loss.item():.4f}',
                        'ss': f'{ss_loss.item():.4f}',
                    })
            
            avg_train_loss = total_loss / num_batches
            avg_main_loss = total_loss_main / num_batches
            avg_ss_loss = total_loss_ss / num_batches
            
            # NOTE: (translated from Chinese)
            if self.enable_monitoring and dev_loader is not None:
                dev_acc, dev_cost = self._evaluate_accuracy_cost(
                    dev_loader, Y_data=Y_dev, C_data=C_dev
                )
                train_metrics = {
                    'loss': float(avg_train_loss),
                    'loss_main': float(avg_main_loss),
                    'loss_ss': float(avg_ss_loss),
                }
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
                
                # NOTE: (translated from Chinese)
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
                          f"loss={avg_train_loss:.4f}, "
                          f"dev_acc={dev_acc:.4f}, "
                          f"cost=${dev_cost:.6f}")
                
                if should_stop:
                    if self.verbose > 0:
                        print(f"\n⏹️  early stopping！")
                    break
            else:
                if self.verbose > 0:
                    print(f"  Epoch {epoch:3d}/{self.max_epochs}: loss={avg_train_loss:.4f}")
        
        # 10. （）
        if self.enable_monitoring and monitor and monitor.best_model_path:
            if Path(monitor.best_model_path).exists():
                if self.verbose > 0:
                    print(f"\n  loadbest model (epoch {monitor.best_epoch})...")
                try:
                    loaded_router = ZooterRouter.load(monitor.best_model_path, device=self.device)
                    self.model = loaded_router.model.to(self.device) if loaded_router.model is not None else self.model
                    self.model.eval()
                    self.tokenizer = loaded_router.tokenizer
                    self.image_transform = loaded_router.image_transform
                    self.model_mapping = loaded_router.model_mapping
                    if self.verbose > 0:
                        print(f"  ✓ best modelload")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"  ⚠️  Failed to loadbest model: {e}")
        
        if self.verbose > 0:
            print("\n✓ Training complete！")
    
    def _derive_hard_labels(self, Y, C):
        """Hard label（）"""
        N, K = Y.shape
        labels = np.zeros(N, dtype=np.int64)
        
        for i in range(N):
            correct_models = np.where(Y[i] > 0)[0]
            if len(correct_models) > 0:
                # NOTE: (translated from Chinese)
                cheapest_idx = correct_models[np.argmin(C[i, correct_models])]
                labels[i] = cheapest_idx
            else:
                # ，
                labels[i] = np.argmin(C[i])
        
        return labels

    def _sample_sample_loss(self, query_features, scores):
        if query_features is None or scores is None or query_features.shape[0] <= 1:
            return torch.tensor(0.0, dtype=torch.float32, device=self.device)
        q = F.normalize(query_features, p=2, dim=1)
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
        log_prob = F.log_softmax(sample_logits, dim=1)
        return -(sim_target[valid] * log_prob[valid]).sum(dim=1).mean()
    
    def _extract_texts_and_images_from_meta(self, meta):
        """ meta DataFrame """
        texts = []
        images = []
        
        for idx, row in meta.iterrows():
            # NOTE: (translated from Chinese)
            if 'text' in row and row['text']:
                text = str(row['text'])
            elif 'prompt' in row and row['prompt']:
                text = str(row['prompt'])
            else:
                text = ""
            texts.append(text)
            
            # NOTE: (translated from Chinese)
            if 'assets' in row and row['assets']:
                images.append(row['assets'])
            elif 'image' in row and row['image']:
                images.append(row['image'])
            else:
                # NOTE: (translated from Chinese)
                images.append(Image.new('RGB', (224, 224), color='black'))
        
        return texts, images
    
    def _evaluate_accuracy_cost(self, data_loader, Y_data=None, C_data=None):
        """
        
        
         router_dc ： Y > 0 
        """
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
        
        #  (Y > 0)
        if Y_data is not None:
            correct = Y_data[np.arange(len(all_preds)), all_preds] > 0
            accuracy = correct.mean()
        else:
            accuracy = 0.0
        
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
        dummy_labels = np.zeros(len(texts), dtype=np.int64)
        dataset = MultiModalDataset(
            images_list=images,
            texts=texts,
            labels=dummy_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            is_soft_label=False,
            image_transform=self.image_transform,
            image_base_dir='.'
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(self.device == 'cuda')  # pin_memoryGPU
        )
        
        self.model.eval()
        all_preds = []
        
        # NOTE: (translated from Chinese)
        if show_progress:
            try:
                from tqdm import tqdm
                data_loader = tqdm(data_loader, desc="", unit="batch")
            except ImportError:
                pass
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                logits = self.model(images, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
        
        return np.array(all_preds)
    
    def save(self, path: str):
        """save"""
        state = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'tokenizer': self.tokenizer,
            'image_transform': self.image_transform,
            'model_mapping': self.model_mapping,
            'hyperparameters': {
                'pretrained_model': self.pretrained_model,
                'max_length': self.max_length,
                'dropout': self.dropout,
                'freeze_backbone': self.freeze_backbone,
                'use_soft_labels': self.use_soft_labels,
                'train_lambda': self.train_lambda,
                'loss_type': self.loss_type,
                'sample_sample_weight': self.sample_sample_weight,
                'sample_sample_temperature': self.sample_sample_temperature,
                'num_clusters': self.num_clusters,
                'cost_scale': self.cost_scale,
                'max_epochs': self.max_epochs,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'warmup_ratio': self.warmup_ratio,
                'weight_decay': self.weight_decay,
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
            pretrained_model=hyperparams['pretrained_model'],
            max_length=hyperparams['max_length'],
            dropout=hyperparams['dropout'],
            freeze_backbone=hyperparams['freeze_backbone'],
            use_soft_labels=hyperparams.get('use_soft_labels', False),
            train_lambda=hyperparams.get('train_lambda', 0.0),
            loss_type=hyperparams.get('loss_type', 'softmax'),
            sample_sample_weight=hyperparams.get('sample_sample_weight', 0.0),
            sample_sample_temperature=hyperparams.get('sample_sample_temperature', 0.1),
            num_clusters=hyperparams.get('num_clusters', 5),
            cost_scale=hyperparams.get('cost_scale', 100.0),
            max_epochs=hyperparams['max_epochs'],
            learning_rate=hyperparams['learning_rate'],
            batch_size=hyperparams['batch_size'],
            warmup_ratio=hyperparams['warmup_ratio'],
            weight_decay=hyperparams['weight_decay'],
            device=device,
            verbose=0
        )
        
        # NOTE: (translated from Chinese)
        router.tokenizer = state['tokenizer']
        router.image_transform = state['image_transform']
        router.model_mapping = state['model_mapping']
        
        # NOTE: (translated from Chinese)
        if state['model_state_dict']:
            num_classes = len(router.model_mapping)
            router.model = LXMERTClassifier(
                num_classes=num_classes,
                dropout=hyperparams['dropout'],
                freeze_backbone=hyperparams['freeze_backbone']
            ).to(device)
            router.model.load_state_dict(state['model_state_dict'])
        
        return router
