#!/usr/bin/env python3
"""
VLC (Vision-Language Compositional) Router

-backbone
：
- VisualBERT: 
- LXMERT: 
- UNITER: （modal type embeddings）
- ViLBERT: （co-attention）

：
- BERT Router: 
- MobileNet Router: Dataset
- UNITER: https://github.com/ChenRocks/UNITER
- ViLBERT: https://github.com/jiasenlu/vilbert_beta
"""

import numpy as np
import pickle
import sys
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
import warnings

from routers.common import RouterBase

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
    from PIL import Image
    from transformers import (
        AutoTokenizer, AutoModel,
        VisualBertModel, VisualBertConfig,
        LxmertModel, LxmertConfig,
        BertModel, BertConfig,
        get_linear_schedule_with_warmup
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


if HAS_TORCH:
    # ========================================================================
    # UNITER Model Implementation ( https://github.com/ChenRocks/UNITER)
    # ========================================================================
    class ModalTypeEmbeddings(nn.Module):
        """（UNITER）"""
        def __init__(self, hidden_size: int, num_types: int = 2):
            super().__init__()
            self.embeddings = nn.Embedding(num_types, hidden_size)
        
        def forward(self, modal_type_ids):
            return self.embeddings(modal_type_ids)
    
    
    class UNITERModel(nn.Module):
        """
        UNITER: UNiversal Image-TExt Representation
        ，modal type embeddings
        
        : https://github.com/ChenRocks/UNITER
        : Chen et al. "UNITER: Learning UNiversal Image-TExt Representations" (ECCV 2020)
        """
        def __init__(self, config=None):
            super().__init__()
            
            if config is None:
                config = BertConfig.from_pretrained('bert-base-uncased')
            
            self.config = config
            
            # BERT backbone
            self.bert = BertModel(config)
            
            # Modal type embeddings（0=, 1=）
            self.modal_type_embeddings = ModalTypeEmbeddings(
                hidden_size=config.hidden_size,
                num_types=2
            )
            
            # NOTE: (translated from Chinese)
            self.img_embeddings = nn.Linear(768, config.hidden_size)  # ViT 768 -> BERT hidden
            
        def forward(
            self,
            input_ids,
            attention_mask,
            visual_embeds,
            visual_attention_mask
        ):
            """
            Args:
                input_ids: (batch, text_seq_len)
                attention_mask: (batch, text_seq_len)
                visual_embeds: (batch, num_patches, 768)
                visual_attention_mask: (batch, num_patches)
            
            Returns:
                pooled_output: (batch, hidden_size)
            """
            batch_size = input_ids.size(0)
            device = input_ids.device
            
            # 1. embeddings
            text_embeds = self.bert.embeddings.word_embeddings(input_ids)
            text_pos_embeds = self.bert.embeddings.position_embeddings(
                torch.arange(input_ids.size(1), device=device).unsqueeze(0).expand_as(input_ids)
            )
            text_embeds = text_embeds + text_pos_embeds
            
            # modal type embedding (=0)
            text_modal_ids = torch.zeros(
                (batch_size, input_ids.size(1)),
                dtype=torch.long,
                device=device
            )
            text_embeds = text_embeds + self.modal_type_embeddings(text_modal_ids)
            
            # 2. embeddings
            img_embeds = self.img_embeddings(visual_embeds)  # Project to BERT hidden size
            
            # （BERT，text_seq_len）
            img_seq_len = visual_embeds.size(1)
            img_pos_ids = torch.arange(
                input_ids.size(1),
                input_ids.size(1) + img_seq_len,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
            
            # max_position_embeddings
            max_pos = self.bert.embeddings.position_embeddings.num_embeddings
            img_pos_ids = img_pos_ids % max_pos
            
            img_pos_embeds = self.bert.embeddings.position_embeddings(img_pos_ids)
            img_embeds = img_embeds + img_pos_embeds
            
            # modal type embedding (=1)
            img_modal_ids = torch.ones(
                (batch_size, img_seq_len),
                dtype=torch.long,
                device=device
            )
            img_embeds = img_embeds + self.modal_type_embeddings(img_modal_ids)
            
            # 3. Concatenate text and image
            combined_embeds = torch.cat([text_embeds, img_embeds], dim=1)
            combined_attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            
            # 4. BERT encoder
            extended_attention_mask = self.bert.get_extended_attention_mask(
                combined_attention_mask,
                combined_attention_mask.shape,
                device
            )
            
            encoder_outputs = self.bert.encoder(
                combined_embeds,
                attention_mask=extended_attention_mask
            )
            
            sequence_output = encoder_outputs[0]
            
            # 5. Pooling ([CLS] token)
            pooled_output = self.bert.pooler(sequence_output)
            
            return type('Outputs', (), {
                'pooler_output': pooled_output,
                'last_hidden_state': sequence_output
            })()
    
    
    # ========================================================================
    # ViLBERT Model Implementation ( https://github.com/jiasenlu/vilbert_beta)
    # ========================================================================
    class CoAttentionLayer(nn.Module):
        """Co-Attention（ViLBERT）"""
        def __init__(self, config):
            super().__init__()
            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
            self.all_head_size = self.num_attention_heads * self.attention_head_size
            
            # Query, Key, Value for both modalities
            self.query1 = nn.Linear(config.hidden_size, self.all_head_size)
            self.key1 = nn.Linear(config.hidden_size, self.all_head_size)
            self.value1 = nn.Linear(config.hidden_size, self.all_head_size)
            
            self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
            self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
            self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
            
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
            
        def transpose_for_scores(self, x):
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
            return x.permute(0, 2, 1, 3)
        
        def forward(self, hidden_states1, hidden_states2, attention_mask1, attention_mask2):
            """
            Co-attention between two modalities
            
            Args:
                hidden_states1: (batch, seq1, hidden)
                hidden_states2: (batch, seq2, hidden)
                attention_mask1: (batch, seq1)
                attention_mask2: (batch, seq2)
            
            Returns:
                attended1, attended2
            """
            # Modality 1 attends to Modality 2
            q1 = self.transpose_for_scores(self.query1(hidden_states1))
            k2 = self.transpose_for_scores(self.key2(hidden_states2))
            v2 = self.transpose_for_scores(self.value2(hidden_states2))
            
            attention_scores1 = torch.matmul(q1, k2.transpose(-1, -2))
            attention_scores1 = attention_scores1 / (self.attention_head_size ** 0.5)
            
            if attention_mask2 is not None:
                extended_mask2 = attention_mask2.unsqueeze(1).unsqueeze(2)
                extended_mask2 = (1.0 - extended_mask2) * -10000.0
                attention_scores1 = attention_scores1 + extended_mask2
            
            attention_probs1 = F.softmax(attention_scores1, dim=-1)
            attention_probs1 = self.dropout(attention_probs1)
            
            context1 = torch.matmul(attention_probs1, v2)
            context1 = context1.permute(0, 2, 1, 3).contiguous()
            new_context1_shape = context1.size()[:-2] + (self.all_head_size,)
            attended1 = context1.view(*new_context1_shape)
            
            # Modality 2 attends to Modality 1
            q2 = self.transpose_for_scores(self.query2(hidden_states2))
            k1 = self.transpose_for_scores(self.key1(hidden_states1))
            v1 = self.transpose_for_scores(self.value1(hidden_states1))
            
            attention_scores2 = torch.matmul(q2, k1.transpose(-1, -2))
            attention_scores2 = attention_scores2 / (self.attention_head_size ** 0.5)
            
            if attention_mask1 is not None:
                extended_mask1 = attention_mask1.unsqueeze(1).unsqueeze(2)
                extended_mask1 = (1.0 - extended_mask1) * -10000.0
                attention_scores2 = attention_scores2 + extended_mask1
            
            attention_probs2 = F.softmax(attention_scores2, dim=-1)
            attention_probs2 = self.dropout(attention_probs2)
            
            context2 = torch.matmul(attention_probs2, v1)
            context2 = context2.permute(0, 2, 1, 3).contiguous()
            new_context2_shape = context2.size()[:-2] + (self.all_head_size,)
            attended2 = context2.view(*new_context2_shape)
            
            return attended1, attended2
    
    
    class ViLBERTModel(nn.Module):
        """
        ViLBERT: Vision-and-Language BERT
        ，co-attention
        
        : https://github.com/jiasenlu/vilbert_beta
        : Lu et al. "ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations" (NeurIPS 2019)
        """
        def __init__(self, config=None):
            super().__init__()
            
            if config is None:
                config = BertConfig.from_pretrained('bert-base-uncased')
            
            self.config = config
            
            # visionlanguage BERT streams
            self.text_bert = BertModel(config)
            self.vision_bert = BertModel(config)
            
            # Co-attention layers
            self.co_attention_layers = nn.ModuleList([
                CoAttentionLayer(config) for _ in range(2)  # ：2co-attention
            ])
            
            # NOTE: (translated from Chinese)
            self.img_embeddings = nn.Linear(768, config.hidden_size)
            
            # Pooler for combined output
            self.pooler = nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.Tanh()
            )
            
        def forward(
            self,
            input_ids,
            attention_mask,
            visual_embeds,
            visual_attention_mask
        ):
            """
            Args:
                input_ids: (batch, text_seq_len)
                attention_mask: (batch, text_seq_len)
                visual_embeds: (batch, num_patches, 768)
                visual_attention_mask: (batch, num_patches)
            
            Returns:
                pooled_output: (batch, hidden_size)
            """
            device = input_ids.device
            
            # 1. Text stream - 
            text_outputs = self.text_bert.embeddings(input_ids)
            
            # 2. Vision stream - 
            img_embeds = self.img_embeddings(visual_embeds)
            # NOTE: (translated from Chinese)
            batch_size, img_seq_len = visual_embeds.shape[:2]
            img_pos_ids = torch.arange(img_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            img_pos_embeds = self.vision_bert.embeddings.position_embeddings(img_pos_ids)
            vision_outputs = img_embeds + img_pos_embeds
            
            # 3. BERT（：）
            # Text encoding
            extended_text_mask = self.text_bert.get_extended_attention_mask(
                attention_mask,
                attention_mask.shape,
                device
            )
            
            # BERT
            num_layers = len(self.text_bert.encoder.layer)
            mid_layer = num_layers // 2
            
            for layer in self.text_bert.encoder.layer[:mid_layer]:
                text_outputs = layer(text_outputs, attention_mask=extended_text_mask)[0]
            
            # Vision encoding
            extended_vision_mask = self.vision_bert.get_extended_attention_mask(
                visual_attention_mask,
                visual_attention_mask.shape,
                device
            )
            
            for layer in self.vision_bert.encoder.layer[:mid_layer]:
                vision_outputs = layer(vision_outputs, attention_mask=extended_vision_mask)[0]
            
            # 4. Co-attention
            for co_attn_layer in self.co_attention_layers:
                text_co_attn, vision_co_attn = co_attn_layer(
                    text_outputs, vision_outputs,
                    attention_mask, visual_attention_mask
                )
                # Residual connection
                text_outputs = text_outputs + text_co_attn
                vision_outputs = vision_outputs + vision_co_attn
            
            # 5. 
            for layer in self.text_bert.encoder.layer[mid_layer:]:
                text_outputs = layer(text_outputs, attention_mask=extended_text_mask)[0]
            
            for layer in self.vision_bert.encoder.layer[mid_layer:]:
                vision_outputs = layer(vision_outputs, attention_mask=extended_vision_mask)[0]
            
            # 6. Pooling: [CLS] token
            text_pooled = text_outputs[:, 0, :]  # [CLS] of text
            vision_pooled = vision_outputs[:, 0, :]  # [CLS] of vision
            
            # NOTE: (translated from Chinese)
            combined = torch.cat([text_pooled, vision_pooled], dim=-1)
            pooled_output = self.pooler(combined)
            
            return type('Outputs', (), {
                'pooler_output': pooled_output,
                'text_hidden_state': text_outputs,
                'vision_hidden_state': vision_outputs
            })()
    
    
    # ========================================================================
    # VLClassifier: Main classifier using VL models
    # ========================================================================
    class VLClassifier(nn.Module):
        """
        -
        
        ：Vision Encoder + VL Model (VisualBERT/LXMERT/UNITER/ViLBERT) + Classifier
        """
        
        def __init__(
            self,
            num_classes: int,
            model_type: str = 'visualbert',
            dropout: float = 0.1,
            freeze_backbone: bool = False
        ):
            """
            Args:
                num_classes: （）
                model_type: 'visualbert', 'lxmert', 'uniter',  'vilbert'
                dropout: Dropout
                freeze_backbone: VLbackbone
            """
            super().__init__()
            
            self.model_type = model_type
            self.num_classes = num_classes
            
            # Vision Encoder: ViT
            from transformers import ViTModel, ViTConfig
            vit_config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
            self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
            self.vision_hidden_size = vit_config.hidden_size  # 768
            
            # VL Model
            if model_type == 'visualbert':
                self.vl_model = VisualBertModel.from_pretrained(
                    'uclanlp/visualbert-vqa-coco-pre'
                )
                self.vl_hidden_size = self.vl_model.config.hidden_size
                
                # Visual projection: ViT (768) -> VisualBERT visual_embedding_dim
                self.visual_projection = nn.Linear(
                    self.vision_hidden_size,
                    self.vl_model.config.visual_embedding_dim
                )
                
            elif model_type == 'lxmert':
                self.vl_model = LxmertModel.from_pretrained(
                    'unc-nlp/lxmert-base-uncased'
                )
                self.vl_hidden_size = self.vl_model.config.hidden_size
                
                # Visual projection: ViT (768) -> LXMERT visual_feat_dim (2048)
                self.visual_projection = nn.Linear(
                    self.vision_hidden_size,
                    2048  # LXMERT expects 2048-dim visual features
                )
                
            elif model_type == 'uniter':
                # UNITER: ，modal type embeddings
                bert_config = BertConfig.from_pretrained('bert-base-uncased')
                self.vl_model = UNITERModel(bert_config)
                self.vl_hidden_size = bert_config.hidden_size
                
                # Visual projection: ViT (768) -> BERT hidden_size (768, )
                # UNITER，identity
                self.visual_projection = nn.Identity()
                
            elif model_type == 'vilbert':
                # ViLBERT: ，co-attention
                bert_config = BertConfig.from_pretrained('bert-base-uncased')
                self.vl_model = ViLBERTModel(bert_config)
                self.vl_hidden_size = bert_config.hidden_size
                
                # Visual projection: ViT (768) -> BERT hidden_size (768, )
                # ViLBERT，identity
                self.visual_projection = nn.Identity()
                
            else:
                raise ValueError(f"Unsupported model_type: {model_type}. "
                               f"Choose from: visualbert, lxmert, uniter, vilbert")
            
            # Tokenizer
            if model_type in ['visualbert', 'lxmert', 'uniter', 'vilbert']:
                self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            
            # Freeze backbone if requested
            if freeze_backbone:
                for param in self.vl_model.parameters():
                    param.requires_grad = False
                for param in self.vision_encoder.parameters():
                    param.requires_grad = False
            
            # Classification head
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.vl_hidden_size, num_classes)
        
        def forward(self, images, texts, return_features: bool = False):
            """
            
            
            Args:
                images: (batch_size, 3, 224, 224) tensor
                texts: List[str], batch_size
            
            Returns:
                logits: (batch_size, num_classes)
            """
            batch_size = images.size(0)
            device = images.device
            
            # 1. 
            vision_outputs = self.vision_encoder(pixel_values=images)
            vision_features = vision_outputs.last_hidden_state  # (batch, num_patches, 768)
            
            # Project to VL model's expected dimension
            vision_features = self.visual_projection(vision_features)  # (batch, num_patches, dim)
            
            # 2. Tokenize texts
            text_encoding = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            # 3. Forward through VL model
            visual_attention_mask = torch.ones(
                vision_features.size()[:-1],
                dtype=torch.long,
                device=device
            )
            
            if self.model_type == 'visualbert':
                # VisualBERT expects visual_embeds and visual_attention_mask
                vl_outputs = self.vl_model(
                    input_ids=text_encoding['input_ids'],
                    attention_mask=text_encoding['attention_mask'],
                    visual_embeds=vision_features,
                    visual_attention_mask=visual_attention_mask
                )
                pooled_output = vl_outputs.pooler_output  # (batch, hidden_size)
                
            elif self.model_type == 'lxmert':
                # LXMERT expects visual_feats and visual_pos
                # visual_pos: bounding box positions, use dummy positions for image patches
                num_patches = vision_features.size(1)
                visual_pos = torch.zeros(
                    batch_size, num_patches, 4,
                    dtype=torch.float,
                    device=device
                )
                
                vl_outputs = self.vl_model(
                    input_ids=text_encoding['input_ids'],
                    attention_mask=text_encoding['attention_mask'],
                    visual_feats=vision_features,
                    visual_pos=visual_pos
                )
                pooled_output = vl_outputs.pooled_output  # (batch, hidden_size)
                
            elif self.model_type == 'uniter':
                # UNITER expects visual_embeds and visual_attention_mask
                vl_outputs = self.vl_model(
                    input_ids=text_encoding['input_ids'],
                    attention_mask=text_encoding['attention_mask'],
                    visual_embeds=vision_features,
                    visual_attention_mask=visual_attention_mask
                )
                pooled_output = vl_outputs.pooler_output  # (batch, hidden_size)
                
            elif self.model_type == 'vilbert':
                # ViLBERT expects visual_embeds and visual_attention_mask
                vl_outputs = self.vl_model(
                    input_ids=text_encoding['input_ids'],
                    attention_mask=text_encoding['attention_mask'],
                    visual_embeds=vision_features,
                    visual_attention_mask=visual_attention_mask
                )
                pooled_output = vl_outputs.pooler_output  # (batch, hidden_size)
            
            # 4. Classification
            features = self.dropout(pooled_output)
            logits = self.classifier(features)
            
            if return_features:
                return logits, features
            return logits
    
    
    class MultiModalDataset(Dataset):
        """
        ：
        
        BERT RouterTextDatasetMobileNet RouterImageDataset
        """
        
        def __init__(
            self,
            texts: List[str],
            image_sources: Union[List[str], List[List[Dict]]],
            labels: Optional[np.ndarray] = None,
            image_base_dir: Optional[str] = None,
            is_soft_label: bool = False
        ):
            """
            Args:
                texts: 
                image_sources: assets
                labels:  (N,)   (N, K) 
                image_base_dir: 
                is_soft_label: 
            """
            self.texts = texts
            self.image_sources = image_sources
            self.labels = labels
            self.is_soft_label = is_soft_label
            self.verbose = 0
            
            # NOTE: (translated from Chinese)
            self.image_base_dir = Path(image_base_dir) if image_base_dir else Path('/root/LMUData')
            
            # （MobileNet）
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            # TSV
            self._tsv_cache = {}
        
        def __len__(self):
            return len(self.texts)
        
        def _load_image_from_tsv(self, asset: Dict) -> Image.Image:
            """TSVload（MobileNet）"""
            import pandas as pd
            
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
            image_source = self.image_sources[idx]
            
            # NOTE: (translated from Chinese)
            try:
                if isinstance(image_source, str):
                    # NOTE: (translated from Chinese)
                    img_path = Path(image_source)
                    if not img_path.is_absolute():
                        img_path = self.image_base_dir / img_path
                    
                    if not img_path.exists():
                        # Fallback: 
                        image = Image.new('RGB', (224, 224), color='black')
                    else:
                        image = Image.open(img_path).convert('RGB')
                        
                elif isinstance(image_source, list) and len(image_source) > 0:
                    # Assets
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
                # NOTE: (translated from Chinese)
                import traceback
                error_detail = f"idx={idx}, type(image_source)={type(image_source)}, "
                if isinstance(image_source, list) and len(image_source) > 0:
                    error_detail += f"asset={image_source[0]}, "
                error_detail += f"error={e}"
                warnings.warn(f"Failed to load image: {error_detail}")
                if self.verbose > 1:
                    traceback.print_exc()
                image = Image.new('RGB', (224, 224), color='black')
            
            # Transform
            image = self.transform(image)
            
            # Label
            if self.labels is not None:
                label = self.labels[idx]
                if self.is_soft_label:
                    label = torch.tensor(label, dtype=torch.float32)
                else:
                    label = torch.tensor(label, dtype=torch.long)
            else:
                label = torch.tensor(0, dtype=torch.long)
            
            return {
                'text': text,
                'image': image,
                'label': label
            }
else:
    VLClassifier = None
    MultiModalDataset = None


class VLCRouter(RouterBase):
    """
    VLC (Vision-Language Compositional) Router
    
    -：
    - VisualBERT: 
    - LXMERT: 
    - UNITER: （modal type embeddings）
    - ViLBERT: （co-attention）
    
    : p_θ(m|text,image) = softmax(VL_Model_θ(text, image))
    : ，
    : argmax_m p_θ(m|text,image)
    """
    
    def __init__(
        self,
        model_type: str = 'visualbert',
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        max_epochs: int = 5,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        random_state: int = 42,
        verbose: int = 1,
        device: Optional[str] = None,
        use_soft_labels: bool = False,
        train_lambda: float = 0.0,
        loss_type: str = 'softmax',
        sample_sample_weight: float = 0.0,
        sample_sample_temperature: float = 0.1,
        cost_scale: float = 100.0,
        enable_monitoring: bool = True,
        patience: int = 2,
        monitor_metric: str = 'rank_score',
        image_base_dir: Optional[str] = None
    ):
        """
        Args:
            model_type: 'visualbert', 'lxmert', 'uniter',  'vilbert'
            dropout: Dropout
            freeze_backbone: VLbackbone
            learning_rate: 
            batch_size: 
            max_epochs: 
            warmup_ratio: 
            weight_decay: 
            random_state: 
            verbose: 
            device: 
            use_soft_labels: 
            train_lambda: lambda
            enable_monitoring: 
            patience: 
            monitor_metric: 
            image_base_dir: 
        """
        if not HAS_TORCH:
            raise ImportError("Please installPyTorchTransformers")
        
        self.model_type = model_type
        self.dropout = dropout
        self.freeze_backbone = freeze_backbone
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.verbose = verbose
        self.use_soft_labels = use_soft_labels
        self.train_lambda = train_lambda
        if loss_type not in {'softmax', 'siglip_relevance'}:
            raise ValueError(f"loss_type must be softmax or siglip_relevance; got {loss_type}")
        self.loss_type = loss_type
        self.sample_sample_weight = float(sample_sample_weight)
        self.sample_sample_temperature = float(sample_sample_temperature)
        self.cost_scale = float(cost_scale)
        self.enable_monitoring = enable_monitoring
        self.patience = patience
        self.monitor_metric = monitor_metric
        # Rank score configuration (used for monitoring on dev)
        self._rank_score_cmin = None
        self._rank_score_cmax = None
        self._rank_score_beta = 0.1
        self.image_base_dir = image_base_dir or '/root/LMUData'
        
        # NOTE: (translated from Chinese)
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # NOTE: (translated from Chinese)
        self.model = None
        self.model_mapping = None
        self.reverse_mapping = None
        
        # NOTE: (translated from Chinese)
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
    
    def save(self, path: str):
        """save"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # NOTE: (translated from Chinese)
        model_backup = self.model
        self.model = None
        
        # router
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # PyTorch
        if model_backup is not None:
            model_path = path.parent / f"{path.stem}_model.pt"
            torch.save({
                'model_state_dict': model_backup.state_dict(),
                'model_type': self.model_type,
                'num_classes': model_backup.num_classes,
                'dropout': self.dropout,
                'freeze_backbone': self.freeze_backbone
            }, model_path)
            
            if self.verbose > 0:
                print(f"  Model saved: {model_path}")
        
        self.model = model_backup
    
    @staticmethod
    def load(path: str):
        """load"""
        path = Path(path)
        
        # router
        with open(path, 'rb') as f:
            router = pickle.load(f)
        
        # PyTorch
        model_path = path.parent / f"{path.stem}_model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=router.device, weights_only=False)
            
            # NOTE: (translated from Chinese)
            router.model = VLClassifier(
                num_classes=checkpoint['num_classes'],
                model_type=checkpoint['model_type'],
                dropout=checkpoint.get('dropout', 0.1),
                freeze_backbone=checkpoint.get('freeze_backbone', False)
            ).to(router.device)
            
            # NOTE: (translated from Chinese)
            router.model.load_state_dict(checkpoint['model_state_dict'])
            
            if router.verbose > 0:
                print(f"   {model_path} load")
        
        return router
    
    @staticmethod
    def _derive_labels(
        Y: np.ndarray,
        C: np.ndarray,
        use_soft_labels: bool = False,
        soft_label_lambda: float = 0.0,
        cost_scale: float = 100.0,
        loss_type: str = 'softmax'
    ):
        """（Hard labelSoft label）"""
        if loss_type == 'siglip_relevance':
            from routers.utils.soft_targets import build_relevance_targets
            T = build_relevance_targets(
                Y, C,
                lam=soft_label_lambda,
                cost_scale=cost_scale,
                fallback='zeros',
            )
            return T.astype(np.float32), np.arange(Y.shape[0])
        if not use_soft_labels:
            # NOTE: (translated from Chinese)
            N, K = Y.shape
            labels = []
            valid_indices = []
            
            for i in range(N):
                correct = np.where(Y[i] == 1)[0]
                if len(correct) > 0:
                    costs = C[i, correct]
                    best_idx = correct[np.argmin(costs)]
                    labels.append(best_idx)
                    valid_indices.append(i)
            
            return np.array(labels), np.array(valid_indices)
        else:
            # NOTE: (translated from Chinese)
            from routers.utils.soft_targets import build_soft_targets
            
            T = build_soft_targets(
                Y, C,
                lam=soft_label_lambda,
                scheme='exp',
                fallback='cheapest',
                cost_scale=cost_scale
            )
            
            # NOTE: (translated from Chinese)
            has_correct = (Y.sum(axis=1) > 0)
            valid_indices = np.where(has_correct)[0]
            
            return T, valid_indices
    
    @staticmethod
    def _create_labels_no_filter(
        Y: np.ndarray,
        C: np.ndarray,
        use_soft_labels: bool = False,
        soft_label_lambda: float = 0.0,
        cost_scale: float = 100.0,
        loss_type: str = 'softmax'
    ):
        """（，dev/testevaluation）
        
        ，
        """
        N, K = Y.shape

        if loss_type == 'siglip_relevance':
            from routers.utils.soft_targets import build_relevance_targets
            return build_relevance_targets(
                Y, C,
                lam=soft_label_lambda,
                cost_scale=cost_scale,
                fallback='zeros',
            ).astype(np.float32)
        
        if not use_soft_labels:
            # ：，
            labels = np.zeros(N, dtype=np.int64)
            
            for i in range(N):
                correct = np.where(Y[i] == 1)[0]
                if len(correct) > 0:
                    # NOTE: (translated from Chinese)
                    costs = C[i, correct]
                    best_idx = correct[np.argmin(costs)]
                    labels[i] = best_idx
                else:
                    # ：（fallback）
                    labels[i] = np.argmin(C[i])
            
            return labels
        else:
            # ：build_soft_targets，fallback
            from routers.utils.soft_targets import build_soft_targets
            
            T = build_soft_targets(
                Y, C,
                lam=soft_label_lambda,
                scheme='exp',
                fallback='cheapest',
                cost_scale=cost_scale
            )
            
            return T
    
    def _extract_data_from_meta(self, meta):
        """"""
        # NOTE: (translated from Chinese)
        if 'text' in meta.columns:
            texts = meta['text'].tolist()
            text_field = 'text'
        elif 'question' in meta.columns:
            texts = meta['question'].tolist()
            text_field = 'question'
        elif 'prompt' in meta.columns:
            texts = meta['prompt'].tolist()
            text_field = 'prompt'
        else:
            raise ValueError("meta'text', 'question''prompt'")
        
        # NOTE: (translated from Chinese)
        if 'assets' in meta.columns:
            image_sources = meta['assets'].tolist()
            image_field = 'assets'
        elif 'image' in meta.columns:
            image_sources = meta['image'].tolist()
            image_field = 'image'
        elif 'image_path' in meta.columns:
            image_sources = meta['image_path'].tolist()
            image_field = 'image_path'
        else:
            # NOTE: (translated from Chinese)
            warnings.warn("meta，")
            image_sources = [[]] * len(texts)
            image_field = 'none'
        
        # NOTE: (translated from Chinese)
        if self.verbose > 1:
            non_empty_texts = len([t for t in texts if t and str(t).strip()])
            non_empty_images = len([img for img in image_sources if img])
            print(f"    📊 :")
            print(f"       : '{text_field}', : {non_empty_texts}/{len(texts)}")
            print(f"       : '{image_field}', : {non_empty_images}/{len(image_sources)}")
            if non_empty_texts > 0:
                print(f"       : {str(texts[0])[:100]}...")
            if non_empty_images > 0:
                first_img_idx = next((i for i, img in enumerate(image_sources) if img), None)
                if first_img_idx is not None:
                    print(f"       : {image_sources[first_img_idx]}")
        
        return texts, image_sources
    
    def fit(
        self,
        Y: np.ndarray,
        C: np.ndarray,
        meta,
        model_mapping: Optional[Dict[int, str]] = None,
        costs: Optional[np.ndarray] = None,
        Y_dev: Optional[np.ndarray] = None,
        C_dev: Optional[np.ndarray] = None,
        meta_dev = None,
        monitor_output_dir: Optional[Path] = None,
        **kwargs
    ):
        """trainingVLC Router"""
        if self.verbose > 0:
            print(f"\n🎓 trainingVLC Router ({self.model_type.upper()})...")
        
        # 1. 
        texts_train, images_train = self._extract_data_from_meta(meta)
        
        # 2. 
        y_or_T, valid_indices = self._derive_labels(
            Y, C,
            use_soft_labels=self.use_soft_labels,
            soft_label_lambda=self.train_lambda,
            cost_scale=self.cost_scale,
            loss_type=self.loss_type,
        )
        
        # NOTE: (translated from Chinese)
        if len(valid_indices) < len(Y):
            if self.verbose > 0:
                print(f"  Filtering training samples: {len(Y)} -> {len(valid_indices)}")
            texts_train = [texts_train[i] for i in valid_indices]
            images_train = [images_train[i] for i in valid_indices]
            Y = Y[valid_indices]
            C = C[valid_indices]
            if self.use_soft_labels:
                y_or_T = y_or_T[valid_indices]
        
        # 3. 
        num_classes = Y.shape[1]
        
        if self.verbose > 0:
            print(f"  load: {self.model_type}")
        
        self.model = VLClassifier(
            num_classes=num_classes,
            model_type=self.model_type,
            dropout=self.dropout,
            freeze_backbone=self.freeze_backbone
        ).to(self.device)
        
        if self.verbose > 0:
            print(f"  ✓ load，: {self.device}")
        
        # 4. 
        if model_mapping is None:
            model_mapping = {i: f'model_{i}' for i in range(num_classes)}
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}

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
        
        # 5. 
        train_dataset = MultiModalDataset(
            texts=texts_train,
            image_sources=images_train,
            labels=y_or_T,
            image_base_dir=self.image_base_dir,
            is_soft_label=(self.use_soft_labels or self.loss_type == 'siglip_relevance')
        )
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            images = torch.stack([item['image'] for item in batch])
            labels = torch.stack([item['label'] for item in batch])
            return texts, images, labels
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        # shuffletrain loader
        train_eval_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # evaluationshuffle
            num_workers=0,
            collate_fn=collate_fn
        )
        
        if self.verbose > 0:
            print(f"  training: {len(train_dataset)}, : {len(train_loader)}")
        
        # NOTE: (translated from Chinese)
        if self.verbose > 1:
            if not self.use_soft_labels:
                train_label_dist = np.bincount(y_or_T, minlength=num_classes)
                print(f"  training: {train_label_dist}")
                print(f"  training: {np.unique(y_or_T)} ({len(np.unique(y_or_T))})")
            else:
                train_label_argmax = np.argmax(y_or_T, axis=1)
                train_label_dist = np.bincount(train_label_argmax, minlength=num_classes)
                print(f"  training(argmax): {train_label_dist}")
                print(f"  training: {np.unique(train_label_argmax)} ({len(np.unique(train_label_argmax))})")
        
        # 6. dev（，dev）
        dev_loader = None
        if self.enable_monitoring and Y_dev is not None and meta_dev is not None:
            texts_dev, images_dev = self._extract_data_from_meta(meta_dev)
            
            # dev（，）
            # ：DataLoader，Y_dev, C_dev
            y_or_T_dev = self._create_labels_no_filter(
                Y_dev, C_dev,
                use_soft_labels=self.use_soft_labels,
                soft_label_lambda=self.train_lambda,
                cost_scale=self.cost_scale,
                loss_type=self.loss_type,
            )
            
            # dataset
            dev_dataset = MultiModalDataset(
                texts=texts_dev,
                image_sources=images_dev,
                labels=y_or_T_dev,
                image_base_dir=self.image_base_dir,
                is_soft_label=(self.use_soft_labels or self.loss_type == 'siglip_relevance')
            )
            
            dev_loader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            if self.verbose > 0:
                print(f"  Dev set: {len(dev_dataset)} samples（，）")
        
        # 7. 
        self._train(
            train_loader=train_loader,
            train_eval_loader=train_eval_loader,
            dev_loader=dev_loader,
            Y_train=Y,
            C_train=C,
            Y_dev=Y_dev if self.enable_monitoring else None,
            C_dev=C_dev if self.enable_monitoring else None,
            monitor_output_dir=monitor_output_dir
        )
        
        if self.verbose > 0:
            print(f"\n✓ VLC RouterTraining complete")
        
        return self
    
    def _train(
        self,
        train_loader: DataLoader,
        train_eval_loader: DataLoader,
        dev_loader: Optional[DataLoader] = None,
        Y_train: Optional[np.ndarray] = None,
        C_train: Optional[np.ndarray] = None,
        Y_dev: Optional[np.ndarray] = None,
        C_dev: Optional[np.ndarray] = None,
        monitor_output_dir: Optional[Path] = None
    ):
        """training"""
        # NOTE: (translated from Chinese)
        num_training_steps = len(train_loader) * self.max_epochs
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # NOTE: (translated from Chinese)
        if self.loss_type == 'siglip_relevance':
            criterion = nn.BCEWithLogitsLoss()
        elif self.use_soft_labels:
            def soft_ce_loss(logits, soft_targets):
                log_probs = F.log_softmax(logits, dim=-1)
                return -torch.sum(soft_targets * log_probs, dim=-1).mean()
            criterion = soft_ce_loss
        else:
            criterion = nn.CrossEntropyLoss()
        
        # NOTE: (translated from Chinese)
        monitor = None
        if self.enable_monitoring and dev_loader is not None:
            from routers.utils.training_monitor import TrainingMonitor
            if monitor_output_dir is None:
                import tempfile
                monitor_output_dir = Path(tempfile.mkdtemp(prefix='vlc_router_'))
            
            monitor = TrainingMonitor(
                output_dir=monitor_output_dir,
                metric=self.monitor_metric,
                patience=self.patience,
                maximize=True if self.monitor_metric in ['accuracy', 'rank_score'] else False,
                save_checkpoints=True,
                verbose=self.verbose
            )
            monitor.start_training()
        
        # NOTE: (translated from Chinese)
        if self.verbose > 0:
            print(f"\n  training...")
            print(f"    Batch size: {self.batch_size}, Max epochs: {self.max_epochs}")
            print(f"    Learning rate: {self.learning_rate}, : {num_warmup_steps}")
        
        try:
            from tqdm import tqdm
            use_tqdm = self.verbose > 0
        except ImportError:
            use_tqdm = False
        
        for epoch in range(1, self.max_epochs + 1):
            # NOTE: (translated from Chinese)
            self.model.train()
            total_loss = 0.0
            total_loss_main = 0.0
            total_loss_ss = 0.0
            num_batches = 0
            
            if use_tqdm:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.max_epochs}",
                           leave=False, ncols=100)
                batch_iterator = pbar
            else:
                batch_iterator = train_loader
            
            for batch_idx, (texts, images, labels) in enumerate(batch_iterator):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # NOTE: (translated from Chinese)
                if self.sample_sample_weight > 0:
                    logits, query_features = self.model(images, texts, return_features=True)
                else:
                    logits = self.model(images, texts)
                    query_features = None
                
                # NOTE: (translated from Chinese)
                if self.loss_type == 'siglip_relevance':
                    main_loss = criterion(logits, labels.float().clamp(0.0, 1.0))
                elif self.use_soft_labels:
                    main_loss = criterion(logits, labels)
                else:
                    main_loss = criterion(logits, labels)
                ss_loss = self._sample_sample_loss(query_features, labels) if self.sample_sample_weight > 0 else logits.new_tensor(0.0)
                loss = main_loss + self.sample_sample_weight * ss_loss
                
                # （batch）
                if self.verbose > 1 and (batch_idx == 0 or batch_idx == len(train_loader) - 1):
                    preds = torch.argmax(logits, dim=-1)
                    print(f"\n  🔍 Epoch {epoch} Batch {batch_idx}:")
                    print(f"     Labels: {labels[:5].cpu().numpy()}")
                    print(f"     Preds:  {preds[:5].cpu().numpy()}")
                    print(f"     Logits[0]: {logits[0][:5].detach().cpu().numpy()}")
                    print(f"     Loss: {loss.item():.4f}")
                
                # NOTE: (translated from Chinese)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                total_loss_main += main_loss.item()
                total_loss_ss += ss_loss.item()
                num_batches += 1
                
                if use_tqdm:
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'main': f'{main_loss.item():.4f}',
                        'ss': f'{ss_loss.item():.4f}',
                    })
            
            if use_tqdm:
                pbar.close()
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_main_loss = total_loss_main / num_batches if num_batches > 0 else 0.0
            avg_ss_loss = total_loss_ss / num_batches if num_batches > 0 else 0.0
            
            # NOTE: (translated from Chinese)
            if monitor and dev_loader is not None:
                #  - shuffletrain_eval_loader
                if self.verbose > 1:
                    print(f"\n  📊 Epoch {epoch} evaluation:")
                    
                train_acc, train_cost, train_preds = self._evaluate(
                    train_eval_loader, Y_data=Y_train, C_data=C_train, 
                    return_preds=True, verbose=(self.verbose > 1)
                )
                dev_acc, dev_cost, dev_preds = self._evaluate(
                    dev_loader, Y_data=Y_dev, C_data=C_dev,
                    return_preds=True, verbose=(self.verbose > 1)
                )
                
                # NOTE: (translated from Chinese)
                if self.verbose > 1:
                    print(f"    Train - : {len(np.unique(train_preds))}/{Y_train.shape[1]} ")
                    print(f"    Train - : {np.bincount(train_preds, minlength=Y_train.shape[1])}")
                    print(f"    Dev   - : {len(np.unique(dev_preds))}/{Y_dev.shape[1]} ")
                    print(f"    Dev   - : {np.bincount(dev_preds, minlength=Y_dev.shape[1])}")
                
                train_metrics = {
                    'loss': float(avg_loss),
                    'loss_main': float(avg_main_loss),
                    'loss_ss': float(avg_ss_loss),
                    'accuracy': float(train_acc),
                    'avg_cost': float(train_cost)
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
                should_stop = monitor.log_epoch(epoch, train_metrics, dev_metrics, self)
                
                if self.verbose > 0:
                    status = "🌟 [BEST]" if monitor.best_epoch == epoch else "         "
                    print(f"  {status} Epoch {epoch:3d}/{self.max_epochs}: "
                          f"loss={avg_loss:.4f}, "
                          f"train_acc={train_acc:.4f}, "
                          f"dev_acc={dev_acc:.4f}, "
                          f"dev_cost=${dev_cost:.6f}")
                
                if should_stop:
                    if self.verbose > 0:
                        print(f"\n  ⏹️  early stopping！epoch: {monitor.best_epoch}")
                    break
            else:
                if self.verbose > 0:
                    print(f"  Epoch {epoch:3d}/{self.max_epochs}: loss={avg_loss:.4f}")
        
        # NOTE: (translated from Chinese)
        if monitor and monitor.best_model_path:
            if self.verbose > 0:
                print(f"\n  loadbest model...")
            try:
                loaded_router = VLCRouter.load(monitor.best_model_path)
                # Ensure weights are restored on the active device for evaluation
                self.model = loaded_router.model.to(self.device) if loaded_router.model is not None else self.model
                self.model.eval()
                if self.verbose > 0:
                    print(f"  ✓ best modelload")
            except Exception as e:
                if self.verbose > 0:
                    print(f"  ⚠️  Failed to loadbest model: {e}")

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
    
    def _evaluate(self, data_loader, Y_data=None, C_data=None, return_preds=False, verbose=False):
        """evaluationAccuracy
        
        Args:
            data_loader: 
            Y_data:  (N, K)
            C_data:  (N, K)
            return_preds: 
            verbose: 
        """
        self.model.eval()
        
        all_preds = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, (texts, images, labels) in enumerate(data_loader):
                images = images.to(self.device)
                logits = self.model(images, texts)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_logits.append(logits.cpu().numpy())
                all_labels.extend(labels.cpu().numpy() if not self.use_soft_labels else torch.argmax(labels, dim=-1).cpu().numpy())
                
                # batch
                if verbose and batch_idx == 0:
                    batch_labels = labels.cpu().numpy() if not self.use_soft_labels else torch.argmax(labels, dim=-1).cpu().numpy()
                    print(f"\n      📋 Batch 0 :")
                    print(f"         samples: {len(preds)}")
                    print(f"         : {batch_labels[:10]}")
                    print(f"         : {preds[:10]}")
                    print(f"         Logits[0]: {logits[0].cpu().numpy()[:5]}")
                    print(f"         Logits[1]: {logits[1].cpu().numpy()[:5] if len(logits) > 1 else 'N/A'}")
                    print(f"         Logits: ={logits.mean().item():.4f}, ={logits.std().item():.4f}")
                    print(f"         Logits: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
        
        all_preds = np.array(all_preds)
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.array(all_labels)
        
        if verbose:
            print(f"\n      📊 :")
            print(f"         samples: {len(all_preds)}")
            print(f"         : {np.unique(all_preds)} ({len(np.unique(all_preds))})")
            print(f"         : {np.unique(all_labels)} ({len(np.unique(all_labels))})")
            print(f"         : {np.bincount(all_preds, minlength=Y_data.shape[1] if Y_data is not None else 17)}")
            print(f"         : {np.bincount(all_labels, minlength=Y_data.shape[1] if Y_data is not None else 17)}")
            print(f"         Logits: ={all_logits.mean():.4f}, ={all_logits.std():.4f}")
            
            # 20
            print(f"\n      🔍 20samples:")
            for i in range(min(20, len(all_preds))):
                correct_mark = "✓" if all_preds[i] == all_labels[i] else "✗"
                if Y_data is not None and i < len(Y_data):
                    is_valid = Y_data[i, all_preds[i]]
                    valid_mark = "✓" if is_valid else "✗"
                    print(f"         [{i:3d}] Label={all_labels[i]:2d}, Pred={all_preds[i]:2d} {correct_mark}, Valid={valid_mark}, Logits_max={all_logits[i].max():.3f}")
                else:
                    print(f"         [{i:3d}] Label={all_labels[i]:2d}, Pred={all_preds[i]:2d} {correct_mark}, Logits_max={all_logits[i].max():.3f}")
        
        # NOTE: (translated from Chinese)
        if Y_data is not None:
            if len(all_preds) != len(Y_data):
                if verbose:
                    print(f"\n      ⚠️ : preds={len(all_preds)}, Y_data={len(Y_data)}")
                accuracy = 0.0
            else:
                #  = （Y_data[i, pred] == 1）
                accuracy = np.mean(Y_data[np.arange(len(all_preds)), all_preds])
                if verbose:
                    # NOTE: (translated from Chinese)
                    correct_count = np.sum(Y_data[np.arange(len(all_preds)), all_preds])
                    print(f"\n      ✅ Accuracy: {correct_count}/{len(all_preds)} = {accuracy:.4f}")
        else:
            accuracy = 0.0
        
        # NOTE: (translated from Chinese)
        if C_data is not None:
            if len(all_preds) != len(C_data):
                avg_cost = 0.0
            else:
                avg_cost = np.mean(C_data[np.arange(len(all_preds)), all_preds])
        else:
            avg_cost = 0.0
        
        if return_preds:
            return accuracy, avg_cost, all_preds
        else:
            return accuracy, avg_cost
    
    def predict(
        self,
        meta,
        X: Optional[np.ndarray] = None,
        batch_size=None,
        num_workers=4,
        show_progress=False,
        **kwargs
    ) -> np.ndarray:
        """
        
        
        Args:
            meta:  DataFrame
            X: （）
            batch_size: batch size（batch_size4，）
            num_workers: （4，0）
            show_progress: 
        
        Returns:
             (N)
        """
        if self.model is None:
            raise ValueError("training")
        
        # batch_size
        if batch_size is None:
            batch_size = self.batch_size * 4
        
        self.model.eval()
        
        # NOTE: (translated from Chinese)
        texts, image_sources = self._extract_data_from_meta(meta)
        
        # NOTE: (translated from Chinese)
        dataset = MultiModalDataset(
            texts=texts,
            image_sources=image_sources,
            labels=None,
            image_base_dir=self.image_base_dir,
            is_soft_label=False
        )
        
        def collate_fn(batch):
            texts = [item['text'] for item in batch]
            images = torch.stack([item['image'] for item in batch])
            return texts, images
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
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
        all_preds = []
        
        with torch.no_grad():
            for texts, images in data_loader:
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images, texts)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                all_preds.extend(preds)
        
        return np.array(all_preds)
    
    def name(self) -> str:
        """"""
        return f"VLCRouter_{self.model_type}"
