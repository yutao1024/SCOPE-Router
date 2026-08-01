#!/usr/bin/env python3
"""
Online SCOPE-Router.

This variant trains the text/image encoders online, then keeps the same
Query/profile projection routing head used by the frozen-embedding SCOPE-Router.
"""

import base64
from io import BytesIO
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from routers.common import RouterBase

try:
    import pandas as pd
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.data.distributed import DistributedSampler
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor
    from PIL import Image
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    pd = None
    torch = None
    nn = None
    F = None
    DataLoader = None
    Dataset = object
    DistributedSampler = None
    dist = None
    DDP = None
    AutoModel = None
    AutoProcessor = None
    AutoTokenizer = None
    CLIPModel = None
    CLIPProcessor = None
    Image = None


class NumpyStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=0, keepdims=True).astype(np.float32)
        self.scale_ = X.std(axis=0, keepdims=True).astype(np.float32)
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


if HAS_DEPS:
    class TextFeatureEncoder(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.model = AutoModel.from_pretrained(model_name)
            self.output_dim = int(getattr(self.model.config, "hidden_size"))

        def forward(self, input_ids, attention_mask):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).type_as(token_embeddings)
            return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)


    class VisionFeatureEncoder(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            if "clip" in model_name.lower():
                self.model = CLIPModel.from_pretrained(model_name)
            else:
                self.model = AutoModel.from_pretrained(model_name)
            config = self.model.config
            if hasattr(config, "projection_dim"):
                self.output_dim = int(config.projection_dim)
            elif hasattr(config, "hidden_size"):
                self.output_dim = int(config.hidden_size)
            elif hasattr(config, "vision_config"):
                self.output_dim = int(getattr(config.vision_config, "hidden_size"))
            else:
                raise ValueError(f"Could not infer vision encoder output dimension for {model_name}")

        def forward(self, pixel_values):
            if hasattr(self.model, "get_image_features"):
                return self.model.get_image_features(pixel_values=pixel_values)
            outputs = self.model(pixel_values=pixel_values)
            if getattr(outputs, "pooler_output", None) is not None:
                return outputs.pooler_output
            return outputs.last_hidden_state[:, 0]


    class ProjectionMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2, dropout: float = 0.1):
            super().__init__()
            layers = []
            dim = input_dim
            for _ in range(max(0, num_layers - 1)):
                layers.append(nn.Linear(dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                dim = hidden_dim
            layers.append(nn.Linear(dim, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)


    class RawQueryDataset(Dataset):
        def __init__(self, meta, targets, tokenizer, image_processor, dataset_dir: str, max_length: int = 128):
            self.meta = meta.reset_index(drop=True)
            self.targets = targets
            self.tokenizer = tokenizer
            self.image_processor = image_processor
            self.dataset_dir = Path(dataset_dir)
            self.max_length = max_length
            self._tsv_cache = {}

        def __len__(self):
            return len(self.meta)

        def _load_tsv(self, tsv_file: str):
            path = Path(tsv_file)
            if not path.is_absolute():
                path = self.dataset_dir / path
            key = str(path)
            if key not in self._tsv_cache:
                df = pd.read_csv(path, sep="\t")
                if "image" in df.columns and "index" in df.columns:
                    df["image"] = df["image"].astype(str)
                    image_map = {str(idx): img for idx, img in zip(df["index"], df["image"])}
                    for idx, value in list(image_map.items()):
                        if len(value) <= 64 and value in image_map and len(image_map[value]) > 64:
                            image_map[idx] = image_map[value]
                    for idx, value in image_map.items():
                        df.loc[df["index"].astype(str) == idx, "image"] = value
                self._tsv_cache[key] = df
            return self._tsv_cache[key]

        def _blank_image(self):
            return Image.new("RGB", (224, 224), color="white")

        def _load_image_from_asset(self, asset):
            if not isinstance(asset, dict):
                return self._blank_image()
            try:
                if asset.get("type") == "image_tsv":
                    tsv_file = asset.get("tsv_file") or asset.get("path") or asset.get("uri")
                    index = int(asset.get("index", asset.get("lineno", 0)))
                    df = self._load_tsv(tsv_file)
                    if index >= len(df):
                        return self._blank_image()
                    row = df.iloc[index]
                    img_str = row["image"] if "image" in df.columns else row.iloc[1]
                    return Image.open(BytesIO(base64.b64decode(img_str))).convert("RGB")

                img_path = asset.get("path") or asset.get("uri")
                if img_path:
                    path = Path(img_path)
                    if not path.is_absolute():
                        path = self.dataset_dir / path
                    if path.exists():
                        return Image.open(path).convert("RGB")
            except Exception:
                return self._blank_image()
            return self._blank_image()

        def __getitem__(self, idx):
            row = self.meta.iloc[idx]
            text = row.get("text", row.get("prompt", row.get("question", "")))
            if not text:
                text = "No text available"

            assets = row.get("assets", [])
            image = self._blank_image()
            if isinstance(assets, list) and len(assets) > 0:
                image = self._load_image_from_asset(assets[0])

            text_inputs = self.tokenizer(
                str(text),
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            image_inputs = self.image_processor(images=image, return_tensors="pt")
            item = {
                "input_ids": text_inputs["input_ids"].squeeze(0),
                "attention_mask": text_inputs["attention_mask"].squeeze(0),
                "pixel_values": image_inputs["pixel_values"].squeeze(0),
                "target": torch.tensor(self.targets[idx]),
            }
            if item["target"].dtype == torch.float64:
                item["target"] = item["target"].float()
            return item
else:
    ProjectionMLP = None
    RawQueryDataset = None


class OnlineScopeRouter(RouterBase):
    def __init__(
        self,
        profile_path: str,
        dataset_dir: str = ".",
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base",
        fusion_method: str = "normalize_concat",
        text_weight: float = 0.5,
        embedding_dim: int = 256,
        query_hidden_dim: int = 512,
        query_layers: int = 2,
        profile_hidden_dim: int = 512,
        profile_layers: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 2e-5,
        profile_learning_rate: Optional[float] = None,
        weight_decay: float = 1e-4,
        batch_size: int = 16,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        max_iter: int = 5,
        max_length: int = 128,
        temperature: float = 0.07,
        loss_type: str = "crm",
        crm_target: str = "relevance",
        train_lambda: float = 10.0,
        use_soft_labels: bool = True,
        cost_scale: float = 100.0,
        patience: int = 3,
        monitor_metric: str = "rank_score",
        query_profile_update: str = "static",
        profile_refresh_batch_size: Optional[int] = None,
        differentiable_calib_size: Optional[int] = None,
        device: Optional[str] = None,
        multi_gpu: bool = False,
        distributed: bool = False,
        local_rank: int = 0,
        rank: int = 0,
        world_size: int = 1,
        random_state: int = 42,
        verbose: int = 1,
    ):
        if not HAS_DEPS:
            raise ImportError("OnlineScopeRouter requires torch, transformers, pandas, and pillow")
        if loss_type == "clip-relevance":
            loss_type = "clip_relevance"
        if loss_type not in {"softmax", "clip", "clip_relevance", "crm"}:
            raise ValueError(f"loss_type must be one of softmax, clip, clip_relevance, crm; got {loss_type}")
        if crm_target not in {"soft", "y", "relevance"}:
            raise ValueError(f"crm_target must be one of soft, y, relevance; got {crm_target}")
        if query_profile_update not in {"static", "epoch_refresh", "differentiable"}:
            raise ValueError(
                "query_profile_update must be one of static, epoch_refresh, differentiable; "
                f"got {query_profile_update}"
            )

        self.profile_path = str(profile_path)
        self.dataset_dir = str(dataset_dir)
        self.text_encoder_name = text_encoder
        self.vision_encoder_name = vision_encoder
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.embedding_dim = int(embedding_dim)
        self.query_hidden_dim = int(query_hidden_dim)
        self.query_layers = int(query_layers)
        self.profile_hidden_dim = profile_hidden_dim
        self.profile_layers = profile_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.profile_learning_rate = profile_learning_rate if profile_learning_rate is not None else learning_rate * 10.0
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.max_iter = max_iter
        self.max_length = max_length
        self.temperature = temperature
        self.loss_type = loss_type
        self.crm_target = crm_target
        self.train_lambda = train_lambda
        self.use_soft_labels = use_soft_labels
        self.cost_scale = cost_scale
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.query_profile_update = query_profile_update
        self.profile_refresh_batch_size = profile_refresh_batch_size
        self.differentiable_calib_size = differentiable_calib_size
        self.multi_gpu = bool(multi_gpu)
        self.distributed = bool(distributed)
        self.local_rank = int(local_rank)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.random_state = random_state
        self.verbose = verbose
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.distributed and torch.cuda.is_available():
            self.device = f"cuda:{self.local_rank}"
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.tokenizer = None
        self.image_processor = None
        self.text_model = None
        self.vision_model = None
        self.query_encoder = None
        self.profile_encoder = None
        self.profile_scaler = None
        self.model_profile = None
        self.model_names = None
        self.model_mapping = None
        self.reverse_mapping = None
        self.costs = None
        self.best_epoch = None
        self.best_dev_metrics = None
        self._last_profile_dim = None
        self._query_dim = None
        self._rank_score_cmin = None
        self._rank_score_cmax = None
        self._rank_score_beta = 0.1
        self._force_unwrapped_forward = False
        self._profile_data = None
        self._behavior_profile_raw = None
        self._calib_meta = None
        self._calib_Y = None
        self._calib_cost_norm = None
        self._calib_Y_tensor = None
        self._calib_cost_norm_tensor = None
        self._behavior_profile_tensor = None
        self._profile_scaler_mean_tensor = None
        self._profile_scaler_scale_tensor = None

    def _unwrap_model(self, model):
        return model.module if isinstance(model, (nn.DataParallel, DDP)) else model

    def _model_state_dict(self, model):
        return self._unwrap_model(model).state_dict()

    def _wrap_distributed_if_needed(self):
        if not self.distributed:
            return
        device_ids = [self.local_rank] if str(self.device).startswith("cuda") else None
        output_device = self.local_rank if device_ids else None
        if self.text_model is not None and not isinstance(self.text_model, DDP):
            self.text_model = DDP(
                self.text_model,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=True,
            )
        if self.vision_model is not None and not isinstance(self.vision_model, DDP):
            self.vision_model = DDP(
                self.vision_model,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=True,
            )
        if self.query_encoder is not None and not isinstance(self.query_encoder, DDP):
            self.query_encoder = DDP(
                self.query_encoder,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=True,
            )
        if self.profile_encoder is not None and not isinstance(self.profile_encoder, DDP):
            self.profile_encoder = DDP(
                self.profile_encoder,
                device_ids=device_ids,
                output_device=output_device,
                find_unused_parameters=True,
            )
        if self.verbose > 0 and self.rank == 0:
            print(f"  Using DDP: rank={self.rank}/{self.world_size}, local_rank={self.local_rank}, device={self.device}")

    def _distributed_barrier(self):
        if not self.distributed:
            return
        if str(self.device).startswith("cuda"):
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def _load_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(self.profile_path, allow_pickle=False)
        if "model_profile" not in data:
            raise ValueError(f"{self.profile_path} must contain model_profile")
        model_profile = data["model_profile"].astype(np.float32)
        if "model_names" in data:
            model_names = data["model_names"].astype(str)
        else:
            model_names = np.array([f"model_{i}" for i in range(model_profile.shape[0])], dtype=str)
        self._profile_data = {key: data[key] for key in data.files}
        return model_profile, model_names

    def _setup_dynamic_query_profile(self, raw_profile: np.ndarray, all_meta):
        if self.query_profile_update == "static":
            return
        required = {"behavior_profile", "Y_calib", "cost_norm_calib", "sample_ids"}
        missing = sorted(required - set(self._profile_data or {}))
        if missing:
            raise ValueError(
                f"{self.query_profile_update} requires a query-aware calibration profile with keys: "
                f"{', '.join(sorted(required))}; missing {', '.join(missing)}"
            )
        if all_meta is None:
            raise ValueError(
                f"{self.query_profile_update} requires calibration_meta=full_meta when calling fit()"
            )

        self._behavior_profile_raw = self._profile_data["behavior_profile"].astype(np.float32)
        self._calib_Y = self._profile_data["Y_calib"].astype(np.float32)
        self._calib_cost_norm = self._profile_data["cost_norm_calib"].astype(np.float32)
        sample_ids = self._profile_data["sample_ids"].astype(str).tolist()
        meta_by_id = all_meta.copy()
        meta_by_id["sample_id"] = meta_by_id["sample_id"].astype(str)
        meta_by_id = meta_by_id.set_index("sample_id", drop=False)
        missing_ids = [sid for sid in sample_ids if sid not in meta_by_id.index]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(f"Missing calibration metadata for {len(missing_ids)} samples: {preview}")
        self._calib_meta = meta_by_id.loc[sample_ids].copy().reset_index(drop=True)

        expected_dim = self._behavior_profile_raw.shape[1] + 3 * self._query_dim
        if raw_profile.shape[1] != expected_dim:
            raise ValueError(
                f"{self.query_profile_update} expects profile_dim=behavior_dim+3*query_dim="
                f"{self._behavior_profile_raw.shape[1]}+3*{self._query_dim}={expected_dim}, "
                f"but profile has dim {raw_profile.shape[1]}. Check online encoders/fusion match the query-aware profile."
            )

        self._calib_Y_tensor = torch.tensor(self._calib_Y, dtype=torch.float32, device=self.device)
        self._calib_cost_norm_tensor = torch.tensor(self._calib_cost_norm, dtype=torch.float32, device=self.device)
        self._behavior_profile_tensor = torch.tensor(self._behavior_profile_raw, dtype=torch.float32, device=self.device)

    def _refresh_scaler_tensors(self):
        self._profile_scaler_mean_tensor = torch.tensor(self.profile_scaler.mean_, dtype=torch.float32, device=self.device)
        self._profile_scaler_scale_tensor = torch.tensor(self.profile_scaler.scale_, dtype=torch.float32, device=self.device)

    def _query_profile_from_features_torch(self, query_features, Y=None, cost_norm=None):
        query_norm = F.normalize(query_features, dim=-1)
        Y = self._calib_Y_tensor if Y is None else Y
        cost_norm = self._calib_cost_norm_tensor if cost_norm is None else cost_norm

        correct_weights = Y.T
        wrong_weights = (1.0 - Y).T
        value_weights = (Y * (1.0 - cost_norm)).T

        def weighted_mean(weights):
            denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
            return weights @ query_norm / denom

        correct_means = weighted_mean(correct_weights)
        wrong_means = weighted_mean(wrong_weights)
        value_means = weighted_mean(value_weights)
        return torch.cat([correct_means, wrong_means, value_means], dim=1)

    def _scaled_profile_from_query_features(self, query_features, Y=None, cost_norm=None):
        query_profile = self._query_profile_from_features_torch(query_features, Y=Y, cost_norm=cost_norm)
        raw_profile = torch.cat([self._behavior_profile_tensor, query_profile], dim=1)
        return (raw_profile - self._profile_scaler_mean_tensor) / self._profile_scaler_scale_tensor

    def _encode_meta_features(self, meta, batch_size: Optional[int] = None, requires_grad: bool = False):
        targets = np.zeros((len(meta), len(self.model_names)), dtype=np.float32)
        old_batch_size = self.batch_size
        if batch_size is not None:
            self.batch_size = batch_size
        loader, _ = self._make_loader(meta, targets, shuffle=False)
        features = []
        context = torch.enable_grad() if requires_grad else torch.no_grad()
        with context:
            for batch in loader:
                features.append(self._encode_query_batch(batch))
        self.batch_size = old_batch_size
        return torch.cat(features, dim=0)

    def _refresh_query_aware_profile(self):
        if self.query_profile_update == "static":
            return torch.tensor(self.model_profile, dtype=torch.float32, device=self.device)
        text_was_training = self.text_model.training
        vision_was_training = self.vision_model.training
        self.text_model.eval()
        self.vision_model.eval()
        refresh_bs = self.profile_refresh_batch_size or self.eval_batch_size or self.batch_size
        query_features = self._encode_meta_features(self._calib_meta, batch_size=refresh_bs, requires_grad=False)
        scaled_profile = self._scaled_profile_from_query_features(query_features)
        self.model_profile = scaled_profile.detach().cpu().numpy().astype(np.float32)
        if text_was_training:
            self.text_model.train()
        if vision_was_training:
            self.vision_model.train()
        return scaled_profile.detach()

    def _differentiable_profile_tensor(self):
        refresh_bs = self.profile_refresh_batch_size or self.batch_size
        calib_meta = self._calib_meta
        Y = self._calib_Y_tensor
        cost_norm = self._calib_cost_norm_tensor
        if self.differentiable_calib_size is not None and self.differentiable_calib_size > 0:
            calib_size = min(int(self.differentiable_calib_size), len(self._calib_meta))
            if calib_size < len(self._calib_meta):
                indices = torch.randperm(len(self._calib_meta), device=self.device)[:calib_size]
                calib_meta = self._calib_meta.iloc[indices.detach().cpu().numpy()].copy().reset_index(drop=True)
                Y = self._calib_Y_tensor.index_select(0, indices)
                cost_norm = self._calib_cost_norm_tensor.index_select(0, indices)
        query_features = self._encode_meta_features(calib_meta, batch_size=refresh_bs, requires_grad=True)
        return self._scaled_profile_from_query_features(query_features, Y=Y, cost_norm=cost_norm)

    def _init_models(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.text_encoder_name)
        if self.text_model is None:
            self.text_model = TextFeatureEncoder(self.text_encoder_name).to(self.device)
        if self.image_processor is None:
            if "clip" in self.vision_encoder_name.lower():
                self.image_processor = CLIPProcessor.from_pretrained(self.vision_encoder_name)
            else:
                self.image_processor = AutoProcessor.from_pretrained(self.vision_encoder_name)
        if self.vision_model is None:
            self.vision_model = VisionFeatureEncoder(self.vision_encoder_name).to(self.device)
        self._wrap_distributed_if_needed()

    def _text_dim(self):
        return int(self._unwrap_model(self.text_model).output_dim)

    def _vision_dim(self):
        return int(self._unwrap_model(self.vision_model).output_dim)

    def _query_dim_from_models(self):
        td = self._text_dim()
        vd = self._vision_dim()
        if self.fusion_method in {"concat", "normalize_concat"}:
            return td + vd
        if self.fusion_method in {"concat_interaction", "normalize_concat_interaction"}:
            return td + vd + 2 * min(td, vd)
        if self.fusion_method == "only_text":
            return td
        if self.fusion_method == "only_image":
            return vd
        if td != vd:
            raise ValueError(f"{self.fusion_method} requires equal text/image dims, got {td} and {vd}")
        return td

    def _encode_text(self, input_ids, attention_mask):
        model = self._unwrap_model(self.text_model) if self._force_unwrapped_forward else self.text_model
        return model(input_ids=input_ids, attention_mask=attention_mask)

    def _encode_vision(self, pixel_values):
        model = self._unwrap_model(self.vision_model) if self._force_unwrapped_forward else self.vision_model
        return model(pixel_values=pixel_values)

    def _fuse_torch(self, text_features, vision_features):
        method = self.fusion_method
        if method == "only_text":
            return text_features
        if method == "only_image":
            return vision_features
        if method == "concat":
            return torch.cat([text_features, vision_features], dim=-1)
        if method == "normalize_concat":
            return torch.cat([F.normalize(text_features, dim=-1), F.normalize(vision_features, dim=-1)], dim=-1)
        if method == "concat_interaction":
            min_dim = min(text_features.shape[-1], vision_features.shape[-1])
            text_shared = text_features[..., :min_dim]
            vision_shared = vision_features[..., :min_dim]
            return torch.cat(
                [
                    text_features,
                    vision_features,
                    text_shared * vision_shared,
                    torch.abs(text_shared - vision_shared),
                ],
                dim=-1,
            )
        if method == "normalize_concat_interaction":
            text_norm = F.normalize(text_features, dim=-1)
            vision_norm = F.normalize(vision_features, dim=-1)
            min_dim = min(text_norm.shape[-1], vision_norm.shape[-1])
            text_shared = text_norm[..., :min_dim]
            vision_shared = vision_norm[..., :min_dim]
            return torch.cat(
                [
                    text_norm,
                    vision_norm,
                    text_shared * vision_shared,
                    torch.abs(text_shared - vision_shared),
                ],
                dim=-1,
            )
        if method == "average":
            return 0.5 * (text_features + vision_features)
        if method == "weighted_average":
            return self.text_weight * text_features + (1.0 - self.text_weight) * vision_features
        raise ValueError(f"Unknown fusion_method: {method}")

    def _encode_query_batch(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        text_features = self._encode_text(input_ids, attention_mask)
        vision_features = self._encode_vision(pixel_values)
        return self._fuse_torch(text_features, vision_features)

    def _score_query_features(self, query_features, profile_x):
        query_encoder = self._unwrap_model(self.query_encoder) if self._force_unwrapped_forward else self.query_encoder
        q = F.normalize(query_encoder(query_features), dim=-1)
        profile_encoder = self._unwrap_model(self.profile_encoder) if self._force_unwrapped_forward else self.profile_encoder
        m = F.normalize(profile_encoder(profile_x), dim=-1)
        return (q @ m.T) / max(self.temperature, 1e-6)

    @staticmethod
    def _hard_targets(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
        labels = np.zeros(Y.shape[0], dtype=np.int64)
        for i in range(Y.shape[0]):
            correct = np.where(Y[i] > 0.5)[0]
            labels[i] = correct[np.argmin(C[i, correct])] if len(correct) > 0 else int(np.argmin(C[i]))
        return labels

    def _soft_targets(self, Y: np.ndarray, C: np.ndarray) -> np.ndarray:
        from routers.utils.soft_targets import build_soft_targets

        return build_soft_targets(Y, C, lam=self.train_lambda, scheme="exp", fallback="cheapest", cost_scale=self.cost_scale).astype(np.float32)

    def _relevance_targets(self, Y: np.ndarray, C: np.ndarray) -> np.ndarray:
        from routers.utils.soft_targets import build_relevance_targets

        return build_relevance_targets(Y, C, lam=self.train_lambda, cost_scale=self.cost_scale, fallback="zeros").astype(np.float32)

    def _targets(self, Y: np.ndarray, C: np.ndarray):
        if self.loss_type == "crm":
            if self.crm_target == "y":
                return (Y > 0.5).astype(np.float32)
            if self.crm_target == "relevance":
                return self._relevance_targets(Y, C)
            if self.use_soft_labels:
                return self._soft_targets(Y, C)
            return self._one_hot(self._hard_targets(Y, C), Y.shape[1])
        if self.loss_type == "clip_relevance":
            return self._relevance_targets(Y, C)
        if self.use_soft_labels:
            return self._soft_targets(Y, C)
        return self._hard_targets(Y, C)

    @staticmethod
    def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
        targets = np.zeros((len(labels), num_classes), dtype=np.float32)
        targets[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
        return targets

    @staticmethod
    def _target_distribution(logits, batch_target):
        if batch_target.dtype in (torch.int64, torch.long):
            return F.one_hot(batch_target, num_classes=logits.shape[1]).float()
        return batch_target.float()

    def _clip_loss(self, logits, batch_target):
        if batch_target.dtype in (torch.int64, torch.long):
            target_dist = F.one_hot(batch_target, num_classes=logits.shape[1]).float()
            q2m_loss = F.cross_entropy(logits, batch_target)
        else:
            target_dist = batch_target.float()
            target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1e-12)
            q2m_loss = -(target_dist * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

        model_mass = target_dist.sum(dim=0)
        valid_models = model_mass > 1e-12
        if not torch.any(valid_models):
            return q2m_loss
        m2q_target = target_dist[:, valid_models].T / model_mass[valid_models, None].clamp_min(1e-12)
        m2q_log_probs = F.log_softmax(logits[:, valid_models].T, dim=-1)
        m2q_loss = -(m2q_target * m2q_log_probs).sum(dim=-1).mean()
        return 0.5 * (q2m_loss + m2q_loss)

    def _compute_loss(self, logits, batch_target):
        if self.loss_type == "softmax":
            if batch_target.dtype in (torch.int64, torch.long):
                return F.cross_entropy(logits, batch_target)
            return -(batch_target.float() * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        if self.loss_type in {"clip", "clip_relevance"}:
            return self._clip_loss(logits, batch_target)
        if self.loss_type == "crm":
            return F.binary_cross_entropy_with_logits(logits, self._target_distribution(logits, batch_target))
        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _make_loader(self, meta, targets, shuffle: bool, distributed: bool = False):
        dataset = RawQueryDataset(meta, targets, self.tokenizer, self.image_processor, self.dataset_dir, self.max_length)
        sampler = None
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                seed=self.random_state,
                drop_last=False,
            )
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            generator=generator if shuffle and sampler is None else None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and str(self.device).startswith("cuda"),
            persistent_workers=self.num_workers > 0,
        )
        return loader, sampler

    def _eval_metrics(self, meta, Y: np.ndarray, C: np.ndarray, profile_tensor):
        old_flag = self._force_unwrapped_forward
        self._force_unwrapped_forward = True
        try:
            eval_batch_size = self.eval_batch_size or max(self.batch_size * 4, self.batch_size)
            preds = self.predict(meta=meta, batch_size=eval_batch_size)
        finally:
            self._force_unwrapped_forward = old_flag
        correct = Y[np.arange(len(preds)), preds]
        costs = C[np.arange(len(preds)), preds]
        metrics = {"accuracy": float(correct.mean()), "avg_cost": float(costs.mean())}
        if self._rank_score_cmin is not None and self._rank_score_cmax is not None:
            try:
                from routers.utils.rank_score import rank_score

                metrics["rank_score"] = float(rank_score(metrics["accuracy"], metrics["avg_cost"], self._rank_score_cmin, self._rank_score_cmax, beta=self._rank_score_beta))
            except Exception:
                pass
        return metrics

    def fit(self, Y: np.ndarray, C: np.ndarray, meta, model_mapping: Optional[Dict[int, str]] = None, costs: Optional[np.ndarray] = None, Y_dev=None, C_dev=None, meta_dev=None, **kwargs):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        self._init_models()

        raw_profile, profile_names = self._load_profile()
        if raw_profile.shape[0] != Y.shape[1]:
            raise ValueError(f"Profile has {raw_profile.shape[0]} models, but Y has {Y.shape[1]} columns")
        self.profile_scaler = NumpyStandardScaler()
        self.model_profile = self.profile_scaler.fit_transform(raw_profile)
        self.model_names = profile_names.tolist()
        self.model_mapping = model_mapping or {i: self.model_names[i] for i in range(len(self.model_names))}
        self.reverse_mapping = {v: k for k, v in self.model_mapping.items()}
        self.costs = np.asarray(costs, dtype=float) if costs is not None else C.mean(axis=0)

        self._rank_score_cmin = kwargs.get("cmin", None)
        self._rank_score_cmax = kwargs.get("cmax", None)
        self._rank_score_beta = float(kwargs.get("rank_score_beta", kwargs.get("beta", 0.1)))

        self._query_dim = self._query_dim_from_models()
        self._setup_dynamic_query_profile(raw_profile, kwargs.get("calibration_meta"))
        self._refresh_scaler_tensors()
        self.query_encoder = ProjectionMLP(
            input_dim=self._query_dim,
            hidden_dim=self.query_hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.query_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.profile_encoder = ProjectionMLP(
            input_dim=self.model_profile.shape[1],
            hidden_dim=self.profile_hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.profile_layers,
            dropout=self.dropout,
        ).to(self.device)
        self._wrap_distributed_if_needed()
        self._last_profile_dim = self.model_profile.shape[1]

        y_or_T = self._targets(Y, C)
        train_loader, train_sampler = self._make_loader(meta, y_or_T, shuffle=True, distributed=self.distributed)
        profile_tensor = torch.tensor(self.model_profile, dtype=torch.float32, device=self.device)
        if self.query_profile_update in {"epoch_refresh", "differentiable"}:
            profile_tensor = self._refresh_query_aware_profile()

        optimizer = torch.optim.AdamW(
            [
                {"params": self.text_model.parameters(), "lr": self.learning_rate},
                {"params": self.vision_model.parameters(), "lr": self.learning_rate},
                {"params": self.query_encoder.parameters(), "lr": self.profile_learning_rate},
                {"params": self.profile_encoder.parameters(), "lr": self.profile_learning_rate},
            ],
            weight_decay=self.weight_decay,
        )

        is_main = (not self.distributed) or self.rank == 0
        if self.verbose > 0 and is_main:
            target_desc = "soft" if self.use_soft_labels else "hard"
            print(f"  Online SCOPE-Router training on {self.device}: {len(meta)} samples, {Y.shape[1]} models")
            print(
                f"  text_encoder={self.text_encoder_name}, vision_encoder={self.vision_encoder_name}, "
                f"query_dim={self._query_dim}, profile_dim={self.model_profile.shape[1]}, "
                f"emb_dim={self.embedding_dim}, query_hidden_dim={self.query_hidden_dim}, "
                f"labels={target_desc}, loss={self.loss_type}, crm_target={self.crm_target}, "
                f"query_profile_update={self.query_profile_update}, "
                f"distributed={self.distributed}, world_size={self.world_size}"
            )

        best_value = -float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, self.max_iter + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.text_model.train()
            self.vision_model.train()
            self.query_encoder.train()
            self.profile_encoder.train()
            losses = []
            for batch in train_loader:
                target = batch["target"].to(self.device)
                optimizer.zero_grad()
                query_features = self._encode_query_batch(batch)
                batch_profile_tensor = (
                    self._differentiable_profile_tensor()
                    if self.query_profile_update == "differentiable"
                    else profile_tensor
                )
                logits = self._score_query_features(query_features, batch_profile_tensor)
                loss = self._compute_loss(logits, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.text_model.parameters())
                    + list(self.vision_model.parameters())
                    + list(self.query_encoder.parameters())
                    + list(self.profile_encoder.parameters()),
                    1.0,
                )
                optimizer.step()
                losses.append(float(loss.item()))

            self._distributed_barrier()
            if self.query_profile_update in {"epoch_refresh", "differentiable"}:
                profile_tensor = self._refresh_query_aware_profile()

            stop_now = False
            if is_main:
                avg_loss = float(np.mean(losses)) if losses else float("inf")
                train_metrics = {"loss": avg_loss}
                metric_value = -avg_loss
                dev_metrics = None
                if Y_dev is not None and C_dev is not None and meta_dev is not None:
                    dev_metrics = self._eval_metrics(meta_dev, Y_dev, C_dev, profile_tensor)
                    metric_value = dev_metrics.get(self.monitor_metric, dev_metrics["accuracy"])
                elif not self.distributed:
                    train_metrics = self._eval_metrics(meta, Y, C, profile_tensor)
                    metric_value = train_metrics.get(self.monitor_metric, train_metrics["accuracy"])

                improved = metric_value > best_value + 1e-6
                if improved:
                    best_value = metric_value
                    self.best_epoch = epoch
                    self.best_dev_metrics = dev_metrics or train_metrics
                    best_state = {
                        "text_model": {k: v.detach().cpu().clone() for k, v in self._model_state_dict(self.text_model).items()},
                        "vision_model": {k: v.detach().cpu().clone() for k, v in self._model_state_dict(self.vision_model).items()},
                        "query_encoder": {k: v.detach().cpu().clone() for k, v in self._model_state_dict(self.query_encoder).items()},
                        "profile_encoder": {k: v.detach().cpu().clone() for k, v in self._model_state_dict(self.profile_encoder).items()},
                        "model_profile": self.model_profile.copy(),
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if self.verbose > 0:
                    msg = f"  Epoch {epoch:3d}: loss={avg_loss:.4f}"
                    if "accuracy" in train_metrics:
                        msg += f" train_acc={train_metrics['accuracy']:.4f}"
                    if dev_metrics is not None:
                        msg += f" dev_acc={dev_metrics['accuracy']:.4f} dev_cost=${dev_metrics['avg_cost']:.6f}"
                        if "rank_score" in dev_metrics:
                            msg += f" rank_score={dev_metrics['rank_score']:.4f}"
                    if improved:
                        msg += " *"
                    print(msg)

                if self.patience > 0 and bad_epochs >= self.patience:
                    if self.verbose > 0:
                        print(f"  Early stopping at epoch {epoch}; best epoch={self.best_epoch}")
                    stop_now = True

            if self.distributed:
                stop_tensor = torch.tensor([1 if stop_now else 0], dtype=torch.int32, device=self.device)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(stop_tensor.item())
            if stop_now:
                break

        if best_state is not None and is_main:
            self._unwrap_model(self.text_model).load_state_dict(best_state["text_model"])
            self._unwrap_model(self.vision_model).load_state_dict(best_state["vision_model"])
            self._unwrap_model(self.query_encoder).load_state_dict(best_state["query_encoder"])
            self._unwrap_model(self.profile_encoder).load_state_dict(best_state["profile_encoder"])
            self.model_profile = best_state["model_profile"]
        return self

    def predict(self, meta, batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        if self.text_model is None or self.vision_model is None or self.query_encoder is None or self.profile_encoder is None:
            raise ValueError("Router is not fitted")
        old_batch_size = self.batch_size
        if batch_size is not None:
            self.batch_size = batch_size
        targets = np.zeros((len(meta), len(self.model_names)), dtype=np.float32)
        loader, _ = self._make_loader(meta, targets, shuffle=False)
        profile_tensor = torch.tensor(self.model_profile, dtype=torch.float32, device=self.device)
        self.text_model.eval()
        self.vision_model.eval()
        self.query_encoder.eval()
        self.profile_encoder.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                query_features = self._encode_query_batch(batch)
                logits = self._score_query_features(query_features, profile_tensor)
                preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        self.batch_size = old_batch_size
        return np.asarray(preds, dtype=np.int64)

    def predict_proba(self, meta, batch_size: Optional[int] = None, **kwargs) -> np.ndarray:
        old_batch_size = self.batch_size
        if batch_size is not None:
            self.batch_size = batch_size
        targets = np.zeros((len(meta), len(self.model_names)), dtype=np.float32)
        loader, _ = self._make_loader(meta, targets, shuffle=False)
        profile_tensor = torch.tensor(self.model_profile, dtype=torch.float32, device=self.device)
        self.text_model.eval()
        self.vision_model.eval()
        self.query_encoder.eval()
        self.profile_encoder.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                query_features = self._encode_query_batch(batch)
                logits = self._score_query_features(query_features, profile_tensor)
                batch_probs = torch.sigmoid(logits) if self.loss_type == "crm" else F.softmax(logits, dim=-1)
                probs.append(batch_probs.cpu().numpy())
        self.batch_size = old_batch_size
        return np.concatenate(probs, axis=0)

    def save(self, path: str):
        data = {
            "config": {
                "profile_path": self.profile_path,
                "dataset_dir": self.dataset_dir,
                "text_encoder": self.text_encoder_name,
                "vision_encoder": self.vision_encoder_name,
                "fusion_method": self.fusion_method,
                "text_weight": self.text_weight,
                "embedding_dim": self.embedding_dim,
                "query_hidden_dim": self.query_hidden_dim,
                "query_layers": self.query_layers,
                "profile_hidden_dim": self.profile_hidden_dim,
                "profile_layers": self.profile_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "profile_learning_rate": self.profile_learning_rate,
                "weight_decay": self.weight_decay,
                "batch_size": self.batch_size,
                "eval_batch_size": self.eval_batch_size,
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "max_iter": self.max_iter,
                "max_length": self.max_length,
                "temperature": self.temperature,
                "loss_type": self.loss_type,
                "crm_target": self.crm_target,
                "train_lambda": self.train_lambda,
                "use_soft_labels": self.use_soft_labels,
                "cost_scale": self.cost_scale,
                "patience": self.patience,
                "monitor_metric": self.monitor_metric,
                "query_profile_update": self.query_profile_update,
                "profile_refresh_batch_size": self.profile_refresh_batch_size,
                "differentiable_calib_size": self.differentiable_calib_size,
                "device": self.device,
                "multi_gpu": False,
                "distributed": False,
                "local_rank": 0,
                "rank": 0,
                "world_size": 1,
                "random_state": self.random_state,
                "verbose": self.verbose,
            },
            "profile_scaler": self.profile_scaler,
            "model_profile": self.model_profile,
            "model_names": self.model_names,
            "model_mapping": self.model_mapping,
            "reverse_mapping": self.reverse_mapping,
            "costs": self.costs,
            "best_epoch": self.best_epoch,
            "best_dev_metrics": self.best_dev_metrics,
            "rank_score": {"cmin": self._rank_score_cmin, "cmax": self._rank_score_cmax, "beta": self._rank_score_beta},
            "last_profile_dim": self._last_profile_dim,
            "query_dim": self._query_dim,
            "text_model_state": self._model_state_dict(self.text_model) if self.text_model is not None else None,
            "vision_model_state": self._model_state_dict(self.vision_model) if self.vision_model is not None else None,
            "query_encoder_state": self._model_state_dict(self.query_encoder) if self.query_encoder is not None else None,
            "profile_encoder_state": self._model_state_dict(self.profile_encoder) if self.profile_encoder is not None else None,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        config = data["config"]
        config.setdefault("multi_gpu", False)
        config.setdefault("distributed", False)
        config.setdefault("local_rank", 0)
        config.setdefault("rank", 0)
        config.setdefault("world_size", 1)
        config.setdefault("eval_batch_size", None)
        config.setdefault("num_workers", 0)
        config.setdefault("pin_memory", False)
        config.setdefault("embedding_dim", data.get("query_dim"))
        config.setdefault("query_hidden_dim", 512)
        config.setdefault("query_layers", 2)
        config.setdefault("query_profile_update", "static")
        config.setdefault("profile_refresh_batch_size", None)
        config.setdefault("differentiable_calib_size", None)
        router = cls(**config)
        router.profile_scaler = data["profile_scaler"]
        router.model_profile = data["model_profile"]
        router.model_names = data["model_names"]
        router.model_mapping = data["model_mapping"]
        router.reverse_mapping = data["reverse_mapping"]
        router.costs = data["costs"]
        router.best_epoch = data.get("best_epoch")
        router.best_dev_metrics = data.get("best_dev_metrics")
        rank_data = data.get("rank_score", {})
        router._rank_score_cmin = rank_data.get("cmin")
        router._rank_score_cmax = rank_data.get("cmax")
        router._rank_score_beta = rank_data.get("beta", 0.1)
        router._last_profile_dim = data["last_profile_dim"]
        router._query_dim = data["query_dim"]
        router._init_models()
        router.query_encoder = ProjectionMLP(
            input_dim=router._query_dim,
            hidden_dim=router.query_hidden_dim,
            output_dim=router.embedding_dim,
            num_layers=router.query_layers,
            dropout=router.dropout,
        ).to(router.device)
        router.profile_encoder = ProjectionMLP(
            input_dim=router._last_profile_dim,
            hidden_dim=router.profile_hidden_dim,
            output_dim=router.embedding_dim,
            num_layers=router.profile_layers,
            dropout=router.dropout,
        ).to(router.device)
        router._unwrap_model(router.text_model).load_state_dict(data["text_model_state"])
        router._unwrap_model(router.vision_model).load_state_dict(data["vision_model_state"])
        router._unwrap_model(router.query_encoder).load_state_dict(data["query_encoder_state"])
        router._unwrap_model(router.profile_encoder).load_state_dict(data["profile_encoder_state"])
        return router

    def __repr__(self):
        return f"OnlineScopeRouter(profile={Path(self.profile_path).name}, loss={self.loss_type})"
