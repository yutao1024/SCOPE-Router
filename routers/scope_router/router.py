#!/usr/bin/env python3
"""
SCOPE-Router.

This router keeps query embeddings frozen and trains lightweight projections:

    query_embedding -> QueryMLP   -> routing space
    model_profile   -> profile projection -> routing space

Scores are cosine/dot-product similarities between query and model embeddings.
The paper configuration trains with Cost-aware Relevance Matching (CRM) and
Routing-Consistency Contrastive Regularization (RCCR).
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from routers.common import RouterBase

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None
    DataLoader = None
    TensorDataset = None


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


if HAS_TORCH:
    LEARNABLE_FUSION_METHODS = {
        "learnable_concat",
        "learnable_gated_sum",
    }


    class ProjectionMLP(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
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


    class LearnableFusionQueryEncoder(nn.Module):
        def __init__(
            self,
            text_dim: int,
            vision_dim: int,
            hidden_dim: int,
            output_dim: int,
            method: str,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            if method not in LEARNABLE_FUSION_METHODS:
                raise ValueError(f"Unknown learnable fusion method: {method}")
            self.text_dim = int(text_dim)
            self.vision_dim = int(vision_dim)
            self.method = method
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.vision_proj = nn.Sequential(
                nn.Linear(self.vision_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            if method == "learnable_gated_sum":
                self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
                fusion_dim = hidden_dim
            else:
                self.gate = None
                fusion_dim = 2 * hidden_dim
            self.out = ProjectionMLP(
                input_dim=fusion_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=max(1, num_layers),
                dropout=dropout,
            )

        def forward(self, x):
            text_x = x[..., :self.text_dim]
            vision_x = x[..., self.text_dim:self.text_dim + self.vision_dim]
            text_h = self.text_proj(text_x)
            vision_h = self.vision_proj(vision_x)
            if self.method == "learnable_gated_sum":
                gate = torch.sigmoid(self.gate(torch.cat([text_h, vision_h], dim=-1)))
                fused = gate * text_h + (1.0 - gate) * vision_h
            else:
                fused = torch.cat([text_h, vision_h], dim=-1)
            return self.out(fused)
else:
    LEARNABLE_FUSION_METHODS = set()
    ProjectionMLP = None


class ScopeRouter(RouterBase):
    def __init__(
        self,
        profile_path: str,
        embedding_dim: int = 256,
        query_hidden_dim: int = 512,
        profile_hidden_dim: int = 512,
        query_layers: int = 2,
        profile_layers: int = 2,
        dropout: float = 0.1,
        fusion_method: str = "normalize_concat",
        text_weight: float = 0.5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_type: str = "adamw",
        lr_scheduler: str = "none",
        min_lr_factor: float = 0.05,
        lr_step_size: int = 20,
        lr_gamma: float = 0.5,
        lr_plateau_patience: int = 5,
        batch_size: int = 512,
        max_iter: int = 100,
        temperature: float = 0.07,
        score_type: str = "dot",
        bilinear_rank: int = 16,
        bilinear_residual_weight: float = 1.0,
        set_encoder_layers: int = 1,
        set_encoder_heads: int = 4,
        set_encoder_ff_dim: int = 128,
        set_encoder_residual_weight: float = 0.25,
        set_encoder_normalize_input: bool = True,
        set_encoder_context_only: bool = False,
        cross_attention_heads: int = 4,
        cross_attention_residual_weight: float = 0.25,
        loss_type: str = "softmax",
        crm_target: str = "soft",
        crm_bias: str = "none",
        blip_match_weight: float = 1.0,
        blip_alpha: float = 0.4,
        learn_blip_alpha: bool = False,
        blip_momentum: float = 0.995,
        rccr_weight: float = 0.0,
        rccr_temperature: float = 0.1,
        learn_rccr_temperature: bool = False,
        train_lambda: float = 10.0,
        use_soft_labels: bool = True,
        cost_scale: float = 100.0,
        patience: int = 20,
        monitor_metric: str = "rank_score",
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: int = 1,
    ):
        if not HAS_TORCH:
            raise ImportError("ScopeRouter requires PyTorch")
        if loss_type == "clip-relevance":
            loss_type = "clip_relevance"

        self.profile_path = str(profile_path)
        self.embedding_dim = embedding_dim
        self.query_hidden_dim = query_hidden_dim
        self.profile_hidden_dim = profile_hidden_dim
        self.query_layers = query_layers
        self.profile_layers = profile_layers
        self.dropout = dropout
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if optimizer_type not in {"adamw", "adam", "sgd"}:
            raise ValueError(f"optimizer_type must be one of adamw, adam, sgd; got {optimizer_type}")
        self.optimizer_type = optimizer_type
        if lr_scheduler not in {"none", "cosine", "plateau", "step"}:
            raise ValueError(f"lr_scheduler must be one of none, cosine, plateau, step; got {lr_scheduler}")
        self.lr_scheduler = lr_scheduler
        self.min_lr_factor = float(min_lr_factor)
        if not 0.0 <= self.min_lr_factor <= 1.0:
            raise ValueError(f"min_lr_factor must be in [0, 1]; got {min_lr_factor}")
        self.lr_step_size = int(lr_step_size)
        if self.lr_step_size <= 0:
            raise ValueError(f"lr_step_size must be positive; got {lr_step_size}")
        self.lr_gamma = float(lr_gamma)
        if self.lr_gamma <= 0:
            raise ValueError(f"lr_gamma must be positive; got {lr_gamma}")
        self.lr_plateau_patience = int(lr_plateau_patience)
        if self.lr_plateau_patience < 0:
            raise ValueError(f"lr_plateau_patience must be non-negative; got {lr_plateau_patience}")
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.temperature = temperature
        if score_type not in {"dot", "lowrank_bilinear", "set_aware", "cross_attention"}:
            raise ValueError(
                "score_type must be one of dot, lowrank_bilinear, set_aware, cross_attention; "
                f"got {score_type}"
            )
        self.score_type = score_type
        self.bilinear_rank = int(bilinear_rank)
        if self.bilinear_rank <= 0:
            raise ValueError(f"bilinear_rank must be positive; got {bilinear_rank}")
        self.bilinear_residual_weight = float(bilinear_residual_weight)
        self.set_encoder_layers = int(set_encoder_layers)
        self.set_encoder_heads = int(set_encoder_heads)
        self.set_encoder_ff_dim = int(set_encoder_ff_dim)
        if self.set_encoder_layers <= 0:
            raise ValueError(f"set_encoder_layers must be positive; got {set_encoder_layers}")
        if self.set_encoder_heads <= 0:
            raise ValueError(f"set_encoder_heads must be positive; got {set_encoder_heads}")
        if self.set_encoder_ff_dim <= 0:
            raise ValueError(f"set_encoder_ff_dim must be positive; got {set_encoder_ff_dim}")
        self.set_encoder_residual_weight = float(set_encoder_residual_weight)
        self.set_encoder_normalize_input = bool(set_encoder_normalize_input)
        self.set_encoder_context_only = bool(set_encoder_context_only)
        self.cross_attention_heads = int(cross_attention_heads)
        if self.cross_attention_heads <= 0:
            raise ValueError(f"cross_attention_heads must be positive; got {cross_attention_heads}")
        self.cross_attention_residual_weight = float(cross_attention_residual_weight)
        if loss_type not in {"softmax", "clip", "clip_relevance", "crm", "blip"}:
            raise ValueError(
                "loss_type must be one of softmax, clip, clip_relevance, crm, blip; "
                f"got {loss_type}"
            )
        self.loss_type = loss_type
        if crm_target not in {"soft", "y", "relevance"}:
            raise ValueError(f"crm_target must be one of soft, y, relevance; got {crm_target}")
        self.crm_target = crm_target
        if crm_bias not in {"none", "global", "profile"}:
            raise ValueError(f"crm_bias must be one of none, global, profile; got {crm_bias}")
        self.crm_bias = crm_bias
        # Kept only for loading older checkpoints/configs; BLIP now uses momentum ITC only.
        self.blip_match_weight = float(blip_match_weight)
        self.blip_alpha = float(blip_alpha)
        self.learn_blip_alpha = bool(learn_blip_alpha)
        self.blip_momentum = float(blip_momentum)
        self.rccr_weight = float(rccr_weight)
        self.rccr_temperature = float(rccr_temperature)
        self.learn_rccr_temperature = bool(learn_rccr_temperature)
        self.train_lambda = train_lambda
        self.use_soft_labels = use_soft_labels
        self.cost_scale = cost_scale
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.random_state = random_state
        self.verbose = verbose

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"

        self.query_scaler = None
        self.profile_scaler = None
        self.query_encoder = None
        self.profile_encoder = None
        self.query_encoder_m = None
        self.profile_encoder_m = None
        self.blip_alpha_logit = None
        self.profile_bias_encoder = None
        self.global_crm_bias = None
        self.rccr_log_temperature = None
        self.query_bilinear = None
        self.profile_bilinear = None
        self.profile_set_encoder = None
        self.query_cross_attention = None
        self.model_profile = None
        self.model_names = None
        self.model_mapping = None
        self.reverse_mapping = None
        self.costs = None
        self.best_epoch = None
        self.best_dev_metrics = None
        self._last_query_dim = None
        self._last_profile_dim = None
        self._last_text_dim = None
        self._last_vision_dim = None
        self._rank_score_cmin = None
        self._rank_score_cmax = None
        self._rank_score_beta = 0.1
        self._cached_profile_tensor = None
        self._cached_profile_features = None
        self._cached_profile_lowrank_features = None
        self._cached_profile_set_features = None
        self._cached_profile_bias = None

    def _is_learnable_fusion(self) -> bool:
        return self.fusion_method in LEARNABLE_FUSION_METHODS

    def _fuse(
        self,
        meta=None,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if X is not None:
            return X.astype(np.float32)

        if X_text is not None and X_vision is not None:
            if self._is_learnable_fusion():
                self._last_text_dim = int(X_text.shape[1])
                self._last_vision_dim = int(X_vision.shape[1])
                return np.hstack([X_text, X_vision]).astype(np.float32)

            from routers.utils.fusion import fuse_embeddings

            return fuse_embeddings(
                X_text,
                X_vision,
                method=self.fusion_method,
                text_weight=self.text_weight,
            ).astype(np.float32)

        if meta is not None:
            if "embedding" in meta.columns:
                return np.vstack(meta["embedding"].values).astype(np.float32)
            if "text_embedding" in meta.columns and "vision_embedding" in meta.columns:
                from routers.utils.fusion import fuse_embeddings

                Xt = np.vstack(meta["text_embedding"].values)
                Xv = np.vstack(meta["vision_embedding"].values)
                if self._is_learnable_fusion():
                    self._last_text_dim = int(Xt.shape[1])
                    self._last_vision_dim = int(Xv.shape[1])
                    return np.hstack([Xt, Xv]).astype(np.float32)
                return fuse_embeddings(Xt, Xv, self.fusion_method, self.text_weight).astype(np.float32)

        raise ValueError("Features required: X or (X_text, X_vision) or meta embeddings")

    def _load_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.load(self.profile_path, allow_pickle=False)
        if "model_profile" not in data:
            raise ValueError(f"{self.profile_path} must contain model_profile")
        model_profile = data["model_profile"].astype(np.float32)
        if "model_names" in data:
            model_names = data["model_names"].astype(str)
        else:
            model_names = np.array([f"model_{i}" for i in range(model_profile.shape[0])], dtype=str)
        return model_profile, model_names

    @staticmethod
    def _hard_targets(Y: np.ndarray, C: np.ndarray) -> np.ndarray:
        labels = np.zeros(Y.shape[0], dtype=np.int64)
        for i in range(Y.shape[0]):
            correct = np.where(Y[i] > 0.5)[0]
            if len(correct) > 0:
                labels[i] = correct[np.argmin(C[i, correct])]
            else:
                labels[i] = int(np.argmin(C[i]))
        return labels

    def _targets(self, Y: np.ndarray, C: np.ndarray) -> np.ndarray:
        if self.use_soft_labels:
            from routers.utils.soft_targets import build_soft_targets

            return build_soft_targets(
                Y,
                C,
                lam=self.train_lambda,
                scheme="exp",
                fallback="cheapest",
                cost_scale=self.cost_scale,
            ).astype(np.float32)
        return self._hard_targets(Y, C)

    def _crm_targets(self, Y: np.ndarray, C: np.ndarray, y_or_T: np.ndarray) -> np.ndarray:
        if self.crm_target == "y":
            return (Y > 0.5).astype(np.float32)
        if self.crm_target == "relevance":
            from routers.utils.soft_targets import build_relevance_targets

            return build_relevance_targets(
                Y,
                C,
                lam=self.train_lambda,
                cost_scale=self.cost_scale,
                fallback="zeros",
            ).astype(np.float32)
        if self.use_soft_labels:
            return y_or_T.astype(np.float32)
        return self._one_hot(y_or_T, Y.shape[1])

    def _relevance_targets(self, Y: np.ndarray, C: np.ndarray) -> np.ndarray:
        from routers.utils.soft_targets import build_relevance_targets

        return build_relevance_targets(
            Y,
            C,
            lam=self.train_lambda,
            cost_scale=self.cost_scale,
            fallback="zeros",
        ).astype(np.float32)

    @staticmethod
    def _one_hot(labels: np.ndarray, num_classes: int) -> np.ndarray:
        targets = np.zeros((len(labels), num_classes), dtype=np.float32)
        targets[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
        return targets

    def _build_modules(self, query_dim: int, profile_dim: int):
        if self._is_learnable_fusion():
            if self._last_text_dim is None or self._last_vision_dim is None:
                raise ValueError(
                    f"{self.fusion_method} requires separate text/image dimensions. "
                    "Fit or predict with X_text and X_vision, not only a pre-fused X array."
                )
            if query_dim != self._last_text_dim + self._last_vision_dim:
                raise ValueError(
                    f"Learnable fusion query_dim mismatch: query={query_dim}, "
                    f"text+vision={self._last_text_dim + self._last_vision_dim}"
                )
            self.query_encoder = LearnableFusionQueryEncoder(
                text_dim=self._last_text_dim,
                vision_dim=self._last_vision_dim,
                hidden_dim=self.query_hidden_dim,
                output_dim=self.embedding_dim,
                method=self.fusion_method,
                num_layers=self.query_layers,
                dropout=self.dropout,
            ).to(self.device)
        else:
            self.query_encoder = ProjectionMLP(
                input_dim=query_dim,
                hidden_dim=self.query_hidden_dim,
                output_dim=self.embedding_dim,
                num_layers=self.query_layers,
                dropout=self.dropout,
            ).to(self.device)
        self.profile_encoder = ProjectionMLP(
            input_dim=profile_dim,
            hidden_dim=self.profile_hidden_dim,
            output_dim=self.embedding_dim,
            num_layers=self.profile_layers,
            dropout=self.dropout,
        ).to(self.device)
        self.query_encoder_m = None
        self.profile_encoder_m = None
        self.blip_alpha_logit = None
        self.rccr_log_temperature = None
        if self.loss_type == "blip":
            if self._is_learnable_fusion():
                self.query_encoder_m = LearnableFusionQueryEncoder(
                    text_dim=self._last_text_dim,
                    vision_dim=self._last_vision_dim,
                    hidden_dim=self.query_hidden_dim,
                    output_dim=self.embedding_dim,
                    method=self.fusion_method,
                    num_layers=self.query_layers,
                    dropout=self.dropout,
                ).to(self.device)
            else:
                self.query_encoder_m = ProjectionMLP(
                    input_dim=query_dim,
                    hidden_dim=self.query_hidden_dim,
                    output_dim=self.embedding_dim,
                    num_layers=self.query_layers,
                    dropout=self.dropout,
                ).to(self.device)
            self.profile_encoder_m = ProjectionMLP(
                input_dim=profile_dim,
                hidden_dim=self.profile_hidden_dim,
                output_dim=self.embedding_dim,
                num_layers=self.profile_layers,
                dropout=self.dropout,
            ).to(self.device)
            self._copy_to_momentum()
            self.query_encoder_m.eval()
            self.profile_encoder_m.eval()
            if self.learn_blip_alpha:
                alpha = float(np.clip(self.blip_alpha, 1e-4, 1.0 - 1e-4))
                raw_alpha = np.log(alpha / (1.0 - alpha))
                self.blip_alpha_logit = nn.Parameter(
                    torch.tensor(raw_alpha, dtype=torch.float32, device=self.device)
                )
        if self.rccr_weight > 0 and self.learn_rccr_temperature:
            init_temp = max(float(self.rccr_temperature), 1e-6)
            self.rccr_log_temperature = nn.Parameter(
                torch.tensor(np.log(init_temp), dtype=torch.float32, device=self.device)
            )
        self.profile_bias_encoder = None
        self.global_crm_bias = None
        self.query_bilinear = None
        self.profile_bilinear = None
        self.profile_set_encoder = None
        self.query_cross_attention = None
        if self.score_type == "lowrank_bilinear":
            self.query_bilinear = nn.Linear(
                self.embedding_dim,
                self.bilinear_rank,
                bias=False,
            ).to(self.device)
            self.profile_bilinear = nn.Linear(
                self.embedding_dim,
                self.bilinear_rank,
                bias=False,
            ).to(self.device)
        elif self.score_type == "set_aware":
            if self.embedding_dim % self.set_encoder_heads != 0:
                raise ValueError(
                    "embedding_dim must be divisible by set_encoder_heads for set_aware scorer; "
                    f"got embedding_dim={self.embedding_dim}, heads={self.set_encoder_heads}"
                )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.embedding_dim,
                nhead=self.set_encoder_heads,
                dim_feedforward=self.set_encoder_ff_dim,
                dropout=self.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.profile_set_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=self.set_encoder_layers,
            ).to(self.device)
        elif self.score_type == "cross_attention":
            if self.embedding_dim % self.cross_attention_heads != 0:
                raise ValueError(
                    "embedding_dim must be divisible by cross_attention_heads for cross_attention scorer; "
                    f"got embedding_dim={self.embedding_dim}, heads={self.cross_attention_heads}"
                )
            self.query_cross_attention = nn.MultiheadAttention(
                embed_dim=self.embedding_dim,
                num_heads=self.cross_attention_heads,
                dropout=self.dropout,
                batch_first=True,
            ).to(self.device)
        if self.loss_type == "crm" and self.crm_bias == "global":
            self.global_crm_bias = nn.Parameter(torch.zeros((), dtype=torch.float32, device=self.device))
        elif self.loss_type == "crm" and self.crm_bias == "profile":
            self.profile_bias_encoder = ProjectionMLP(
                input_dim=profile_dim,
                hidden_dim=self.profile_hidden_dim,
                output_dim=1,
                num_layers=self.profile_layers,
                dropout=self.dropout,
            ).to(self.device)
        self._last_query_dim = query_dim
        self._last_profile_dim = profile_dim
        self._invalidate_profile_cache()

    def _momentum_pairs(self):
        return (
            (self.query_encoder, self.query_encoder_m),
            (self.profile_encoder, self.profile_encoder_m),
        )

    def _copy_to_momentum(self):
        with torch.no_grad():
            for online, momentum in self._momentum_pairs():
                if online is None or momentum is None:
                    continue
                for param, param_m in zip(online.parameters(), momentum.parameters()):
                    param_m.data.copy_(param.data)
                    param_m.requires_grad = False

    def _update_momentum(self):
        if self.loss_type != "blip":
            return
        momentum = self.blip_momentum
        with torch.no_grad():
            for online, momentum_model in self._momentum_pairs():
                if online is None or momentum_model is None:
                    continue
                for param, param_m in zip(online.parameters(), momentum_model.parameters()):
                    param_m.data.mul_(momentum).add_(param.data, alpha=1.0 - momentum)
                momentum_model.eval()

    def _score_with_encoders(self, query_x, profile_x, query_encoder, profile_encoder):
        q = F.normalize(query_encoder(query_x), dim=-1)
        m = self._profile_features_for_scoring(profile_encoder(profile_x))
        return self._pair_logits(q, m)

    def _lowrank_features(self, features, projection):
        return F.normalize(projection(features), dim=-1)

    def _profile_features_for_scoring(self, profile_features):
        if self.score_type == "set_aware" and not self.set_encoder_normalize_input:
            return profile_features
        return F.normalize(profile_features, dim=-1)

    def _set_aware_profile_features(self, m):
        if self.score_type != "set_aware":
            return m
        contextual = self.profile_set_encoder(m.unsqueeze(0)).squeeze(0)
        if self.set_encoder_context_only:
            return F.normalize(contextual, dim=-1)
        return F.normalize(m + self.set_encoder_residual_weight * contextual, dim=-1)

    def _cross_attention_query_features(self, q, m):
        if self.score_type != "cross_attention":
            return q
        batch_size = q.shape[0]
        profile_tokens = m.unsqueeze(0).expand(batch_size, -1, -1)
        contextual, _ = self.query_cross_attention(
            q.unsqueeze(1),
            profile_tokens,
            profile_tokens,
            need_weights=False,
        )
        return F.normalize(q + self.cross_attention_residual_weight * contextual.squeeze(1), dim=-1)

    def _pair_logits(self, q, m, m_lowrank=None, set_encoded: bool = False):
        if self.score_type == "set_aware" and not set_encoded:
            m = self._set_aware_profile_features(m)
        if self.score_type == "cross_attention":
            q = self._cross_attention_query_features(q, m)
        logits = q @ m.T
        if self.score_type == "lowrank_bilinear":
            q_lowrank = self._lowrank_features(q, self.query_bilinear)
            if m_lowrank is None:
                m_lowrank = self._lowrank_features(m, self.profile_bilinear)
            logits = logits + self.bilinear_residual_weight * (q_lowrank @ m_lowrank.T)
        return logits / max(self.temperature, 1e-6)

    def _score_query_features(self, query_features, profile_x):
        q = F.normalize(query_features, dim=-1)
        m = self._profile_features_for_scoring(self.profile_encoder(profile_x))
        logits = self._pair_logits(q, m)
        if self.loss_type == "crm":
            if self.crm_bias == "global" and self.global_crm_bias is not None:
                logits = logits + self.global_crm_bias
            elif self.crm_bias == "profile" and self.profile_bias_encoder is not None:
                logits = logits + self.profile_bias_encoder(profile_x).squeeze(-1)[None, :]
        return logits

    def _invalidate_profile_cache(self):
        self._cached_profile_tensor = None
        self._cached_profile_features = None
        self._cached_profile_lowrank_features = None
        self._cached_profile_set_features = None
        self._cached_profile_bias = None

    def _profile_tensor_for_inference(self):
        if self._cached_profile_tensor is None:
            self._cached_profile_tensor = torch.as_tensor(
                self.model_profile,
                dtype=torch.float32,
                device=self.device,
            )
        return self._cached_profile_tensor

    def _profile_features_for_inference(self):
        if self._cached_profile_features is None:
            self.profile_encoder.eval()
            with torch.inference_mode():
                self._cached_profile_features = self._profile_features_for_scoring(
                    self.profile_encoder(self._profile_tensor_for_inference())
                )
        return self._cached_profile_features

    def _profile_lowrank_features_for_inference(self):
        if self.score_type != "lowrank_bilinear":
            return None
        if self._cached_profile_lowrank_features is None:
            self.profile_bilinear.eval()
            with torch.inference_mode():
                self._cached_profile_lowrank_features = self._lowrank_features(
                    self._profile_features_for_inference(),
                    self.profile_bilinear,
                )
        return self._cached_profile_lowrank_features

    def _profile_set_features_for_inference(self):
        if self.score_type != "set_aware":
            return self._profile_features_for_inference()
        if self._cached_profile_set_features is None:
            self.profile_set_encoder.eval()
            with torch.inference_mode():
                self._cached_profile_set_features = self._set_aware_profile_features(
                    self._profile_features_for_inference()
                )
        return self._cached_profile_set_features

    def _profile_bias_for_inference(self):
        if self.loss_type != "crm" or self.crm_bias != "profile" or self.profile_bias_encoder is None:
            return None
        if self._cached_profile_bias is None:
            self.profile_bias_encoder.eval()
            with torch.inference_mode():
                self._cached_profile_bias = self.profile_bias_encoder(
                    self._profile_tensor_for_inference()
                ).squeeze(-1)
        return self._cached_profile_bias

    def _score_query_features_cached(self, query_features):
        q = F.normalize(query_features, dim=-1)
        logits = self._pair_logits(
            q,
            self._profile_set_features_for_inference(),
            m_lowrank=self._profile_lowrank_features_for_inference(),
            set_encoded=self.score_type == "set_aware",
        )
        if self.loss_type == "crm":
            if self.crm_bias == "global" and self.global_crm_bias is not None:
                logits = logits + self.global_crm_bias
            else:
                profile_bias = self._profile_bias_for_inference()
                if profile_bias is not None:
                    logits = logits + profile_bias[None, :]
        return logits

    def _score(self, query_x, profile_x):
        return self._score_query_features(self.query_encoder(query_x), profile_x)

    def _current_blip_alpha(self):
        if self.learn_blip_alpha and self.blip_alpha_logit is not None:
            return torch.sigmoid(self.blip_alpha_logit)
        return torch.tensor(self.blip_alpha, dtype=torch.float32, device=self.device)

    def _current_rccr_temperature(self):
        if self.rccr_log_temperature is not None:
            return torch.exp(self.rccr_log_temperature).clamp(1e-4, 10.0)
        return torch.tensor(self.rccr_temperature, dtype=torch.float32, device=self.device)

    def _build_optimizer(self, param_groups):
        if self.optimizer_type == "adamw":
            return torch.optim.AdamW(param_groups, lr=self.learning_rate)
        if self.optimizer_type == "adam":
            return torch.optim.Adam(param_groups, lr=self.learning_rate)
        if self.optimizer_type == "sgd":
            return torch.optim.SGD(param_groups, lr=self.learning_rate, momentum=0.9, nesterov=True)
        raise ValueError(f"Unknown optimizer_type: {self.optimizer_type}")

    def _build_lr_scheduler(self, optimizer):
        if self.lr_scheduler == "none":
            return None
        if self.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, self.max_iter),
                eta_min=self.learning_rate * self.min_lr_factor,
            )
        if self.lr_scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_step_size,
                gamma=self.lr_gamma,
            )
        if self.lr_scheduler == "plateau":
            mode = "min" if self.monitor_metric == "avg_cost" else "max"
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=mode,
                factor=self.lr_gamma,
                patience=self.lr_plateau_patience,
                min_lr=self.learning_rate * self.min_lr_factor,
            )
        raise ValueError(f"Unknown lr_scheduler: {self.lr_scheduler}")

    @staticmethod
    def _current_lr(optimizer) -> float:
        return float(optimizer.param_groups[0]["lr"])

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
            q2m_log_probs = F.log_softmax(logits, dim=-1)
            q2m_loss = -(target_dist * q2m_log_probs).sum(dim=-1).mean()

        model_mass = target_dist.sum(dim=0)
        valid_models = model_mass > 1e-12
        if not torch.any(valid_models):
            return q2m_loss
        m2q_target = target_dist[:, valid_models].T / model_mass[valid_models, None].clamp_min(1e-12)
        m2q_log_probs = F.log_softmax(logits[:, valid_models].T, dim=-1)
        m2q_loss = -(m2q_target * m2q_log_probs).sum(dim=-1).mean()
        return 0.5 * (q2m_loss + m2q_loss)

    def _blip_itc_loss(self, logits, momentum_logits, batch_target):
        base_target = self._target_distribution(logits, batch_target)
        alpha = self._current_blip_alpha()

        with torch.no_grad():
            teacher_q2m = F.softmax(momentum_logits, dim=-1)
        q2m_target = (1.0 - alpha) * base_target + alpha * teacher_q2m
        q2m_log_probs = F.log_softmax(logits, dim=-1)
        q2m_loss = -(q2m_target * q2m_log_probs).sum(dim=-1).mean()

        model_mass = base_target.sum(dim=0)
        valid_models = model_mass > 1e-12
        if not torch.any(valid_models):
            return q2m_loss

        with torch.no_grad():
            base_m2q = base_target[:, valid_models].T / model_mass[valid_models, None].clamp_min(1e-12)
            teacher_m2q = F.softmax(momentum_logits[:, valid_models].T, dim=-1)
        m2q_target = (1.0 - alpha) * base_m2q + alpha * teacher_m2q
        m2q_log_probs = F.log_softmax(logits[:, valid_models].T, dim=-1)
        m2q_loss = -(m2q_target * m2q_log_probs).sum(dim=-1).mean()
        return 0.5 * (q2m_loss + m2q_loss)

    def _compute_loss(self, logits, batch_target, momentum_logits=None):
        if self.loss_type == "softmax":
            if self.use_soft_labels:
                log_probs = F.log_softmax(logits, dim=-1)
                return -(batch_target * log_probs).sum(dim=-1).mean()
            return F.cross_entropy(logits, batch_target)

        if self.loss_type in {"clip", "clip_relevance"}:
            return self._clip_loss(logits, batch_target)

        if self.loss_type == "blip":
            if momentum_logits is None:
                return self._clip_loss(logits, batch_target)
            return self._blip_itc_loss(logits, momentum_logits, batch_target)

        if self.loss_type == "crm":
            batch_target = self._target_distribution(logits, batch_target)
            return F.binary_cross_entropy_with_logits(logits, batch_target)

        raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def _rccr_loss(self, query_features, logits, batch_target):
        if self.rccr_weight <= 0 or query_features.shape[0] <= 1:
            return torch.tensor(0.0, dtype=query_features.dtype, device=query_features.device)

        target_dist = self._target_distribution(logits, batch_target).float().clamp_min(0.0)
        target_mass = target_dist.sum(dim=1, keepdim=True)
        valid = target_mass.squeeze(1) > 1e-12
        if int(valid.sum().item()) <= 1:
            return torch.tensor(0.0, dtype=query_features.dtype, device=query_features.device)

        q = F.normalize(query_features[valid], dim=-1)
        t = target_dist[valid] / target_mass[valid].clamp_min(1e-12)
        pos_weight = t @ t.T
        pos_weight.fill_diagonal_(0.0)

        pos_mass = pos_weight.sum(dim=1, keepdim=True)
        has_pos = pos_mass.squeeze(1) > 1e-12
        if not torch.any(has_pos):
            return torch.tensor(0.0, dtype=query_features.dtype, device=query_features.device)

        pos_weight = pos_weight / pos_mass.clamp_min(1e-12)
        sample_logits = (q @ q.T) / self._current_rccr_temperature().clamp_min(1e-6)
        sample_logits = sample_logits.masked_fill(
            torch.eye(sample_logits.shape[0], dtype=torch.bool, device=sample_logits.device),
            -torch.finfo(sample_logits.dtype).max,
        )
        log_probs = F.log_softmax(sample_logits, dim=1)
        return -(pos_weight[has_pos] * log_probs[has_pos]).sum(dim=1).mean()

    def _eval_metrics(self, X_scaled: np.ndarray, Y: np.ndarray, C: np.ndarray) -> Dict[str, float]:
        preds = self.predict(X=X_scaled, already_scaled=True)
        correct = Y[np.arange(len(preds)), preds]
        costs = C[np.arange(len(preds)), preds]
        metrics = {
            "accuracy": float(correct.mean()),
            "avg_cost": float(costs.mean()),
        }
        if self._rank_score_cmin is not None and self._rank_score_cmax is not None:
            try:
                from routers.utils.rank_score import rank_score

                metrics["rank_score"] = float(rank_score(
                    metrics["accuracy"],
                    metrics["avg_cost"],
                    self._rank_score_cmin,
                    self._rank_score_cmax,
                    beta=self._rank_score_beta,
                ))
            except Exception:
                pass
        return metrics

    def fit(
        self,
        Y: np.ndarray,
        C: np.ndarray,
        meta,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        model_mapping: Optional[Dict[int, str]] = None,
        costs: Optional[np.ndarray] = None,
        Y_dev: Optional[np.ndarray] = None,
        C_dev: Optional[np.ndarray] = None,
        X_dev: Optional[np.ndarray] = None,
        X_text_dev: Optional[np.ndarray] = None,
        X_vision_dev: Optional[np.ndarray] = None,
        meta_dev=None,
        **kwargs,
    ):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        Xf = self._fuse(meta, X, X_text, X_vision)
        self.query_scaler = NumpyStandardScaler()
        X_scaled = self.query_scaler.fit_transform(Xf)

        raw_profile, profile_names = self._load_profile()
        if raw_profile.shape[0] != Y.shape[1]:
            raise ValueError(f"Profile has {raw_profile.shape[0]} models, but Y has {Y.shape[1]} columns")
        self.profile_scaler = NumpyStandardScaler()
        self.model_profile = self.profile_scaler.fit_transform(raw_profile)
        self.model_names = profile_names.tolist()

        if model_mapping is None:
            model_mapping = {i: self.model_names[i] for i in range(len(self.model_names))}
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}
        self.costs = np.asarray(costs, dtype=float) if costs is not None else C.mean(axis=0)

        self._rank_score_cmin = kwargs.get("cmin", None)
        self._rank_score_cmax = kwargs.get("cmax", None)
        self._rank_score_beta = float(kwargs.get("rank_score_beta", kwargs.get("beta", 0.1)))

        y_or_T = self._targets(Y, C)
        self._build_modules(query_dim=X_scaled.shape[1], profile_dim=self.model_profile.shape[1])

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        if self.loss_type == "crm":
            target_tensor = torch.tensor(self._crm_targets(Y, C, y_or_T), dtype=torch.float32)
        elif self.loss_type == "clip_relevance":
            target_tensor = torch.tensor(self._relevance_targets(Y, C), dtype=torch.float32)
        elif self.use_soft_labels:
            target_tensor = torch.tensor(y_or_T, dtype=torch.float32)
        else:
            target_tensor = torch.tensor(y_or_T, dtype=torch.long)
        dataset = TensorDataset(X_tensor, target_tensor)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, generator=generator)

        profile_tensor = torch.tensor(self.model_profile, dtype=torch.float32, device=self.device)
        params = list(self.query_encoder.parameters()) + list(self.profile_encoder.parameters())
        if self.query_bilinear is not None:
            params.extend(self.query_bilinear.parameters())
        if self.profile_bilinear is not None:
            params.extend(self.profile_bilinear.parameters())
        if self.profile_set_encoder is not None:
            params.extend(self.profile_set_encoder.parameters())
        if self.query_cross_attention is not None:
            params.extend(self.query_cross_attention.parameters())
        if self.global_crm_bias is not None:
            params.append(self.global_crm_bias)
        if self.profile_bias_encoder is not None:
            params.extend(self.profile_bias_encoder.parameters())
        param_groups = [{"params": params, "weight_decay": self.weight_decay}]
        if self.blip_alpha_logit is not None:
            param_groups.append({"params": [self.blip_alpha_logit], "weight_decay": 0.0})
        if self.rccr_log_temperature is not None:
            param_groups.append({"params": [self.rccr_log_temperature], "weight_decay": 0.0})
        optimizer = self._build_optimizer(param_groups)
        scheduler = self._build_lr_scheduler(optimizer)

        X_dev_scaled = None
        if Y_dev is not None and C_dev is not None:
            Xf_dev = self._fuse(meta_dev, X_dev, X_text_dev, X_vision_dev)
            X_dev_scaled = self.query_scaler.transform(Xf_dev)

        if self.verbose > 0:
            label = "soft" if self.use_soft_labels else "hard"
            print(f"  SCOPE-Router training on {self.device}: {len(X_scaled)} samples, {Y.shape[1]} models")
            print(
                f"  query_dim={X_scaled.shape[1]}, profile_dim={self.model_profile.shape[1]}, "
                f"emb_dim={self.embedding_dim}, labels={label}, loss={self.loss_type}, "
                f"optimizer={self.optimizer_type}, lr={self.learning_rate:g}, "
                f"lr_scheduler={self.lr_scheduler}, min_lr_factor={self.min_lr_factor:g}, "
                f"crm_target={self.crm_target}, crm_bias={self.crm_bias}, "
                f"blip_alpha={self.blip_alpha:g}, learn_blip_alpha={self.learn_blip_alpha}, "
                f"blip_momentum={self.blip_momentum:g}, "
                f"rccr_weight={self.rccr_weight:g}, "
                f"rccr_temperature={self.rccr_temperature:g}, "
                f"learn_rccr_temperature={self.learn_rccr_temperature}"
            )

        best_value = -float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, self.max_iter + 1):
            self.query_encoder.train()
            self.profile_encoder.train()
            if self.query_bilinear is not None:
                self.query_bilinear.train()
            if self.profile_bilinear is not None:
                self.profile_bilinear.train()
            if self.profile_set_encoder is not None:
                self.profile_set_encoder.train()
            if self.query_cross_attention is not None:
                self.query_cross_attention.train()
            losses = []
            for batch in loader:
                batch_X, batch_target = batch
                batch_X = batch_X.to(self.device)
                batch_target = batch_target.to(self.device)
                optimizer.zero_grad()
                query_features = self.query_encoder(batch_X)
                logits = self._score_query_features(query_features, profile_tensor)
                momentum_logits = None
                if self.loss_type == "blip":
                    with torch.no_grad():
                        self.query_encoder_m.eval()
                        self.profile_encoder_m.eval()
                        momentum_logits = self._score_with_encoders(
                            batch_X,
                            profile_tensor,
                            self.query_encoder_m,
                            self.profile_encoder_m,
                        )
                base_loss = self._compute_loss(logits, batch_target, momentum_logits=momentum_logits)
                sample_loss = self._rccr_loss(query_features, logits, batch_target)
                loss = base_loss + self.rccr_weight * sample_loss
                loss.backward()
                optimizer.step()
                self._update_momentum()
                losses.append(float(loss.item()))

            self._invalidate_profile_cache()
            train_metrics = self._eval_metrics(X_scaled, Y, C)
            metric_value = train_metrics.get(self.monitor_metric, train_metrics["accuracy"])
            dev_metrics = None
            if X_dev_scaled is not None:
                dev_metrics = self._eval_metrics(X_dev_scaled, Y_dev, C_dev)
                metric_value = dev_metrics.get(self.monitor_metric, dev_metrics["accuracy"])

            improved = metric_value > best_value + 1e-6
            if improved:
                best_value = metric_value
                self.best_epoch = epoch
                self.best_dev_metrics = dev_metrics or train_metrics
                best_state = {
                    "query_encoder": {k: v.detach().cpu().clone() for k, v in self.query_encoder.state_dict().items()},
                    "profile_encoder": {k: v.detach().cpu().clone() for k, v in self.profile_encoder.state_dict().items()},
                    "query_bilinear": (
                        {k: v.detach().cpu().clone() for k, v in self.query_bilinear.state_dict().items()}
                        if self.query_bilinear is not None else None
                    ),
                    "profile_bilinear": (
                        {k: v.detach().cpu().clone() for k, v in self.profile_bilinear.state_dict().items()}
                        if self.profile_bilinear is not None else None
                    ),
                    "profile_set_encoder": (
                        {k: v.detach().cpu().clone() for k, v in self.profile_set_encoder.state_dict().items()}
                        if self.profile_set_encoder is not None else None
                    ),
                    "query_cross_attention": (
                        {k: v.detach().cpu().clone() for k, v in self.query_cross_attention.state_dict().items()}
                        if self.query_cross_attention is not None else None
                    ),
                    "profile_bias_encoder": (
                        {k: v.detach().cpu().clone() for k, v in self.profile_bias_encoder.state_dict().items()}
                        if self.profile_bias_encoder is not None else None
                    ),
                    "global_crm_bias": (
                        self.global_crm_bias.detach().cpu().clone()
                        if self.global_crm_bias is not None else None
                    ),
                    "blip_alpha_logit": (
                        self.blip_alpha_logit.detach().cpu().clone()
                        if self.blip_alpha_logit is not None else None
                    ),
                    "rccr_log_temperature": (
                        self.rccr_log_temperature.detach().cpu().clone()
                        if self.rccr_log_temperature is not None else None
                    ),
                }
                bad_epochs = 0
            else:
                bad_epochs += 1

            if scheduler is not None:
                if self.lr_scheduler == "plateau":
                    scheduler.step(metric_value)
                else:
                    scheduler.step()

            if self.verbose > 0 and (epoch == 1 or epoch % 10 == 0 or improved):
                msg = f"  Epoch {epoch:3d}: loss={np.mean(losses):.4f} train_acc={train_metrics['accuracy']:.4f}"
                if dev_metrics is not None:
                    msg += f" dev_acc={dev_metrics['accuracy']:.4f} dev_cost=${dev_metrics['avg_cost']:.6f}"
                    if "rank_score" in dev_metrics:
                        msg += f" rank_score={dev_metrics['rank_score']:.4f}"
                if self.loss_type == "blip" and self.learn_blip_alpha:
                    msg += f" alpha={float(self._current_blip_alpha().detach().cpu()):.4f}"
                if self.rccr_log_temperature is not None:
                    msg += f" ss_temp={float(self._current_rccr_temperature().detach().cpu()):.4f}"
                if self.lr_scheduler != "none":
                    msg += f" lr={self._current_lr(optimizer):.2e}"
                if improved:
                    msg += " *"
                print(msg)

            if self.patience > 0 and bad_epochs >= self.patience:
                if self.verbose > 0:
                    print(f"  Early stopping at epoch {epoch}; best epoch={self.best_epoch}")
                break

        if best_state is not None:
            self.query_encoder.load_state_dict(best_state["query_encoder"])
            self.profile_encoder.load_state_dict(best_state["profile_encoder"])
            if self.query_bilinear is not None and best_state["query_bilinear"] is not None:
                self.query_bilinear.load_state_dict(best_state["query_bilinear"])
            if self.profile_bilinear is not None and best_state["profile_bilinear"] is not None:
                self.profile_bilinear.load_state_dict(best_state["profile_bilinear"])
            if self.profile_set_encoder is not None and best_state["profile_set_encoder"] is not None:
                self.profile_set_encoder.load_state_dict(best_state["profile_set_encoder"])
            if self.query_cross_attention is not None and best_state["query_cross_attention"] is not None:
                self.query_cross_attention.load_state_dict(best_state["query_cross_attention"])
            if self.profile_bias_encoder is not None and best_state["profile_bias_encoder"] is not None:
                self.profile_bias_encoder.load_state_dict(best_state["profile_bias_encoder"])
            if self.global_crm_bias is not None and best_state["global_crm_bias"] is not None:
                self.global_crm_bias.data.copy_(best_state["global_crm_bias"].to(self.device))
            if self.blip_alpha_logit is not None and best_state["blip_alpha_logit"] is not None:
                self.blip_alpha_logit.data.copy_(best_state["blip_alpha_logit"].to(self.device))
            if self.rccr_log_temperature is not None and best_state["rccr_log_temperature"] is not None:
                self.rccr_log_temperature.data.copy_(best_state["rccr_log_temperature"].to(self.device))

        self._invalidate_profile_cache()
        return self

    def predict(
        self,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
        already_scaled: bool = False,
    ) -> np.ndarray:
        Xf = X.astype(np.float32) if already_scaled else self.query_scaler.transform(self._fuse(meta, X, X_text, X_vision))
        self.query_encoder.eval()
        self.profile_encoder.eval()
        if self.query_bilinear is not None:
            self.query_bilinear.eval()
        if self.profile_bilinear is not None:
            self.profile_bilinear.eval()
        if self.profile_set_encoder is not None:
            self.profile_set_encoder.eval()
        if self.query_cross_attention is not None:
            self.query_cross_attention.eval()
        if self.profile_bias_encoder is not None:
            self.profile_bias_encoder.eval()
        with torch.inference_mode():
            X_tensor = torch.as_tensor(Xf, dtype=torch.float32, device=self.device)
            logits = self._score_query_features_cached(self.query_encoder(X_tensor))
            return logits.argmax(dim=1).cpu().numpy()

    def predict_proba(
        self,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None,
    ) -> np.ndarray:
        Xf = self.query_scaler.transform(self._fuse(meta, X, X_text, X_vision))
        self.query_encoder.eval()
        self.profile_encoder.eval()
        if self.query_bilinear is not None:
            self.query_bilinear.eval()
        if self.profile_bilinear is not None:
            self.profile_bilinear.eval()
        if self.profile_set_encoder is not None:
            self.profile_set_encoder.eval()
        if self.query_cross_attention is not None:
            self.query_cross_attention.eval()
        if self.profile_bias_encoder is not None:
            self.profile_bias_encoder.eval()
        with torch.inference_mode():
            X_tensor = torch.as_tensor(Xf, dtype=torch.float32, device=self.device)
            logits = self._score_query_features_cached(self.query_encoder(X_tensor))
            if self.loss_type == "crm":
                probs = torch.sigmoid(logits)
            else:
                probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy()

    def save(self, path: str):
        data = {
            "config": {
                "profile_path": self.profile_path,
                "embedding_dim": self.embedding_dim,
                "query_hidden_dim": self.query_hidden_dim,
                "profile_hidden_dim": self.profile_hidden_dim,
                "query_layers": self.query_layers,
                "profile_layers": self.profile_layers,
                "dropout": self.dropout,
                "fusion_method": self.fusion_method,
                "text_weight": self.text_weight,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "optimizer_type": self.optimizer_type,
                "lr_scheduler": self.lr_scheduler,
                "min_lr_factor": self.min_lr_factor,
                "lr_step_size": self.lr_step_size,
                "lr_gamma": self.lr_gamma,
                "lr_plateau_patience": self.lr_plateau_patience,
                "batch_size": self.batch_size,
                "max_iter": self.max_iter,
                "temperature": self.temperature,
                "score_type": self.score_type,
                "bilinear_rank": self.bilinear_rank,
                "bilinear_residual_weight": self.bilinear_residual_weight,
                "set_encoder_layers": self.set_encoder_layers,
                "set_encoder_heads": self.set_encoder_heads,
                "set_encoder_ff_dim": self.set_encoder_ff_dim,
                "set_encoder_residual_weight": self.set_encoder_residual_weight,
                "set_encoder_normalize_input": self.set_encoder_normalize_input,
                "set_encoder_context_only": self.set_encoder_context_only,
                "cross_attention_heads": self.cross_attention_heads,
                "cross_attention_residual_weight": self.cross_attention_residual_weight,
                "loss_type": self.loss_type,
                "crm_target": self.crm_target,
                "crm_bias": self.crm_bias,
                "blip_match_weight": self.blip_match_weight,
                "blip_alpha": self.blip_alpha,
                "learn_blip_alpha": self.learn_blip_alpha,
                "blip_momentum": self.blip_momentum,
                "rccr_weight": self.rccr_weight,
                "rccr_temperature": self.rccr_temperature,
                "learn_rccr_temperature": self.learn_rccr_temperature,
                "train_lambda": self.train_lambda,
                "use_soft_labels": self.use_soft_labels,
                "cost_scale": self.cost_scale,
                "patience": self.patience,
                "monitor_metric": self.monitor_metric,
                "device": self.device,
                "random_state": self.random_state,
                "verbose": self.verbose,
            },
            "query_scaler": self.query_scaler,
            "profile_scaler": self.profile_scaler,
            "model_profile": self.model_profile,
            "model_names": self.model_names,
            "model_mapping": self.model_mapping,
            "reverse_mapping": self.reverse_mapping,
            "costs": self.costs,
            "best_epoch": self.best_epoch,
            "best_dev_metrics": self.best_dev_metrics,
            "rank_score": {
                "cmin": self._rank_score_cmin,
                "cmax": self._rank_score_cmax,
                "beta": self._rank_score_beta,
            },
            "last_dims": {
                "query": self._last_query_dim,
                "profile": self._last_profile_dim,
                "text": self._last_text_dim,
                "vision": self._last_vision_dim,
            },
            "query_encoder_state": (
                {k: v.detach().cpu() for k, v in self.query_encoder.state_dict().items()}
                if self.query_encoder is not None else None
            ),
            "profile_encoder_state": (
                {k: v.detach().cpu() for k, v in self.profile_encoder.state_dict().items()}
                if self.profile_encoder is not None else None
            ),
            "query_bilinear_state": (
                {k: v.detach().cpu() for k, v in self.query_bilinear.state_dict().items()}
                if self.query_bilinear is not None else None
            ),
            "profile_bilinear_state": (
                {k: v.detach().cpu() for k, v in self.profile_bilinear.state_dict().items()}
                if self.profile_bilinear is not None else None
            ),
            "profile_set_encoder_state": (
                {k: v.detach().cpu() for k, v in self.profile_set_encoder.state_dict().items()}
                if self.profile_set_encoder is not None else None
            ),
            "query_cross_attention_state": (
                {k: v.detach().cpu() for k, v in self.query_cross_attention.state_dict().items()}
                if self.query_cross_attention is not None else None
            ),
            "profile_bias_encoder_state": (
                {k: v.detach().cpu() for k, v in self.profile_bias_encoder.state_dict().items()}
                if self.profile_bias_encoder is not None else None
            ),
            "global_crm_bias": (
                self.global_crm_bias.detach().cpu()
                if self.global_crm_bias is not None else None
            ),
            "blip_alpha_logit": (
                self.blip_alpha_logit.detach().cpu()
                if self.blip_alpha_logit is not None else None
            ),
            "rccr_log_temperature": (
                self.rccr_log_temperature.detach().cpu()
                if self.rccr_log_temperature is not None else None
            ),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        config = data["config"]
        config.setdefault("loss_type", "softmax")
        config.setdefault("optimizer_type", "adamw")
        config.setdefault("lr_scheduler", "none")
        config.setdefault("min_lr_factor", 0.05)
        config.setdefault("lr_step_size", 20)
        config.setdefault("lr_gamma", 0.5)
        config.setdefault("lr_plateau_patience", 5)
        config.setdefault("crm_target", "soft")
        config.setdefault("crm_bias", "none")
        config.setdefault("score_type", "dot")
        config.setdefault("bilinear_rank", 16)
        config.setdefault("bilinear_residual_weight", 1.0)
        config.setdefault("set_encoder_layers", 1)
        config.setdefault("set_encoder_heads", 4)
        config.setdefault("set_encoder_ff_dim", 128)
        config.setdefault("set_encoder_residual_weight", 0.25)
        config.setdefault("set_encoder_normalize_input", True)
        config.setdefault("set_encoder_context_only", False)
        config.setdefault("cross_attention_heads", 4)
        config.setdefault("cross_attention_residual_weight", 0.25)
        config.setdefault("blip_match_weight", 1.0)
        config.setdefault("blip_alpha", 0.4)
        config.setdefault("learn_blip_alpha", False)
        config.setdefault("blip_momentum", 0.995)
        config.setdefault("rccr_weight", 0.0)
        config.setdefault("rccr_temperature", 0.1)
        config.setdefault("learn_rccr_temperature", False)
        router = cls(**config)
        router.query_scaler = data["query_scaler"]
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
        dims = data.get("last_dims", {})
        router._last_text_dim = dims.get("text")
        router._last_vision_dim = dims.get("vision")
        router._build_modules(query_dim=dims["query"], profile_dim=dims["profile"])
        router.query_encoder.load_state_dict(data["query_encoder_state"])
        router.profile_encoder.load_state_dict(data["profile_encoder_state"])
        if router.query_bilinear is not None and data.get("query_bilinear_state") is not None:
            router.query_bilinear.load_state_dict(data["query_bilinear_state"])
        if router.profile_bilinear is not None and data.get("profile_bilinear_state") is not None:
            router.profile_bilinear.load_state_dict(data["profile_bilinear_state"])
        if router.profile_set_encoder is not None and data.get("profile_set_encoder_state") is not None:
            router.profile_set_encoder.load_state_dict(data["profile_set_encoder_state"])
        if router.query_cross_attention is not None and data.get("query_cross_attention_state") is not None:
            router.query_cross_attention.load_state_dict(data["query_cross_attention_state"])
        if router.loss_type == "blip":
            if router.blip_alpha_logit is not None and data.get("blip_alpha_logit") is not None:
                router.blip_alpha_logit.data.copy_(data["blip_alpha_logit"].to(router.device))
            router._copy_to_momentum()
            router.query_encoder_m.eval()
            router.profile_encoder_m.eval()
        if router.profile_bias_encoder is not None and data.get("profile_bias_encoder_state") is not None:
            router.profile_bias_encoder.load_state_dict(data["profile_bias_encoder_state"])
        if router.global_crm_bias is not None and data.get("global_crm_bias") is not None:
            router.global_crm_bias.data.copy_(data["global_crm_bias"].to(router.device))
        if router.rccr_log_temperature is not None and data.get("rccr_log_temperature") is not None:
            router.rccr_log_temperature.data.copy_(data["rccr_log_temperature"].to(router.device))
        router._invalidate_profile_cache()
        return router

    def __repr__(self):
        mode = "soft" if self.use_soft_labels else "hard"
        return f"ScopeRouter(profile={Path(self.profile_path).name}, emb_dim={self.embedding_dim}, labels={mode})"
