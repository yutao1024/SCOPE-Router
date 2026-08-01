#!/usr/bin/env python3
"""
LinearR: softmax（，）

RouterEval/RouterBenchLinearR，LogisticRegression。
（）（）。
"""

import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional, Dict
import warnings

from routers.common import RouterBase

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# PyTorchGPU
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    optim = None


if HAS_TORCH:
    class PyTorchLogisticRegression(nn.Module):
        """PyTorchLogisticRegression（softmax）"""
        def __init__(self, input_dim, num_classes, penalty='l2', C=1.0):
            super().__init__()
            self.num_classes = num_classes
            self.penalty = penalty
            self.C = C
            # ： -> 
            self.linear = nn.Linear(input_dim, num_classes)
        
        def forward(self, x):
            return self.linear(x)
    
    class PyTorchStandardScaler:
        """PyTorchStandardScaler"""
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
        
        def fit(self, X):
            # Xnumpytorch tensor
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            self.mean_ = X.mean(dim=0, keepdim=True)
            self.scale_ = X.std(dim=0, keepdim=True)
            # NOTE: (translated from Chinese)
            self.scale_[self.scale_ == 0] = 1.0
            return self
        
        def transform(self, X):
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            return (X - self.mean_) / self.scale_
        
        def fit_transform(self, X):
            return self.fit(X).transform(X)
        
        def inverse_transform(self, X):
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X)
            return X * self.scale_ + self.mean_
else:
    PyTorchLogisticRegression = None
    PyTorchStandardScaler = None


class LinearRouter(RouterBase):
    """
    LinearR: softmax
    
    : p_θ(m|x) = softmax(W_m^T φ(x))
    : 
      - : ，
      - : Soft-CE，
    : argmax_m p_θ(m|x)
    """
    
    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        max_iter: int = 200,
        fusion_method: str = 'normalize_concat',
        text_weight: float = 0.5,
        multiclass: str = 'multinomial',
        solver: Optional[str] = None,
        verbose: int = 1,
        device: Optional[str] = None,
        text_encoder: str = "BAAI/bge-m3",
        vision_encoder: str = "facebook/dinov2-base",
        use_soft_labels: bool = False,
        train_lambda: float = 0.0,
        enable_monitoring: bool = True,
        patience: int = 100,
        monitor_metric: str = 'rank_score'
    ):
        """
        Args:
            penalty:  ('l1' or 'l2')
            C: （）
            max_iter: 
            fusion_method:  ('concat', 'average', 'weighted_average', 'normalize_concat')
            text_weight: （weighted_average）
            multiclass:  ('multinomial' or 'ovr')
            solver: ，multiclass
            verbose:  (0=, 1=, >1=)
            device:  ('cuda', 'cpu', None=)
            text_encoder: （: "BAAI/bge-m3"）
            vision_encoder: （: "facebook/dinov2-base"）
            use_soft_labels: （False）
            train_lambda: lambda（0.0）
            enable_monitoring: （False）
            patience: （20）
            monitor_metric: （'accuracy'）
        """
        if not HAS_SKLEARN:
            raise ImportError("Please install: pip install scikit-learn")
        
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.multiclass = multiclass
        self.verbose = verbose
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.use_soft_labels = use_soft_labels
        self.train_lambda = train_lambda
        self.enable_monitoring = enable_monitoring
        self.patience = patience
        self.monitor_metric = monitor_metric
        
        # NOTE: (translated from Chinese)
        if device is None:
            # ：GPU（PyTorch）
            if HAS_TORCH:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.use_gpu = (self.device == 'cuda')
            else:
                self.device = 'cpu'
                self.use_gpu = False
        else:
            self.device = device
            if device == 'cuda':
                if HAS_TORCH and torch.cuda.is_available():
                    self.use_gpu = True
                else:
                    warnings.warn("PyTorchGPU，CPU")
                    self.use_gpu = False
                    self.device = 'cpu'
            else:
                self.use_gpu = False
        
        self.pytorch_model = None  # PyTorch（）
        
        # solver
        if solver is None:
            # ，epoch（adam/sgd）
            if enable_monitoring and HAS_TORCH:
                solver = 'adam'
            elif multiclass == 'multinomial':
                solver = 'lbfgs'
            else:
                solver = 'liblinear'
        self.solver = solver
        
        self.clf = None
        self.scaler = None
        self.model_mapping = None  # {model_index: model_id}
        self.reverse_mapping = None  # {model_id: model_index}
        self.costs = None  # np.ndarray(K,) - 
        # Rank score configuration (used for monitoring on dev)
        self._rank_score_cmin = None
        self._rank_score_cmax = None
        self._rank_score_beta = 0.1
    
    def _fuse(
        self,
        meta,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """（KNNRouter）"""
        from routers.utils.fusion import fuse_embeddings
        
        if X is not None:
            return X
        
        if X_text is not None and X_vision is not None:
            return fuse_embeddings(X_text, X_vision, self.fusion_method, self.text_weight)
        
        if meta is not None:
            if 'embedding' in meta.columns:
                return np.vstack(meta['embedding'].values)
            
            if 'text_embedding' in meta.columns and 'vision_embedding' in meta.columns:
                Xt = np.vstack(meta['text_embedding'].values)
                Xv = np.vstack(meta['vision_embedding'].values)
                return fuse_embeddings(Xt, Xv, self.fusion_method, self.text_weight)
        
        raise ValueError(
            "Features required：X  (X_text, X_vision)  meta  embedding "
        )
    
    @staticmethod
    def _derive_labels(Y: np.ndarray, C: np.ndarray, use_soft_labels: bool = False, 
                      soft_label_lambda: float = 0.0) -> np.ndarray:
        """
        Y/C
        
        Args:
            Y:  (N x K)
            C:  (N x K)
            use_soft_labels: （False，）
            soft_label_lambda: lambda（）
        
        Returns:
            use_soft_labels=False: (labels, valid_indices)
            use_soft_labels=True: (soft_targets, valid_indices)
        """
        if not use_soft_labels:
            # ： (one-hot)
            N, K = Y.shape
            labels = []
            valid_indices = []
            
            for i in range(N):
                correct = np.where(Y[i] == 1)[0]
                if len(correct) > 0:
                    # ，
                    costs = C[i, correct]
                    best_idx = correct[np.argmin(costs)]
                    labels.append(best_idx)
                    valid_indices.append(i)
                # else: 
            
            return np.array(labels), np.array(valid_indices)
        else:
            # NOTE: (translated from Chinese)
            from routers.utils.soft_targets import build_soft_targets
            
            T = build_soft_targets(Y, C, lam=soft_label_lambda, scheme='exp', 
                                  fallback='cheapest')
            
            # NOTE: (translated from Chinese)
            has_correct = (Y.sum(axis=1) > 0)
            valid_indices = np.where(has_correct)[0]
            
            return T, valid_indices
    
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
        meta_dev = None,
        monitor_output_dir: Optional[Path] = None,
        **kwargs
    ):
        """
        LinearR
        
        Args:
            Y:  (N x K)
            C:  (N x K)
            meta: DataFrame
            X:  (N x D)，
            X_text:  (N x D_text)，
            X_vision:  (N x D_vision)，
            model_mapping:  {0: 'model1', ...}
            costs:  (K,)，
            Y_dev: Dev（）
            C_dev: Dev（）
            X_dev, X_text_dev, X_vision_dev, meta_dev: Dev（）
            monitor_output_dir: 
        """
        # NOTE: (translated from Chinese)
        Xf = self._fuse(meta, X, X_text, X_vision)
        
        # NOTE: (translated from Chinese)
        y_or_T, valid_indices = self._derive_labels(Y, C, 
                                                    use_soft_labels=self.use_soft_labels,
                                                    soft_label_lambda=self.train_lambda)
        
        if len(valid_indices) < len(Y):
            print(f"  Filtering training samples: {len(Y)} -> {len(valid_indices)} "
                  f"(removed {len(Y) - len(valid_indices)} no correct model)")
            Xf = Xf[valid_indices]
            Y = Y[valid_indices]
            C = C[valid_indices]
            if self.use_soft_labels:
                y_or_T = y_or_T[valid_indices]
        
        # NOTE: (translated from Chinese)
        print(f"   (: {self.device})...")
        if self.use_gpu and HAS_TORCH:
            # PyTorchStandardScaler
            self.scaler = PyTorchStandardScaler()
            X_scaled = self.scaler.fit_transform(Xf)
            X_scaled_np = X_scaled.cpu().numpy() if isinstance(X_scaled, torch.Tensor) else X_scaled
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(Xf)
            X_scaled_np = X_scaled
        
        # LogisticRegression
        device_str = "GPU (PyTorch)" if self.use_gpu else "CPU (sklearn)"
        mode_str = f", soft_labels λ={self.train_lambda}" if self.use_soft_labels else ""
        print(f"  trainingLogisticRegression ({device_str}{mode_str}, max_iter={self.max_iter})...")
        
        if self.use_soft_labels:
            num_classes = Y.shape[1]  # K models
        else:
            num_classes = len(np.unique(y_or_T))
        
        if self.verbose > 0:
            print(f"    samples: {len(y_or_T)}, : {X_scaled_np.shape[1]}, : {num_classes}")
        
        # model_mapping（checkpoint）
        if model_mapping is None:
            K = Y.shape[1]
            model_mapping = {i: f'model_{i}' for i in range(K)}
        
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}
        
        # NOTE: (translated from Chinese)
        self._last_X_shape = X_scaled_np.shape
        
        # NOTE: (translated from Chinese)
        if costs is not None:
            self.costs = np.asarray(costs, dtype=float)
        elif 'costs' in kwargs:
            self.costs = np.asarray(kwargs['costs'], dtype=float)
        else:
            # ，C
            self.costs = C.mean(axis=0)

        # Rank score bounds for dev monitoring (optional but recommended)
        # Prefer explicit bounds from caller (e.g. cost_bounds.json), else fallback to observed costs.
        try:
            cmin = kwargs.get("cmin", None)
            cmax = kwargs.get("cmax", None)
            beta = kwargs.get("rank_score_beta", kwargs.get("beta", 0.1))
            if cmin is None or cmax is None:
                # Use global observed bounds (train/dev) as a fallback
                cmin = float(np.min(C_dev)) if C_dev is not None else float(np.min(C))
                cmax = float(np.max(C_dev)) if C_dev is not None else float(np.max(C))
            self._rank_score_cmin = float(cmin)
            self._rank_score_cmax = float(cmax)
            self._rank_score_beta = float(beta)
        except Exception:
            # Never fail training because of bounds parsing
            self._rank_score_cmin = None
            self._rank_score_cmax = None
            self._rank_score_beta = 0.1
        
        if self.use_gpu and HAS_TORCH:
            # dev（，dev）
            X_dev_scaled_np = None
            Y_dev_filtered = None
            C_dev_filtered = None
            
            if self.enable_monitoring and Y_dev is not None:
                # dev（）
                Xf_dev = self._fuse(meta_dev, X_dev, X_text_dev, X_vision_dev)
                
                # dev
                X_dev_scaled = self.scaler.transform(Xf_dev)
                X_dev_scaled_np = X_dev_scaled.cpu().numpy() if isinstance(X_dev_scaled, torch.Tensor) else X_dev_scaled
                
                # dev
                Y_dev_filtered = Y_dev
                C_dev_filtered = C_dev
                
                if self.verbose > 0:
                    print(f"  Dev set: {len(Y_dev)} samples（，）")
            
            # PyTorch GPU
            self._train_pytorch(
                X_scaled_np, y_or_T, num_classes, 
                use_soft=self.use_soft_labels,
                X_dev=X_dev_scaled_np,
                Y_dev=Y_dev_filtered,
                C_dev=C_dev_filtered,
                monitor_output_dir=monitor_output_dir
            )
        else:
            # sklearn CPU
            self.clf = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                max_iter=self.max_iter,
                multi_class=self.multiclass,
                solver=self.solver,
                random_state=42,
                verbose=max(0, self.verbose - 1)
            )
            
            if self.verbose > 0:
                print("    Training......", end="", flush=True)
            
            # sklearn，
            if self.use_soft_labels:
                # NOTE: (translated from Chinese)
                y_hard = y_or_T.argmax(axis=1)
            else:
                y_hard = y_or_T
            
            self.clf.fit(X_scaled_np, y_hard)
            
            # NOTE: (translated from Chinese)
            train_accuracy = self.clf.score(X_scaled_np, y_hard)
        
        if self.verbose > 0:
            print(" ✓")
            if not (self.use_gpu and HAS_TORCH):  # PyTorch_train_pytorch
                print(f"    Train setAccuracy: {train_accuracy:.4f}")
        
        # model_mapping、costs
        print(f"  LinearRTraining complete: ={self.device}, Fusion method={self.fusion_method}, "
              f"={X_scaled_np.shape[1]}, Num models={len(self.model_mapping)}")
    
    def _train_pytorch(
        self, 
        X_scaled: np.ndarray, 
        y_or_T: np.ndarray, 
        num_classes: int, 
        use_soft: bool = False,
        X_dev: Optional[np.ndarray] = None,
        Y_dev: Optional[np.ndarray] = None,
        C_dev: Optional[np.ndarray] = None,
        monitor_output_dir: Optional[Path] = None
    ):
        """PyTorch GPUtrainingLogisticRegression（training）"""
        if not HAS_TORCH or PyTorchLogisticRegression is None:
            raise ImportError("Please installPyTorchGPUtraining")
        
        # NOTE: (translated from Chinese)
        use_monitoring = (self.enable_monitoring and X_dev is not None and 
                         Y_dev is not None and C_dev is not None)
        
        if use_monitoring:
            from routers.utils.training_monitor import TrainingMonitor
            if monitor_output_dir is None:
                import tempfile
                monitor_output_dir = Path(tempfile.mkdtemp(prefix='linear_router_'))
            
            monitor = TrainingMonitor(
                output_dir=monitor_output_dir,
                metric=self.monitor_metric,
                patience=self.patience,
                maximize=True if self.monitor_metric in ['accuracy', 'rank_score'] else False,
                save_checkpoints=True,  # savedevbest model
                verbose=self.verbose
            )
            monitor.start_training()
        else:
            monitor = None
        
        if self.verbose > 0 and not use_monitoring:
            print("    Training......", end="", flush=True)
        
        # PyTorch
        self.pytorch_model = PyTorchLogisticRegression(
            input_dim=X_scaled.shape[1],
            num_classes=num_classes,
            penalty=self.penalty,
            C=self.C
        ).to(self.device)
        
        # NOTE: (translated from Chinese)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        if use_soft:
            # ：y_or_T (N, K) 
            T_tensor = torch.FloatTensor(y_or_T).to(self.device)
            
            # Soft-CE
            def soft_ce_loss(logits, soft_targets):
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return -torch.sum(soft_targets * log_probs, dim=-1).mean()
            
            criterion = soft_ce_loss
        else:
            # ：y_or_T (N,) 
            y_tensor = torch.LongTensor(y_or_T).to(self.device)
            criterion = nn.CrossEntropyLoss()
        
        # （L-BFGS，Adam）
        if self.solver == 'lbfgs':
            optimizer = optim.LBFGS(
                self.pytorch_model.parameters(),
                lr=1.0,
                max_iter=self.max_iter,
                history_size=10
            )
            
            def closure():
                optimizer.zero_grad()
                outputs = self.pytorch_model(X_tensor)
                if use_soft:
                    loss = criterion(outputs, T_tensor)
                else:
                    loss = criterion(outputs, y_tensor)
                # L2
                if self.penalty == 'l2' and self.C > 0:
                    l2_reg = sum(p.pow(2.0).sum() for p in self.pytorch_model.parameters())
                    loss = loss + (1.0 / (2.0 * self.C)) * l2_reg
                loss.backward()
                return loss
            
            optimizer.step(closure)
            final_epoch = self.max_iter
        else:
            # AdamSGD
            optimizer = optim.Adam(self.pytorch_model.parameters(), lr=0.01)
            final_epoch = 0
            
            # dev（）
            if use_monitoring:
                X_dev_tensor = torch.FloatTensor(X_dev).to(self.device)
            
            for epoch in range(1, self.max_iter + 1):
                # NOTE: (translated from Chinese)
                self.pytorch_model.train()
                optimizer.zero_grad()
                outputs = self.pytorch_model(X_tensor)
                if use_soft:
                    loss = criterion(outputs, T_tensor)
                else:
                    loss = criterion(outputs, y_tensor)
                
                # L2
                if self.penalty == 'l2' and self.C > 0:
                    l2_reg = sum(p.pow(2.0).sum() for p in self.pytorch_model.parameters())
                    loss = loss + (1.0 / (2.0 * self.C)) * l2_reg
                
                loss.backward()
                optimizer.step()
                
                if use_monitoring:
                    monitor.start_epoch(epoch)

                    self.pytorch_model.eval()
                    with torch.no_grad():
                        train_outputs = self.pytorch_model(X_tensor)
                        train_preds = train_outputs.argmax(dim=1).cpu().numpy()
                        if use_soft:
                            y_true = y_or_T.argmax(axis=1)
                        else:
                            y_true = y_or_T
                        train_acc = (train_preds == y_true).mean()
                        train_metrics = {'accuracy': float(train_acc), 'avg_cost': 0.0}

                        dev_outputs = self.pytorch_model(X_dev_tensor)
                        dev_preds = dev_outputs.argmax(dim=1).cpu().numpy()
                        dev_correct = Y_dev[np.arange(len(dev_preds)), dev_preds]
                        dev_acc = dev_correct.mean()
                        dev_costs = C_dev[np.arange(len(dev_preds)), dev_preds]
                        dev_avg_cost = dev_costs.mean()

                    dev_metrics = {'accuracy': float(dev_acc), 'avg_cost': float(dev_avg_cost)}

                    # Add Rank Score for monitoring / selection (dev set)
                    try:
                        from routers.utils.rank_score import rank_score
                        cmin = self._rank_score_cmin
                        cmax = self._rank_score_cmax
                        beta = self._rank_score_beta
                        if cmin is None or cmax is None:
                            cmin = float(np.min(C_dev))
                            cmax = float(np.max(C_dev))
                        dev_metrics['rank_score'] = float(rank_score(float(dev_acc), float(dev_avg_cost), float(cmin), float(cmax), beta=float(beta)))
                        dev_metrics['rank_score_beta'] = float(beta)
                    except Exception:
                        # If rank_score can't be computed, leave it absent.
                        pass

                    should_stop = monitor.log_epoch(epoch, train_metrics, dev_metrics, self)
                    if should_stop:
                        final_epoch = epoch
                        break
                else:
                    if epoch > 10 and epoch % 50 == 0:
                        with torch.no_grad():
                            outputs = self.pytorch_model(X_tensor)
                            if use_soft:
                                current_loss = criterion(outputs, T_tensor).item()
                            else:
                                current_loss = criterion(outputs, y_tensor).item()
                            if current_loss < 1e-6:
                                final_epoch = epoch
                                break
                
                final_epoch = epoch
            
            # NOTE: (translated from Chinese)
            if use_monitoring:
                monitor.end_training(final_epoch)
                monitor.plot_training_curves()
                
                # NOTE: (translated from Chinese)
                if monitor.best_model_path and Path(monitor.best_model_path).exists():
                    if self.verbose > 0:
                        print(f"\n  Loading best model from epoch {monitor.best_epoch}...")
                    try:
                        best_router = LinearRouter.load(monitor.best_model_path)
                        if best_router is not None:
                            if best_router.pytorch_model is not None:
                                self.pytorch_model = best_router.pytorch_model
                                self.scaler = best_router.scaler
                                # （model_mapping）
                                if self.verbose > 0:
                                    print(f"  ✓ Best model loaded (dev {self.monitor_metric}={monitor.best_metric_value:.6f})")
                            else:
                                if self.verbose > 0:
                                    print(f"  ⚠️  Loaded router has no pytorch_model, keeping current model")
                        else:
                            if self.verbose > 0:
                                print(f"  ⚠️  Best router loading returned None")
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"  ⚠️  Could not load best model: {e}")
                            import traceback
                            if self.verbose > 1:
                                traceback.print_exc()
        
        # NOTE: (translated from Chinese)
        if self.verbose > 0:
            self.pytorch_model.eval()
            with torch.no_grad():
                outputs = self.pytorch_model(X_tensor)
                predictions = outputs.argmax(dim=1).cpu().numpy()
                if use_soft:
                    # ，argmax(T)
                    y_true = y_or_T.argmax(axis=1)
                else:
                    y_true = y_or_T
                train_accuracy = (predictions == y_true).mean()
            
            if self.solver == 'lbfgs':
                print(f"    Train setAccuracy: {train_accuracy:.4f}")
            else:
                print(f"    Train setAccuracy: {train_accuracy:.4f} (epochs: {final_epoch})")
    
    def _logits(self, X_scaled: np.ndarray) -> np.ndarray:
        """logits（）"""
        if self.pytorch_model is not None:
            # PyTorch
            self.pytorch_model.eval()
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                outputs = self.pytorch_model(X_tensor)
                logits = outputs.cpu().numpy()
            return logits
        else:
            # sklearndecision_function
            z = self.clf.decision_function(X_scaled)
            
            # （1D）（2D）
            if z.ndim == 1:
                # ：(N,)，(N, 2)
                z = np.column_stack([-z, z])
            
            return z
    
    def predict(
        self,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        """
        
        
        Args:
            X: ，
            X_text: ，
            X_vision: ，
            meta: DataFrame，
        
        Returns:
             (N,)
        """
        Xf = self._fuse(meta, X, X_text, X_vision)
        
        if self.use_gpu and HAS_TORCH:
            # PyTorch scalertensor
            X_scaled = self.scaler.transform(Xf)
            if isinstance(X_scaled, torch.Tensor):
                X_scaled_np = X_scaled.cpu().numpy()
            else:
                X_scaled_np = X_scaled
        else:
            X_scaled = self.scaler.transform(Xf)
            X_scaled_np = X_scaled
        
        # NOTE: (translated from Chinese)
        if self.pytorch_model is not None:
            # PyTorch
            self.pytorch_model.eval()
            X_tensor = torch.FloatTensor(X_scaled_np).to(self.device)
            with torch.no_grad():
                outputs = self.pytorch_model(X_tensor)
                predictions = outputs.argmax(dim=1).cpu().numpy()
            return predictions
        else:
            return self.clf.predict(X_scaled_np)
    
    def predict_proba(
        self,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        """"""
        Xf = self._fuse(meta, X, X_text, X_vision)
        
        if self.use_gpu and HAS_TORCH:
            # PyTorch scaler
            X_scaled = self.scaler.transform(Xf)
            if isinstance(X_scaled, torch.Tensor):
                X_scaled_np = X_scaled.cpu().numpy()
            else:
                X_scaled_np = X_scaled
        else:
            X_scaled = self.scaler.transform(Xf)
            X_scaled_np = X_scaled
        
        if self.pytorch_model is not None:
            # PyTorch
            self.pytorch_model.eval()
            X_tensor = torch.FloatTensor(X_scaled_np).to(self.device)
            with torch.no_grad():
                outputs = self.pytorch_model(X_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            return probs
        else:
            return self.clf.predict_proba(X_scaled_np)
    
    def save(self, path: str):
        """save"""
        save_dict = {
            'clf': self.clf,
            'scaler': self.scaler,
            'penalty': self.penalty,
            'C': self.C,
            'max_iter': self.max_iter,
            'fusion_method': self.fusion_method,
            'text_weight': self.text_weight,
            'multiclass': self.multiclass,
            'solver': self.solver,
            'verbose': self.verbose,
            'device': self.device,
            'use_gpu': self.use_gpu,
            'model_mapping': self.model_mapping,
            'reverse_mapping': self.reverse_mapping,
            'costs': self.costs,
            'text_encoder': self.text_encoder,
            'vision_encoder': self.vision_encoder,
            'use_soft_labels': self.use_soft_labels,
            'train_lambda': self.train_lambda
        }
        
        # PyTorch（）
        if self.pytorch_model is not None:
            if HAS_TORCH:
                save_dict['pytorch_model_state'] = self.pytorch_model.state_dict()
                save_dict['pytorch_input_dim'] = self._last_X_shape[1] if hasattr(self, '_last_X_shape') else None
            else:
                warnings.warn("PyTorchsave，PyTorch")
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @classmethod
    def load(cls, path: str):
        """load"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        router = cls(
            penalty=data.get('penalty', 'l2'),
            C=data.get('C', 1.0),
            max_iter=data.get('max_iter', 200),
            fusion_method=data.get('fusion_method', 'normalize_concat'),
            text_weight=data.get('text_weight', 0.5),
            multiclass=data.get('multiclass', 'multinomial'),
            solver=data.get('solver', None),
            verbose=data.get('verbose', 1),
            device=data.get('device', None),
            text_encoder=data.get('text_encoder', 'BAAI/bge-m3'),
            vision_encoder=data.get('vision_encoder', 'facebook/dinov2-base'),
            use_soft_labels=data.get('use_soft_labels', False),
            train_lambda=data.get('train_lambda', 0.0)
        )
        
        router.clf = data.get('clf')
        router.scaler = data['scaler']
        router.model_mapping = data.get('model_mapping')
        router.reverse_mapping = data.get('reverse_mapping')
        if router.reverse_mapping is None and router.model_mapping:
            router.reverse_mapping = {v: k for k, v in router.model_mapping.items()}
        router.costs = data.get('costs')
        
        # PyTorch（）
        if 'pytorch_model_state' in data and HAS_TORCH:
            input_dim = data.get('pytorch_input_dim')
            if input_dim and router.model_mapping:
                num_classes = len(router.model_mapping)
                try:
                    router.pytorch_model = PyTorchLogisticRegression(
                        input_dim=input_dim,
                        num_classes=num_classes,
                        penalty=router.penalty,
                        C=router.C
                    ).to(router.device)
                    router.pytorch_model.load_state_dict(data['pytorch_model_state'])
                    router.use_gpu = (router.device == 'cuda')
                except Exception as e:
                    print(f"Warning: Could not load PyTorch model: {e}")
                    import traceback
                    traceback.print_exc()
                    router.pytorch_model = None
            else:
                print(f"Warning: Cannot load PyTorch model - input_dim={input_dim}, model_mapping={router.model_mapping is not None}")
                router.pytorch_model = None
        elif 'pytorch_model_state' not in data:
            print(f"Warning: No pytorch_model_state in saved data")
        elif not HAS_TORCH:
            print(f"Warning: PyTorch not available")
        
        return router
    
    def __repr__(self):
        soft_info = f", soft_λ={self.train_lambda}" if self.use_soft_labels else ""
        return (f"LinearRouter(penalty={self.penalty}, C={self.C}, "
                f"fusion={self.fusion_method}{soft_info})")
