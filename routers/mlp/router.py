#!/usr/bin/env python3
"""
MLPR: MLP


"""

import numpy as np
import pickle
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple, Literal
import warnings

from routers.common import RouterBase

try:
    from sklearn.neural_network import MLPClassifier
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

# cuML（）
try:
    import cuml
    from cuml.neural_network import MLPClassifier as CuMLMLPClassifier
    from cuml.preprocessing import StandardScaler as CuMLStandardScaler
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    CuMLMLPClassifier = None
    CuMLStandardScaler = None


if HAS_TORCH:
    class PyTorchMLP(nn.Module):
        """PyTorchMLP"""
        def __init__(self, input_dim, hidden_sizes, num_classes, activation='relu', dropout=0.0):
            super().__init__()
            layers = []
            in_dim = input_dim
            
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(in_dim, hidden_dim))
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'logistic':
                    layers.append(nn.Sigmoid())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
            
            layers.append(nn.Linear(in_dim, num_classes))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
else:
    PyTorchMLP = None


class MLPRouter(RouterBase):
    """
    MLPR: MLP
    
    : p_θ(m|x) = softmax(MLP(φ(x)))
    : 
      - : ，
      - : Soft-CE，
    : argmax_m p_θ(m|x)
    """
    
    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (256,),
        activation: str = 'relu',
        alpha: float = 0.0001,
        max_iter: int = 300,
        fusion_method: str = 'normalize_concat',
        text_weight: float = 0.5,
        solver: str = 'adam',
        learning_rate_init: float = 0.001,
        random_state: int = 42,
        verbose: int = 1,
        device: Optional[str] = None,
        use_pytorch: bool = True,
        use_soft_labels: bool = False,
        train_lambda: float = 0.0,
        text_encoder: str = 'BAAI/bge-m3',
        vision_encoder: str = 'facebook/dinov2-base',
        enable_monitoring: bool = True,
        patience: int = 100,
        monitor_metric: str = 'rank_score'
    ):
        """
        Args:
            hidden_sizes: ，(256,)(256, 128)
            activation:  ('relu', 'tanh', 'logistic')
            alpha: L2
            max_iter: 
            fusion_method: 
            text_weight: （weighted_average）
            solver:  ('adam', 'lbfgs', 'sgd')
            learning_rate_init: 
            random_state: 
            verbose:  (0=, 1=, >1=)
            device:  ('cuda', 'cpu', None=)
            use_pytorch: PyTorch（GPU，True）
            use_soft_labels: （False）
            train_lambda: lambda（0.0）
            text_encoder: （: "BAAI/bge-m3"）
            vision_encoder: （: "facebook/dinov2-base"）
        """
        if not HAS_SKLEARN:
            raise ImportError("Please install: pip install scikit-learn")
        
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.alpha = alpha
        self.max_iter = max_iter
        self.fusion_method = fusion_method
        self.text_weight = text_weight
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.random_state = random_state
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.verbose = verbose
        self.use_pytorch = use_pytorch
        self.use_soft_labels = use_soft_labels
        self.train_lambda = train_lambda
        self.enable_monitoring = enable_monitoring
        self.patience = patience
        self.monitor_metric = monitor_metric
        # Rank score configuration (used for monitoring on dev)
        self._rank_score_cmin = None
        self._rank_score_cmax = None
        self._rank_score_beta = 0.1
        
        # NOTE: (translated from Chinese)
        if device is None:
            if use_pytorch and HAS_TORCH:
                # PyTorch，GPU
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.use_gpu = (self.device == 'cuda')
            elif HAS_CUML:
                # cuML
                try:
                    import cuml
                    if cuml.common.has_gpu():
                        self.device = 'cuda'
                        self.use_gpu = True
                    else:
                        self.device = 'cpu'
                        self.use_gpu = False
                except:
                    self.device = 'cpu'
                    self.use_gpu = False
            else:
                self.device = 'cpu'
                self.use_gpu = False
        else:
            self.device = device
            if device == 'cuda':
                if use_pytorch and HAS_TORCH and torch.cuda.is_available():
                    self.use_gpu = True
                elif HAS_CUML:
                    try:
                        import cuml
                        self.use_gpu = cuml.common.has_gpu()
                    except:
                        self.use_gpu = False
                else:
                    warnings.warn("PyTorch/cuMLGPU，CPU")
                    self.use_gpu = False
                    self.device = 'cpu'
            else:
                self.use_gpu = False
        
        self.clf = None
        self.scaler = None
        self.model_mapping = None
        self.reverse_mapping = None
        self.costs = None  # 
        self.pytorch_model = None  # PyTorch（）
    
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
        
        raise ValueError("Features required：X  (X_text, X_vision)  meta  embedding ")
    
    @staticmethod
    def _derive_labels(Y: np.ndarray, C: np.ndarray, use_soft_labels: bool = False, 
                       soft_label_lambda: float = 0.0, cost_scale: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Y/C（）
        
        Args:
            Y:  (N x K)
            C:  (N x K)
            use_soft_labels: 
            soft_label_lambda: lambda
            cost_scale: （100）
            
        Returns:
            use_soft_labels=False:
                - labels: (N,) ，
                - valid_indices: (N,) 
            use_soft_labels=True:
                - T: (N, K) ，
                - valid_indices: (N,) 
        """
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
            
            T = build_soft_targets(Y, C, lam=soft_label_lambda, scheme='exp', 
                                  fallback='cheapest', cost_scale=cost_scale)
            
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
        MLPR
        
        Args:
            Y:  (N x K)
            C:  (N x K)
            meta: DataFrame
            X: ，
            X_text: ，
            X_vision: ，
            model_mapping: 
            costs:  (K,)，/
        """
        # NOTE: (translated from Chinese)
        Xf = self._fuse(meta, X, X_text, X_vision)
        
        # NOTE: (translated from Chinese)
        y_or_T, valid_indices = self._derive_labels(
            Y, C, 
            use_soft_labels=self.use_soft_labels,
            soft_label_lambda=self.train_lambda,
            cost_scale=100.0  # 100x
        )
        
        if len(valid_indices) < len(Y):
            print(f"  Filtering training samples: {len(Y)} -> {len(valid_indices)} "
                  f"(removed {len(Y) - len(valid_indices)} no correct model)")
            Xf = Xf[valid_indices]
            Y = Y[valid_indices]
            C = C[valid_indices]
            if self.use_soft_labels:
                y_or_T = y_or_T[valid_indices]
        
        # NOTE: (translated from Chinese)
        device_str = f" ({self.device})" if self.use_gpu else ""
        if self.verbose > 0:
            print(f"  {device_str}...")
        
        if self.use_gpu and HAS_CUML:
            self.scaler = CuMLStandardScaler()
            import cupy as cp
            Xf_gpu = cp.asarray(Xf, dtype=cp.float32)
            X_scaled = self.scaler.fit_transform(Xf_gpu)
            X_scaled_np = cp.asnumpy(X_scaled)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(Xf)
            X_scaled_np = X_scaled
        
        # NOTE: (translated from Chinese)
        if model_mapping is None:
            K = Y.shape[1]
            model_mapping = {i: f'model_{i}' for i in range(K)}
        
        self.model_mapping = model_mapping
        self.reverse_mapping = {v: k for k, v in model_mapping.items()}
        
        # NOTE: (translated from Chinese)
        if costs is not None:
            self.costs = np.asarray(costs, dtype=float)
        elif 'costs' in kwargs:
            self.costs = np.asarray(kwargs['costs'], dtype=float)
        else:
            # C
            self.costs = C.mean(axis=0)

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
        
        num_classes = len(model_mapping)
        
        # NOTE: (translated from Chinese)
        self._last_X_shape = X_scaled_np.shape
        
        # dev（）
        # dev（，dev）
        X_dev_scaled_np = None
        Y_dev_filtered = None
        C_dev_filtered = None
        
        if self.enable_monitoring and Y_dev is not None:
            # dev（）
            Xf_dev = self._fuse(meta_dev, X_dev, X_text_dev, X_vision_dev)
            # dev
            if self.use_gpu and HAS_CUML:
                import cupy as cp
                Xf_dev_gpu = cp.asarray(Xf_dev, dtype=cp.float32)
                X_dev_scaled = self.scaler.transform(Xf_dev_gpu)
                X_dev_scaled_np = cp.asnumpy(X_dev_scaled)
            else:
                X_dev_scaled_np = self.scaler.transform(Xf_dev)
            
            # dev
            Y_dev_filtered = Y_dev
            C_dev_filtered = C_dev
            
            if self.verbose > 0:
                print(f"  Dev set: {len(Y_dev)} samples（，）")
        
        # MLP（PyTorchsklearn）
        if self.use_pytorch and HAS_TORCH and self.use_gpu:
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
            # sklearn CPU（）
            if self.use_soft_labels:
                print("  ⚠️  sklearnUnsupportedSoft labeltraining，Hard label")
                y, _ = self._derive_labels(Y, C, use_soft_labels=False)
            else:
                y = y_or_T
            self._train_sklearn(X_scaled_np, y, num_classes, model_mapping)
        
        print(f"  MLPRTraining complete: ={self.device}, Fusion method={self.fusion_method}, "
              f"={self.hidden_sizes}, ={X_scaled_np.shape[1]}, "
              f"Num models={len(model_mapping)}")
    
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
        """PyTorch GPUtrainingMLP（training）
        
        Args:
            X_scaled: (N, D) 
            y_or_T: (N,)   (N, K) 
            num_classes: 
            use_soft: 
            X_dev, Y_dev, C_dev: Dev（）
            monitor_output_dir: 
        """
        # NOTE: (translated from Chinese)
        use_monitoring = (self.enable_monitoring and X_dev is not None and 
                         Y_dev is not None and C_dev is not None)
        
        if use_monitoring:
            from routers.utils.training_monitor import TrainingMonitor
            if monitor_output_dir is None:
                import tempfile
                monitor_output_dir = Path(tempfile.mkdtemp(prefix='mlp_router_'))
            
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
        
        if self.verbose > 0:
            label_type = "Soft label" if use_soft else "Hard label"
            monitor_str = " (training)" if use_monitoring else ""
            print(f"  trainingPyTorch MLP (GPU, max_iter={self.max_iter}, {label_type}{monitor_str})...")
            print(f"    samples: {len(y_or_T)}, : {X_scaled.shape[1]}, : {num_classes}")
            print(f"    : {self.hidden_sizes}, : {self.activation}")
        
        # PyTorch
        if not HAS_TORCH or PyTorchMLP is None:
            raise ImportError("Please installPyTorchGPUtraining")
        self.pytorch_model = PyTorchMLP(
            input_dim=X_scaled.shape[1],
            hidden_sizes=self.hidden_sizes,
            num_classes=num_classes,
            activation=self.activation
        ).to(self.device)
        
        # NOTE: (translated from Chinese)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # NOTE: (translated from Chinese)
        if use_soft:
            # ：KL (Soft Cross-Entropy)
            T_tensor = torch.FloatTensor(y_or_T).to(self.device)
            
            def soft_ce_loss(logits, soft_targets):
                """Soft Cross-Entropy: -Σ t_i log(p_i)"""
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                return -torch.sum(soft_targets * log_probs, dim=-1).mean()
            
            criterion = soft_ce_loss
            
            if self.verbose > 0:
                print(f"    Soft-CE loss (train_lambda={self.train_lambda})")
        else:
            # ：CE
            y_tensor = torch.LongTensor(y_or_T).to(self.device)
            criterion = nn.CrossEntropyLoss()
        
        if self.solver.lower() == 'adam':
            optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate_init, 
                                 weight_decay=self.alpha)
        elif self.solver.lower() == 'sgd':
            optimizer = optim.SGD(self.pytorch_model.parameters(), lr=self.learning_rate_init,
                                weight_decay=self.alpha, momentum=0.9)
        else:
            optimizer = optim.Adam(self.pytorch_model.parameters(), lr=self.learning_rate_init,
                                weight_decay=self.alpha)
        
        # NOTE: (translated from Chinese)
        batch_size = min(1024, len(X_scaled))
        best_loss = float('inf')
        simple_patience = 10  # early stoppingpatience
        no_improve = 0
        
        # dev（）
        if use_monitoring:
            X_dev_tensor = torch.FloatTensor(X_dev).to(self.device)
        
        if self.verbose > 0 and not use_monitoring:
            print("    Training......", end="", flush=True)
        
        for epoch in range(1, self.max_iter + 1):
            self.pytorch_model.train()
            
            # NOTE: (translated from Chinese)
            indices = torch.randperm(len(X_tensor))
            X_shuffled = X_tensor[indices]
            
            if use_soft:
                T_shuffled = T_tensor[indices]
            else:
                y_shuffled = y_tensor[indices]
            
            # NOTE: (translated from Chinese)
            epoch_loss = 0.0
            for i in range(0, len(X_shuffled), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.pytorch_model(batch_X)
                
                if use_soft:
                    batch_T = T_shuffled[i:i+batch_size]
                    loss = criterion(outputs, batch_T)
                else:
                    batch_y = y_shuffled[i:i+batch_size]
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # NOTE: (translated from Chinese)
            if use_monitoring:
                monitor.start_epoch(epoch)
                
                # train
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
                    
                    # dev
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
                        pass
                
                # NOTE: (translated from Chinese)
                should_stop = monitor.log_epoch(epoch, train_metrics, dev_metrics, self)
                if should_stop:
                    break
            else:
                # NOTE: (translated from Chinese)
                avg_loss = epoch_loss / (len(X_shuffled) // batch_size + 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= simple_patience:
                        break
        
        # NOTE: (translated from Chinese)
        if use_monitoring:
            monitor.end_training(epoch)
            monitor.plot_training_curves()
            
            # NOTE: (translated from Chinese)
            if monitor.best_model_path and Path(monitor.best_model_path).exists():
                if self.verbose > 0:
                    print(f"\n  Loading best model from epoch {monitor.best_epoch}...")
                try:
                    best_router = MLPRouter.load(monitor.best_model_path)
                    if best_router is not None:
                        self.pytorch_model = best_router.pytorch_model
                        self.scaler = best_router.scaler
                        # （model_mapping）
                        if self.verbose > 0:
                            print(f"  ✓ Best model loaded (dev {self.monitor_metric}={monitor.best_metric_value:.6f})")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"  ⚠️  Could not load best model: {e}")
        
        # NOTE: (translated from Chinese)
        self.pytorch_model.eval()
        with torch.no_grad():
            outputs = self.pytorch_model(X_tensor)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            
            if use_soft:
                # NOTE: (translated from Chinese)
                y_true = y_or_T.argmax(axis=1)
            else:
                y_true = y_or_T
            
            train_accuracy = (predictions == y_true).mean()
        
        if self.verbose > 0:
            print(" ✓")
            print(f"    Training complete (epochs: {epoch+1}, Train setAccuracy: {train_accuracy:.4f})")
        
        # ，sklearn（predict_proba）
        # pytorch_model
    
    def _train_sklearn(self, X_scaled: np.ndarray, y: np.ndarray, num_classes: int, model_mapping: Dict):
        """sklearn CPUtrainingMLP"""
        # MLPClassifier
        if self.verbose > 0:
            print(f"  trainingMLPClassifier (solver={self.solver}, max_iter={self.max_iter})...")
            print(f"    samples: {len(y)}, : {X_scaled.shape[1]}, : {num_classes}")
            print(f"    : {self.hidden_sizes}, : {self.activation}")
            print("    Training......", end="", flush=True)
        
        self.clf = MLPClassifier(
            hidden_layer_sizes=self.hidden_sizes,
            activation=self.activation,
            alpha=self.alpha,
            max_iter=self.max_iter,
            solver=self.solver,
            learning_rate_init=self.learning_rate_init,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=max(0, self.verbose - 1)
        )
        
        self.clf.fit(X_scaled, y)
        
        # NOTE: (translated from Chinese)
        train_accuracy = self.clf.score(X_scaled, y)
        
        if self.verbose > 0:
            print(" ✓")
            if hasattr(self.clf, 'n_iter_'):
                print(f"    : {self.clf.n_iter_}")
            print(f"    Train setAccuracy: {train_accuracy:.4f}")
    
    def _logits(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        logits（，logits）
        """
        if self.pytorch_model is not None:
            # PyTorch
            self.pytorch_model.eval()
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            with torch.no_grad():
                outputs = self.pytorch_model(X_tensor)
                logits = outputs.cpu().numpy()
            return logits
        else:
            # sklearn
            return self.clf.predict_log_proba(X_scaled)
    
    def predict(
        self,
        X: Optional[np.ndarray] = None,
        X_text: Optional[np.ndarray] = None,
        X_vision: Optional[np.ndarray] = None,
        meta=None
    ) -> np.ndarray:
        """
        
        
        Returns:
             (N,)
        """
        Xf = self._fuse(meta, X, X_text, X_vision)
        
        if self.use_gpu and HAS_CUML:
            import cupy as cp
            Xf_gpu = cp.asarray(Xf, dtype=cp.float32)
            X_scaled = self.scaler.transform(Xf_gpu)
            X_scaled_np = cp.asnumpy(X_scaled)
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
        
        if self.use_gpu and HAS_CUML:
            import cupy as cp
            Xf_gpu = cp.asarray(Xf, dtype=cp.float32)
            X_scaled = self.scaler.transform(Xf_gpu)
            X_scaled_np = cp.asnumpy(X_scaled)
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
            'hidden_sizes': self.hidden_sizes,
            'activation': self.activation,
            'alpha': self.alpha,
            'max_iter': self.max_iter,
            'fusion_method': self.fusion_method,
            'text_weight': self.text_weight,
            'solver': self.solver,
            'learning_rate_init': self.learning_rate_init,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'device': self.device,
            'use_pytorch': self.use_pytorch,
            'use_gpu': self.use_gpu,
            'use_soft_labels': self.use_soft_labels,
            'train_lambda': self.train_lambda,
            'text_encoder': self.text_encoder,
            'vision_encoder': self.vision_encoder,
            'model_mapping': self.model_mapping,
            'reverse_mapping': self.reverse_mapping,
            'costs': self.costs
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
            hidden_sizes=tuple(data.get('hidden_sizes', (256,))),
            activation=data.get('activation', 'relu'),
            alpha=data.get('alpha', 0.0001),
            max_iter=data.get('max_iter', 300),
            fusion_method=data.get('fusion_method', 'normalize_concat'),
            text_weight=data.get('text_weight', 0.5),
            solver=data.get('solver', 'adam'),
            learning_rate_init=data.get('learning_rate_init', 0.001),
            random_state=data.get('random_state', 42),
            verbose=data.get('verbose', 1),
            device=data.get('device', None),
            use_pytorch=data.get('use_pytorch', True),
            use_soft_labels=data.get('use_soft_labels', False),
            train_lambda=data.get('train_lambda', 0.0),
            text_encoder=data.get('text_encoder', 'BAAI/bge-m3'),
            vision_encoder=data.get('vision_encoder', 'facebook/dinov2-base')
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
                    router.pytorch_model = PyTorchMLP(
                        input_dim=input_dim,
                        hidden_sizes=router.hidden_sizes,
                        num_classes=num_classes,
                        activation=router.activation
                    ).to(router.device)
                    router.pytorch_model.load_state_dict(data['pytorch_model_state'])
                    router.use_pytorch = True
                except Exception as e:
                    print(f"Warning: Could not load PyTorch model: {e}")
                    router.pytorch_model = None
        
        return router
    
    def __repr__(self):
        soft_info = f", soft_λ={self.train_lambda}" if self.use_soft_labels else ""
        return (f"MLPRouter(hidden={self.hidden_sizes}, "
                f"fusion={self.fusion_method}{soft_info})")

