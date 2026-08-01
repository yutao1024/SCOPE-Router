#!/usr/bin/env python3
"""
Training Monitor for VLM Routers

Provides training supervision, early stopping, and checkpoint management.
"""

import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import time


class TrainingMonitor:
    """Monitor training progress with early stopping and checkpoint management"""
    
    def __init__(
        self,
        output_dir: Path,
        metric: str = 'rank_score',  # or 'accuracy', 'avg_cost', etc.
        patience: int = 10,
        min_delta: float = 1e-4,
        maximize: bool = True,
        save_checkpoints: bool = True,
        verbose: int = 1
    ):
        """
        Args:
            output_dir: Directory to save checkpoints and logs
            metric: Metric to monitor for early stopping ('rank_score', 'accuracy', etc.)
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum improvement to reset patience
            maximize: True if higher metric is better, False otherwise
            save_checkpoints: Whether to save model checkpoints
            verbose: Verbosity level (0=silent, 1=epoch, 2=detailed)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.maximize = maximize
        self.save_checkpoints = save_checkpoints
        self.verbose = verbose
        
        # Training history
        self.history = {
            'train': [],
            'dev': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_epoch = 0
        self.best_metric_value = float('-inf') if maximize else float('inf')
        self.best_model_path = None
        self.epochs_without_improvement = 0
        
        # Timing
        self.start_time = None
        self.epoch_start_time = None
    
    def is_better(self, current: float, best: float) -> bool:
        """Check if current metric is better than best"""
        if self.maximize:
            return current > best + self.min_delta
        else:
            return current < best - self.min_delta
    
    def start_training(self):
        """Mark training start"""
        self.start_time = time.time()
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Training Monitor Started")
            print(f"  Metric: {self.metric} ({'maximize' if self.maximize else 'minimize'})")
            print(f"  Early stopping: patience={self.patience}, min_delta={self.min_delta}")
            print(f"  Output: {self.output_dir}")
            print(f"{'='*60}\n")
    
    def start_epoch(self, epoch: int):
        """Mark epoch start"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        dev_metrics: Dict[str, float],
        router=None
    ) -> bool:
        """
        Log epoch results and check early stopping
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics dict
            dev_metrics: Dev metrics dict
            router: Router model to save (optional)
        
        Returns:
            should_stop: True if early stopping triggered
        """
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        # Record history
        self.history['train'].append(train_metrics)
        self.history['dev'].append(dev_metrics)
        self.history['epoch_times'].append(epoch_time)
        
        # Check if this is the best model
        current_metric = dev_metrics.get(self.metric, None)
        if current_metric is None:
            print(f"  Warning: Metric '{self.metric}' not found in dev_metrics")
            return False
        
        is_best = self.is_better(current_metric, self.best_metric_value)
        
        if is_best:
            self.best_metric_value = current_metric
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            
            # Save best model
            if self.save_checkpoints and router is not None:
                checkpoint_path = self.output_dir / 'best_model.pkl'
                router.save(str(checkpoint_path))
                self.best_model_path = str(checkpoint_path)
                
                if self.verbose >= 2:
                    print(f"    💾 Saved best model: {checkpoint_path}")
        else:
            self.epochs_without_improvement += 1
        
        # Logging
        if self.verbose >= 1:
            status = "✓ BEST" if is_best else f"  ({self.epochs_without_improvement}/{self.patience})"
            print(f"  Epoch {epoch:3d} [{epoch_time:5.1f}s]: "
                  f"train={train_metrics.get('accuracy', 0):.4f}/{train_metrics.get('avg_cost', 0):.6f} "
                  f"dev={dev_metrics.get('accuracy', 0):.4f}/{dev_metrics.get('avg_cost', 0):.6f} "
                  f"{self.metric}={current_metric:.6f} {status}")
        
        # Check early stopping
        should_stop = self.epochs_without_improvement >= self.patience
        
        if should_stop and self.verbose >= 1:
            print(f"\n  ⏹  Early stopping triggered at epoch {epoch}")
            print(f"      Best epoch: {self.best_epoch}, Best {self.metric}: {self.best_metric_value:.6f}")
        
        # Save periodic checkpoints
        if self.save_checkpoints and router is not None and epoch % 10 == 0:
            checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch}.pkl'
            router.save(str(checkpoint_path))
            if self.verbose >= 2:
                print(f"    💾 Saved checkpoint: {checkpoint_path}")
        
        return should_stop
    
    def end_training(self, final_epoch: int):
        """Mark training end and save summary"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            'total_epochs': final_epoch,
            'best_epoch': self.best_epoch,
            'best_metric': self.metric,
            'best_metric_value': float(self.best_metric_value),
            'total_time_seconds': total_time,
            'best_model_path': self.best_model_path,
            'history': {
                'train': self.history['train'],
                'dev': self.history['dev'],
                'epoch_times': self.history['epoch_times']
            }
        }
        
        # Save summary
        summary_path = self.output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose > 0:
            print(f"\n{'='*60}")
            print(f"Training Complete")
            print(f"  Total time: {total_time:.1f}s ({final_epoch} epochs)")
            print(f"  Best epoch: {self.best_epoch}")
            print(f"  Best {self.metric}: {self.best_metric_value:.6f}")
            if self.best_model_path:
                print(f"  Best model: {self.best_model_path}")
            print(f"  Summary: {summary_path}")
            print(f"{'='*60}\n")
        
        return summary
    
    def plot_training_curves(self, output_file: Optional[str] = None):
        """Plot training curves"""
        if not self.history['train']:
            print("  No training history to plot")
            return
        
        # Extract metrics
        train_acc = [m.get('accuracy', np.nan) for m in self.history['train']]
        dev_acc = [m.get('accuracy', np.nan) for m in self.history['dev']]
        train_cost = [m.get('avg_cost', np.nan) for m in self.history['train']]
        dev_cost = [m.get('avg_cost', np.nan) for m in self.history['dev']]
        
        epochs = np.arange(1, len(train_acc) + 1)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Accuracy plot
        axes[0].plot(epochs, train_acc, 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[0].plot(epochs, dev_acc, 'r-', label='Dev', linewidth=2, alpha=0.7)
        axes[0].axvline(self.best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {self.best_epoch})')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Training & Dev Accuracy', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Cost plot
        axes[1].plot(epochs, train_cost, 'b-', label='Train', linewidth=2, alpha=0.7)
        axes[1].plot(epochs, dev_cost, 'r-', label='Dev', linewidth=2, alpha=0.7)
        axes[1].axvline(self.best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {self.best_epoch})')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Average Cost ($)', fontsize=12)
        axes[1].set_title('Training & Dev Cost', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file is None:
            output_file = self.output_dir / 'training_curves.png'
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose > 0:
            print(f"  Training curves saved: {output_file}")
    
    def get_best_model_info(self) -> Dict[str, Any]:
        """Get information about the best model"""
        return {
            'epoch': self.best_epoch,
            'metric': self.metric,
            'metric_value': self.best_metric_value,
            'model_path': self.best_model_path,
            'train_metrics': self.history['train'][self.best_epoch - 1] if self.best_epoch > 0 and self.best_epoch <= len(self.history['train']) else None,
            'dev_metrics': self.history['dev'][self.best_epoch - 1] if self.best_epoch > 0 and self.best_epoch <= len(self.history['dev']) else None
        }


class CheckpointManager:
    """Manage model checkpoints during training"""
    
    def __init__(self, output_dir: Path, keep_best_n: int = 3):
        """
        Args:
            output_dir: Directory to save checkpoints
            keep_best_n: Number of best checkpoints to keep
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        
        self.checkpoints = []  # List of (metric_value, epoch, path) tuples
    
    def save_checkpoint(
        self,
        router,
        epoch: int,
        metric_value: float,
        is_best: bool = False,
        maximize: bool = True
    ):
        """
        Save a checkpoint
        
        Args:
            router: Router model to save
            epoch: Current epoch
            metric_value: Current metric value
            is_best: Whether this is the best model so far
            maximize: Whether higher metric is better
        """
        # Save checkpoint
        if is_best:
            checkpoint_path = self.output_dir / 'best_model.pkl'
        else:
            checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch}.pkl'
        
        router.save(str(checkpoint_path))
        
        # Track checkpoint
        self.checkpoints.append((metric_value, epoch, str(checkpoint_path)))
        
        # Sort by metric value
        self.checkpoints.sort(key=lambda x: x[0], reverse=maximize)
        
        # Remove old checkpoints (keep best N + the 'best_model.pkl')
        if len(self.checkpoints) > self.keep_best_n:
            for _, _, path in self.checkpoints[self.keep_best_n:]:
                if path != str(self.output_dir / 'best_model.pkl') and Path(path).exists():
                    Path(path).unlink()
            
            self.checkpoints = self.checkpoints[:self.keep_best_n]
    
    def load_best_checkpoint(self, router_class):
        """Load the best checkpoint"""
        best_path = self.output_dir / 'best_model.pkl'
        if best_path.exists():
            return router_class.load(str(best_path))
        else:
            raise FileNotFoundError(f"Best model not found: {best_path}")


if __name__ == '__main__':
    print("Training monitor module loaded.")
    print("Use: from routers.utils.training_monitor import TrainingMonitor")

