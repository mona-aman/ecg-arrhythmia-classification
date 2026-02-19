"""
Training Pipeline for ECG Transformers
Handles training, validation, and evaluation for both 1D and 2D models
UPDATED: Now supports 2D representations and per-class AUC
"""

import torch
import torch.nn as nn
import torch.optim as optim
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except Exception:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False
import numpy as np
from scipy.signal import savgol_filter
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

plt = None
sns = None

from ..models import create_model
from ..data.torch_dataset import ECGDataModule


class ECGTrainer:
    """Trainer class for ECG Transformer models"""
    
    def __init__(self,
                 model_type: str,
                 model_size: str = 'small',
                 data_dir: str = 'data/processed',
                 output_dir: str = 'outputs',
                 device: str = 'auto',
                 representation_type: str = 'mel',  # ✅ NEW: For 2D models
                 
                 # Training hyperparameters
                 batch_size: int = 32,
                 num_epochs: int = 35,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_epochs: int = 10,
                 
                 # Training settings
                 use_class_weights: bool = True,
                 augment_train: bool = True,
                 mixed_precision: bool = True,
                 gradient_clip_val: float = 1.0,
                 
                 # Early stopping
                 patience: int = 15,
                 min_delta: float = 0.001,
                 
                 # Logging
                 log_interval: int = 10,
                 save_best_only: bool = True):
        
        self.model_type = model_type
        self.model_size = model_size
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.representation_type = representation_type  # ✅ NEW
        
        # Create experiment directory
        if model_type == '2d':
            exp_name = f"{model_type}_{representation_type}_{model_size}_{int(time.time())}"
        else:
            exp_name = f"{model_type}_{model_size}_{int(time.time())}"
        self.exp_dir = self.output_dir / exp_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Training hyperparameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        
        # Training settings
        self.use_class_weights = use_class_weights
        self.augment_train = augment_train
        self.mixed_precision = mixed_precision
        self.gradient_clip_val = gradient_clip_val
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        
        # Logging
        self.log_interval = log_interval
        self.save_best_only = save_best_only
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.data_module = None
        self.writer = None
        self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
    def setup(self):
        """Setup model, data, and training components"""
        print(f"Setting up {self.model_type.upper()} ECG Transformer ({self.model_size})")
        if self.model_type == '2d':
            print(f"2D Representation: {self.representation_type}")
        print(f"Device: {self.device}")
        print(f"Experiment directory: {self.exp_dir}")
        
        # Setup data with representation type
        # For 2D models the model performs the 2D conversion internally (avoids double-conversion),
        # so request raw 1D signals from the data module. For 1D models, request 1D as well.
        if self.model_type == '2d':
            data_repr_type = '1d'  # dataset should return raw 1D signals; model will convert to 2D
        else:
            data_repr_type = '1d'
        
        self.data_module = ECGDataModule(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            augment_train=self.augment_train,
            representation_type=data_repr_type  # ✅ NEW
        )
        self.data_module.setup()
        
        data_stats = self.data_module.get_stats()
        print(f"Dataset: {data_stats['train_size']} train, {data_stats['val_size']} val, {data_stats['test_size']} test")
        
        # Create model. When building a 2D model, pass the desired representation type
        # (e.g. 'cwt', 'mel') so the model constructs the correct internal converter.
        self.model = create_model(
            model_type=self.model_type,
            size=self.model_size,
            num_classes=data_stats['num_classes'],
            representation_type=self.representation_type if self.model_type == '2d' else None
        )
        self.model.to(self.device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Setup loss function
        if self.use_class_weights:
            class_weights = self.data_module.class_weights.to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler
        total_steps = len(self.data_module.train_loader) * self.num_epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-7
        )
        
        # Setup mixed precision
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Setup logging
        if SummaryWriter is not None:
            try:
                self.writer = SummaryWriter(str(self.exp_dir / 'logs'))
            except Exception as e:
                print(f"Warning: unable to initialize SummaryWriter: {e}")
                self.writer = None
        else:
            print("tensorboard not available; continuing without SummaryWriter")
            self.writer = None
        
        # Save configuration
        self._save_config()
        
    def _save_config(self):
        """Save training configuration"""
        config = {
            'model_type': self.model_type,
            'model_size': self.model_size,
            'representation_type': self.representation_type,  # ✅ NEW
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'use_class_weights': self.use_class_weights,
            'augment_train': self.augment_train,
            'mixed_precision': self.mixed_precision,
            'gradient_clip_val': self.gradient_clip_val,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'device': str(self.device),
            'data_stats': self.data_module.get_stats()
        }
        
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch - WITH NaN PREVENTION"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        processed_batches = 0
        skipped_batches = 0
        
        for batch_idx, (data, target) in enumerate(self.data_module.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Clamp extreme input values
            data = torch.clamp(data, min=-10, max=10)
            
            # Check for NaN/Inf in inputs
            if torch.isnan(data).any() or torch.isinf(data).any():
                print(f"Warning: NaN/Inf in input data at batch {batch_idx}, skipping...")
                skipped_batches += 1
                continue
            
            self.optimizer.zero_grad()
            
            # Forward + loss
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                if not torch.isfinite(loss):
                    print(f"Warning: non-finite loss at batch {batch_idx}: {loss.item()}")
                    skipped_batches += 1
                    continue
                
                if loss.item() > 100:
                    print(f"Warning: Very high loss {loss.item():.2f} at batch {batch_idx}")
                    skipped_batches += 1
                    continue
                
                self.scaler.scale(loss).backward()
                
                try:
                    self.scaler.unscale_(self.optimizer)
                except RuntimeError as e:
                    print(f"Warning: scaler.unscale_() error at batch {batch_idx}: {e}")
                    self.optimizer.zero_grad()
                    try:
                        self.scaler.update()
                    except Exception:
                        pass
                    skipped_batches += 1
                    continue
                
                # Check for NaN gradients
                has_nan_grad = False
                for p in self.model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"Warning: NaN in gradients at batch {batch_idx}, skipping...")
                    self.optimizer.zero_grad()
                    try:
                        self.scaler.update()
                    except Exception:
                        pass
                    skipped_batches += 1
                    continue
                
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                
                if not torch.isfinite(loss):
                    print(f"Warning: non-finite loss at batch {batch_idx}: {loss.item()}")
                    skipped_batches += 1
                    continue
                
                if loss.item() > 100:
                    print(f"Warning: Very high loss {loss.item():.2f} at batch {batch_idx}")
                    skipped_batches += 1
                    continue
                
                loss.backward()
                
                has_nan_grad = False
                for p in self.model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"Warning: NaN in gradients at batch {batch_idx}, skipping...")
                    self.optimizer.zero_grad()
                    skipped_batches += 1
                    continue
                
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Statistics
            processed_batches += 1
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            # Logging
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.data_module.train_loader)}: '
                    f'Loss {loss.item():.4f}, LR {current_lr:.6f}')
                if skipped_batches > 0:
                    print(f'  (Skipped {skipped_batches} batches with NaN/extreme values)')
        
        if processed_batches > 0:
            avg_loss = total_loss / processed_batches
            accuracy = total_correct / total_samples
        else:
            print("ERROR: No batches were successfully processed!")
            avg_loss = float('nan')
            accuracy = 0.0
        
        if skipped_batches > 0:
            print(f"Epoch {self.current_epoch} summary: Processed {processed_batches}, Skipped {skipped_batches}")
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model with comprehensive metrics"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.data_module.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.data_module.val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        cm = confusion_matrix(all_targets, all_preds)
        n_classes = cm.shape[0]
        
        sensitivity_per_class = []
        specificity_per_class = []
        
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sensitivity_per_class.append(sensitivity)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_per_class.append(specificity)
        
        class_counts = np.bincount(all_targets)
        weights = class_counts / len(all_targets)
        avg_sensitivity = np.average(sensitivity_per_class, weights=weights)
        avg_specificity = np.average(specificity_per_class, weights=weights)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        # ✅ NEW: Per-class AUC
        per_class_auc = []
        for i in range(n_classes):
            try:
                class_auc = roc_auc_score(
                    (all_targets == i).astype(int),
                    all_probs[:, i]
                )
                per_class_auc.append(class_auc)
            except Exception:
                per_class_auc.append(0.0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': avg_sensitivity,
            'specificity': avg_specificity,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'auc_per_class': per_class_auc  # ✅ NEW: Return per-class AUC
        }
    
    def train(self):
        """Main training loop"""
        if self.model is None:
            self.setup()
        
        print(f"Starting training for {self.num_epochs} epochs")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            if self.writer is not None:
                try:
                    self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
                    self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
                    self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
                    self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
                    self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
                    self.writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
                    self.writer.add_scalar('Learning_Rate', current_lr, epoch)
                except Exception as e:
                    print(f"Warning: error while writing to TensorBoard: {e}")
            
            is_best = val_metrics['f1'] > self.best_val_score + self.min_delta
            
            if is_best:
                self.best_val_score = val_metrics['f1']
                self.patience_counter = 0
                
                if self.save_best_only:
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
            else:
                self.patience_counter += 1
            
            if not self.save_best_only:
                self._save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 60)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        self._save_training_results(training_time)
        self._plot_training_curves()
        
        return self.training_history
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history,
            'config': {
                'model_type': self.model_type,
                'model_size': self.model_size,
                'representation_type': self.representation_type,
                'num_classes': self.data_module.num_classes
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.exp_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.exp_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  → New best model saved (F1: {metrics['f1']:.4f})")
    
    def _save_training_results(self, training_time: float):
        """Save training results and history"""
        results = {
            'best_val_f1': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'training_time': training_time,
            'final_metrics': {
                'train_loss': self.training_history['train_loss'][-1],
                'train_acc': self.training_history['train_acc'][-1],
                'val_loss': self.training_history['val_loss'][-1],
                'val_acc': self.training_history['val_acc'][-1],
                'val_f1': self.training_history['val_f1'][-1]
            }
        }
        
        with open(self.exp_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(self.exp_dir / 'training_history.csv', index=False)

    def _plot_training_curves(self):
        """Plot and save training curves"""
        try:
            import os
            mpl_config = self.exp_dir / '.mplconfig'
            mpl_config.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault('MPLCONFIGDIR', str(mpl_config))

            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
            except Exception:
                sns = None
        except Exception as e:
            print(f"Warning: matplotlib not available for plotting: {e}")
            return

        def smooth_curve(values, window=5):
            if len(values) < window:
                return values
            series = pd.Series(values)
            return series.rolling(window=window, min_periods=1).mean().values

        epochs = range(1, len(self.training_history['train_loss']) + 1)

        # Plot Loss separately
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, self.training_history['train_loss'], label='Train', alpha=0.3, color='blue')
            ax.plot(epochs, smooth_curve(self.training_history['train_loss']), label='Train (smoothed)', color='blue')
            ax.plot(epochs, self.training_history['val_loss'], label='Val', alpha=0.3, color='orange')
            ax.plot(epochs, smooth_curve(self.training_history['val_loss']), label='Val (smoothed)', color='orange')
            ax.set_title('Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            loss_path = self.exp_dir / 'training_loss.png'
            plt.tight_layout()
            plt.savefig(loss_path, dpi=300, bbox_inches='tight')
            print(f"Saved training loss plot to {loss_path}")
        except Exception as e:
            print(f"Warning: failed to save training loss plot: {e}")
        finally:
            plt.close()

        # Plot Accuracy separately
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, self.training_history['train_acc'], label='Train', alpha=0.3, color='blue')
            ax.plot(epochs, smooth_curve(self.training_history['train_acc']), label='Train (smoothed)', color='blue')
            ax.plot(epochs, self.training_history['val_acc'], label='Val', alpha=0.3, color='orange')
            ax.plot(epochs, smooth_curve(self.training_history['val_acc']), label='Val (smoothed)', color='orange')
            ax.set_title('Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            acc_path = self.exp_dir / 'training_accuracy.png'
            plt.tight_layout()
            plt.savefig(acc_path, dpi=300, bbox_inches='tight')
            print(f"Saved training accuracy plot to {acc_path}")
        except Exception as e:
            print(f"Warning: failed to save training accuracy plot: {e}")
        finally:
            plt.close()

        # Plot Validation F1 separately
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs, self.training_history['val_f1'], label='Val F1', alpha=0.3, color='green')
            ax.plot(epochs, smooth_curve(self.training_history['val_f1']), label='Val F1 (smoothed)', color='green')
            ax.set_title('Validation F1 Score')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            f1_path = self.exp_dir / 'training_val_f1.png'
            plt.tight_layout()
            plt.savefig(f1_path, dpi=300, bbox_inches='tight')
            print(f"Saved validation F1 plot to {f1_path}")
        except Exception as e:
            print(f"Warning: failed to save validation F1 plot: {e}")
        finally:
            plt.close()
    
    def test(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate model on test set"""
        if checkpoint_path is None:
            checkpoint_path = self.exp_dir / 'best_model.pth'
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in self.data_module.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        cm = confusion_matrix(all_targets, all_preds)
        n_classes = cm.shape[0]
        
        sensitivity_per_class = []
        specificity_per_class = []
        
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sensitivity_per_class.append(sensitivity)
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_per_class.append(specificity)
        
        class_counts = np.bincount(all_targets)
        weights = class_counts / len(all_targets)
        avg_sensitivity = np.average(sensitivity_per_class, weights=weights)
        avg_specificity = np.average(specificity_per_class, weights=weights)
        
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        # ✅ NEW: Per-class AUC
        per_class_auc = []
        for i in range(n_classes):
            try:
                class_auc = roc_auc_score(
                    (all_targets == i).astype(int),
                    all_probs[:, i]
                )
                per_class_auc.append(class_auc)
            except Exception:
                per_class_auc.append(0.0)
        
        test_results = {
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': avg_sensitivity,
            'specificity': avg_specificity,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'auc_per_class': per_class_auc,  # ✅ NEW
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'sensitivity_per_class': sensitivity_per_class,
            'specificity_per_class': specificity_per_class,
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'predictions': all_preds.tolist(),
            'targets': all_targets.tolist(),
            'probabilities': all_probs.tolist()
        }
        
        with open(self.exp_dir / 'test_results.json', 'w') as f:
            json.dump({k: v for k, v in test_results.items() 
                    if k not in ['predictions', 'targets', 'probabilities']}, f, indent=2)
        
        np.save(self.exp_dir / 'test_predictions.npy', all_preds)
        np.save(self.exp_dir / 'test_targets.npy', all_targets)
        np.save(self.exp_dir / 'test_probabilities.npy', all_probs)
        
        self._plot_confusion_matrix(cm)
        
        return test_results
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot and save confusion matrix"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            try:
                import seaborn as sns
            except Exception:
                sns = None
        except Exception as e:
            print(f"Warning: matplotlib not available: {e}")
            return

        plt.figure(figsize=(8, 6))

        # Use seaborn if available, otherwise fall back to pure matplotlib
        if sns is not None:
            try:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
            except Exception as e:
                print(f"Warning: seaborn heatmap failed, falling back to matplotlib: {e}")
                sns = None

        if sns is None:
            ax = plt.gca()
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.colorbar(im, ax=ax)

            # Annotate each cell with the integer count
            n_classes = cm.shape[0]
            for i in range(n_classes):
                for j in range(n_classes):
                    val = int(cm[i, j])
                    # choose text color for readability
                    text_color = 'white' if val > (cm.max() / 2.0) else 'black'
                    ax.text(j, i, str(val), ha='center', va='center', color=text_color)

        save_path = self.exp_dir / 'confusion_matrix.png'
        try:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        except Exception as e:
            print(f"Warning: failed to save confusion matrix to {save_path}: {e}")
        finally:
            plt.close()