"""
Training Pipeline for ECG Fusion Model
Handles training, validation, and evaluation for bidirectional cross-attention fusion
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from pathlib import Path
import time
import json
from typing import Dict, Any, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.ecg_fusion_model import create_fusion_model
from src.data.fusion_dataset import ECGFusionDataModule


class ECGFusionTrainer:
    """Trainer class for ECG Fusion Model"""
    
    def __init__(self,
                 # Model configuration
                 d_model: int = 384,
                 d_model_2d: int = None,
                 num_classes: int = 4,
                 num_heads: int = 6,
                 dim_head: int = 64,
                 cross_attn_depth: int = 1,
                 fusion_type: str = 'gated',
                 cosine_sim_attn: bool = False,
                 freeze_encoders: bool = True,
                 
                 # Pre-trained model paths
                 model_1d_path: str = None,
                 model_2d_path: str = None,
                 
                 # Data configuration
                 data_dir: str = 'data/processed',
                 representation_type: str = 'mel',
                 
                 # Output
                 output_dir: str = 'outputs/fusion',
                 exp_name: str = None,
                 device: str = 'auto',
                 
                 # Training hyperparameters
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 warmup_epochs: int = 5,
                
                 # Differential learning rates for fine-tuning
                 lr_fusion_head: float = 1e-5,
                 lr_encoder_unfrozen: float = 1e-6,
                 unfreeze_last_n_blocks: int = 1,
                 
                 # Training settings
                 use_class_weights: bool = True,
                 mixed_precision: bool = True,
                 gradient_clip_val: float = 1.0,
                 
                 # Early stopping
                 patience: int = 15,
                 min_delta: float = 0.001,
                 
                 # Logging
                 log_interval: int = 10,
                 save_best_only: bool = True):
        
        # Model configuration
        self.d_model = d_model
        self.d_model_2d = d_model_2d if d_model_2d is not None else d_model
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.cross_attn_depth = cross_attn_depth
        self.fusion_type = fusion_type
        self.cosine_sim_attn = cosine_sim_attn
        self.freeze_encoders = freeze_encoders
        
        # Pre-trained paths
        self.model_1d_path = model_1d_path
        self.model_2d_path = model_2d_path
        
        # Data configuration
        self.data_dir = data_dir
        self.representation_type = representation_type
        self.output_dir = Path(output_dir)
        
        # Create experiment directory
        if exp_name is None:
            exp_name = f"fusion_{fusion_type}_{int(time.time())}"
        self.exp_name = exp_name
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
        
        # Differential learning rates
        self.lr_fusion_head = lr_fusion_head
        self.lr_encoder_unfrozen = lr_encoder_unfrozen
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        
        # Training settings
        self.use_class_weights = use_class_weights
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
        print(f"Setting up ECG Fusion Model ({self.fusion_type} fusion)")
        print(f"Device: {self.device}")
        print(f"Experiment directory: {self.exp_dir}")
        
        # Setup data
        self.data_module = ECGFusionDataModule(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            representation_type=self.representation_type
        )
        self.data_module.setup()
        
        data_stats = self.data_module.get_stats()
        print(f"Dataset: {data_stats['train_size']} train, "
              f"{data_stats['val_size']} val, {data_stats['test_size']} test")
        
        # Create fusion model
        self.model = create_fusion_model(
            d_model=self.d_model,
            d_model_2d=self.d_model_2d,
            num_classes=self.num_classes,
            fusion_type=self.fusion_type,
            cross_attn_depth=self.cross_attn_depth,
            freeze_encoders=self.freeze_encoders,
            cosine_sim_attn=self.cosine_sim_attn
        )
        
        # Load pre-trained encoders
        if self.model_1d_path is None or self.model_2d_path is None:
            raise ValueError("Must provide paths to pre-trained 1D and 2D models!")
        
        self.model.load_pretrained_encoders(
            path_1d=self.model_1d_path,
            path_2d=self.model_2d_path
        )
        
        self.model.to(self.device)
        
        # Print model info
        param_stats = self.model.count_parameters()
        print(f"\nModel parameters:")
        print(f"  Total: {param_stats['total']:,}")
        print(f"  Trainable: {param_stats['trainable']:,}")
        print(f"  Cross-attention: {param_stats['cross_attention']:,}")
        print(f"  Fusion: {param_stats['fusion']:,}")
        print(f"  Classifier: {param_stats['classifier']:,}")
        
        # Setup loss function
        if self.use_class_weights:
            # Calculate class weights from training data
            train_labels = []
            for _, _, labels in self.data_module.train_loader:
                train_labels.extend(labels.numpy())
            
            class_counts = np.bincount(train_labels, minlength=self.num_classes)
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * self.num_classes
            class_weights = torch.FloatTensor(class_weights).to(self.device)
            
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
            print(f"\nUsing class weights: {class_weights.cpu().numpy()}")
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Setup optimizer with differential learning rates (always run)
        # Implements: freeze-all -> selectively unfreeze last N blocks + final norm -> param groups
        param_groups = self._setup_differential_learning_rates()

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay
        )
        
        # Setup scheduler with warmup
        total_steps = len(self.data_module.train_loader) * self.num_epochs
        warmup_steps = len(self.data_module.train_loader) * self.warmup_epochs
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        # Setup mixed precision
        if self.mixed_precision and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            print("Using mixed precision training")
        
        # Setup logging
        self.writer = SummaryWriter(self.exp_dir / 'logs')
        
        # Save configuration
        self._save_config()

    def _setup_differential_learning_rates(self):
        """
        Setup differential learning rates with selective unfreezing.

        Strategy:
        1. Freeze all encoder parameters by default
        2. Selectively unfreeze last N transformer blocks + final norm
        3. Use very small LR for unfrozen encoder layers (self.lr_encoder_unfrozen)
        4. Use small LR for fusion head (self.lr_fusion_head)
        5. Keep new components (cross-attention, fusion, classifier) trainable

        Returns:
            List of parameter groups with differential learning rates
        """
        print("\n=== Setting up Differential Learning Rates ===")

        # Ensure encoders are loaded
        if self.model.encoder_1d is None or self.model.encoder_2d is None:
            raise RuntimeError("Encoders must be loaded before setting up differential LRs")

        # Get the modules of each encoder Sequential
        encoder_1d_modules = list(self.model.encoder_1d.children())
        encoder_2d_modules = list(self.model.encoder_2d.children())

        # 1) Freeze all encoder params
        print("\n1. Freezing all encoder parameters...")
        for p in self.model.encoder_1d.parameters():
            p.requires_grad = False
        for p in self.model.encoder_2d.parameters():
            p.requires_grad = False
        print("   ✓ All encoder parameters frozen")

        # 2) Unfreeze last N transformer blocks + final norm, if not fully frozen
        if self.unfreeze_last_n_blocks > 0 and not self.freeze_encoders:
            print(f"\n2. Unfreezing last {self.unfreeze_last_n_blocks} transformer block(s)...")

            num_blocks_1d = max(0, len(encoder_1d_modules) - 2)  # minus patch_embed and norm
            num_blocks_2d = max(0, len(encoder_2d_modules) - 2)

            blocks_to_unfreeze_1d = list(range(max(0, num_blocks_1d - self.unfreeze_last_n_blocks), num_blocks_1d))
            blocks_to_unfreeze_2d = list(range(max(0, num_blocks_2d - self.unfreeze_last_n_blocks), num_blocks_2d))

            print(f"   1D encoder: {num_blocks_1d} blocks, unfreezing indices {blocks_to_unfreeze_1d}")
            print(f"   2D encoder: {num_blocks_2d} blocks, unfreezing indices {blocks_to_unfreeze_2d}")

            # Unfreeze selected 1D blocks
            for block_idx in blocks_to_unfreeze_1d:
                block_module = encoder_1d_modules[block_idx + 1]  # +1 for patch_embed
                for p in block_module.parameters():
                    p.requires_grad = True

            # Unfreeze selected 2D blocks
            for block_idx in blocks_to_unfreeze_2d:
                block_module = encoder_2d_modules[block_idx + 1]
                for p in block_module.parameters():
                    p.requires_grad = True

            # Unfreeze final norms
            final_norm_1d = encoder_1d_modules[-1]
            final_norm_2d = encoder_2d_modules[-1]
            for p in final_norm_1d.parameters():
                p.requires_grad = True
            for p in final_norm_2d.parameters():
                p.requires_grad = True

            print("   ✓ Unfroze selected transformer blocks and final norms")
        else:
            print("\n2. Keeping all encoder parameters frozen (freeze_encoders=True or unfreeze_last_n_blocks=0)")

        # 3) Build optimizer parameter groups
        print("\n3. Setting up optimizer parameter groups...")

        # Unfrozen encoder params (if any)
        unfrozen_encoder_params = [
            p for p in self.model.encoder_1d.parameters() if p.requires_grad
        ] + [
            p for p in self.model.encoder_2d.parameters() if p.requires_grad
        ]

        # Fusion head (cross-attn + fusion + classifier)
        fusion_head_params = (
            list(self.model.bi_cross_attention.parameters()) +
            list(self.model.fusion.parameters()) +
            list(self.model.classifier.parameters())
        )

        param_groups = []

        if len(unfrozen_encoder_params) > 0:
            param_groups.append({
                'params': unfrozen_encoder_params,
                'lr': self.lr_encoder_unfrozen,
                'name': 'unfrozen_encoders'
            })
            print(f"   • Unfrozen encoder layers: {len(unfrozen_encoder_params)} params, LR={self.lr_encoder_unfrozen}")

        param_groups.append({
            'params': fusion_head_params,
            'lr': self.lr_fusion_head,
            'name': 'fusion_head'
        })
        print(f"   • Fusion head (cross-attn + fusion + classifier): {len(fusion_head_params)} params, LR={self.lr_fusion_head}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\n=== Summary ===")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable:,} ({100*trainable/total_params:.2f}%)")
        print(f"Frozen parameters: {total_params - trainable:,}")

        return param_groups
    
    def _save_config(self):
        """Save training configuration"""
        config = {
            'model': {
                'd_model': self.d_model,
                'd_model_2d': self.d_model_2d,
                'num_classes': self.num_classes,
                'num_heads': self.num_heads,
                'dim_head': self.dim_head,
                'cross_attn_depth': self.cross_attn_depth,
                'fusion_type': self.fusion_type,
                'cosine_sim_attn': self.cosine_sim_attn,
                'freeze_encoders': self.freeze_encoders
            },
            'data': {
                'data_dir': str(self.data_dir),
                'representation_type': self.representation_type,
                'batch_size': self.batch_size
            },
            'training': {
                'num_epochs': self.num_epochs,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'warmup_epochs': self.warmup_epochs,
                    'lr_fusion_head': self.lr_fusion_head,
                    'lr_encoder_unfrozen': self.lr_encoder_unfrozen,
                    'unfreeze_last_n_blocks': self.unfreeze_last_n_blocks,
                'use_class_weights': self.use_class_weights,
                'mixed_precision': self.mixed_precision,
                'gradient_clip_val': self.gradient_clip_val
            },
            'early_stopping': {
                'patience': self.patience,
                'min_delta': self.min_delta
            },
            'pretrained_models': {
                'model_1d_path': str(self.model_1d_path),
                'model_2d_path': str(self.model_2d_path)
            },
            'device': str(self.device),
            'data_stats': self.data_module.get_stats()
        }
        
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Keep encoders in eval mode if frozen
        if self.freeze_encoders:
            self.model.encoder_1d.eval()
            self.model.encoder_2d.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data_1d, data_2d, target) in enumerate(self.data_module.train_loader):
            data_1d = data_1d.to(self.device)
            data_2d = data_2d.to(self.device)
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.model(data_1d, data_2d)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                
                if self.gradient_clip_val > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data_1d, data_2d)
                loss = self.criterion(output, target)
                loss.backward()
                
                if self.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
                
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            # Logging
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.data_module.train_loader)}: '
                      f'Loss {loss.item():.4f}, LR {current_lr:.6f}')
        
        avg_loss = total_loss / len(self.data_module.train_loader)
        accuracy = total_correct / total_samples
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data_1d, data_2d, target in self.data_module.val_loader:
                data_1d = data_1d.to(self.device)
                data_2d = data_2d.to(self.device)
                target = target.to(self.device)
                
                if self.mixed_precision and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        output = self.model(data_1d, data_2d)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data_1d, data_2d)
                    loss = self.criterion(output, target)
                
                total_loss += loss.item()
                
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Convert to numpy
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        avg_loss = total_loss / len(self.data_module.val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        # Calculate confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Calculate sensitivity and specificity
        sensitivity_per_class = []
        specificity_per_class = []
        
        for i in range(self.num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            sensitivity_per_class.append(sensitivity)
            specificity_per_class.append(specificity)
        
        # Weighted averages
        class_counts = np.bincount(all_targets)
        weights = class_counts / len(all_targets)
        avg_sensitivity = np.average(sensitivity_per_class, weights=weights)
        avg_specificity = np.average(specificity_per_class, weights=weights)
        
        # AUC
        try:
            auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'sensitivity': avg_sensitivity,
            'specificity': avg_specificity,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm
        }
    
    def train(self):
        """Main training loop"""
        if self.model is None:
            self.setup()
        
        print(f"\nStarting training for {self.num_epochs} epochs")
        print("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            print(f"  Per-class F1: {[f'{f:.3f}' for f in val_metrics['f1_per_class']]}")
            print(f"  LR: {current_lr:.6f}")
            
            # TensorBoard logging
            self.writer.add_scalar('Train/Loss', train_metrics['loss'], epoch)
            self.writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/Loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('Val/Accuracy', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
            self.writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Save checkpoint
            is_best = val_metrics['f1'] > self.best_val_score + self.min_delta
            
            if is_best:
                self.best_val_score = val_metrics['f1']
                self.patience_counter = 0
                
                if self.save_best_only:
                    self._save_checkpoint(epoch, val_metrics, is_best=True)
                
                print(f"  → New best model! F1: {val_metrics['f1']:.4f}")
            else:
                self.patience_counter += 1
            
            if not self.save_best_only:
                self._save_checkpoint(epoch, val_metrics, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            print("-" * 60)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Save final results
        self._save_training_results(training_time)
        
        # Plot training curves
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
                'd_model': self.d_model,
                'num_classes': self.num_classes,
                'fusion_type': self.fusion_type
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.exp_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.exp_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def _save_training_results(self, training_time: float):
        """Save training results"""
        results = {
            'best_val_f1': self.best_val_score,
            'total_epochs': self.current_epoch + 1,
            'training_time': training_time,
            'fusion_type': self.fusion_type,
            'freeze_encoders': self.freeze_encoders,
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
        
        # Save training history
        history_df = pd.DataFrame(self.training_history)
        history_df.to_csv(self.exp_dir / 'training_history.csv', index=False)
    
    def _plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.training_history['train_loss'], label='Train', alpha=0.6)
        axes[0, 0].plot(epochs, self.training_history['val_loss'], label='Val', alpha=0.6)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(epochs, self.training_history['train_acc'], label='Train', alpha=0.6)
        axes[0, 1].plot(epochs, self.training_history['val_acc'], label='Val', alpha=0.6)
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1, 0].plot(epochs, self.training_history['val_f1'], label='Val F1', alpha=0.6)
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        fig.delaxes(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(self.exp_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def test(self, checkpoint_path: Optional[str] = None):
        """Test the model"""
        if checkpoint_path is None:
            checkpoint_path = self.exp_dir / 'best_model.pth'
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data_1d, data_2d, target in self.data_module.test_loader:
                data_1d = data_1d.to(self.device)
                data_2d = data_2d.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data_1d, data_2d)
                probs = torch.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate comprehensive metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        cm = confusion_matrix(all_targets, all_preds)
        
        test_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist()
        }
        
        # Save results
        with open(self.exp_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
        
        print("\nTest Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Per-class F1: {[f'{f:.3f}' for f in f1_per_class]}")
        
        return test_results
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(self.exp_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    print("Testing Fusion Trainer...")
    print("Note: Requires pre-trained 1D and 2D models")