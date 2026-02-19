#!/usr/bin/env python3
"""
Phase 2: Cross-Attention Fusion for ECG Classification
FINAL FIX: Properly removes classification heads from encoders
Verified by diagnostic script
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add project paths
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
for path in [project_root, src_dir, src_dir / 'data', src_dir / 'models']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class CrossAttentionFusion(nn.Module):
    """Cross-attention module to fuse 1D and 2D features"""
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        B = query.shape[0]
        
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)
        
        Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
        out = self.out_proj(out).squeeze(1)
        
        return out


class EncoderWithoutHead(nn.Module):
    """
    Wrapper that extracts features from encoder WITHOUT using classification head
    CRITICAL: This ensures we get 384D embeddings, not 4D logits
    """
    
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        
        # Check if encoder has transformer components
        if not (hasattr(encoder, 'patch_embed') and 
                hasattr(encoder, 'blocks') and 
                hasattr(encoder, 'norm')):
            raise ValueError("Encoder must have patch_embed, blocks, and norm attributes")
        
        print(f"  ✓ Wrapped encoder (bypasses classification head)")
    
    def forward(self, x):
        """Extract CLS token features directly from transformer"""
        # Patch embedding
        x = self.encoder.patch_embed(x)
        
        # Transformer blocks
        for block in self.encoder.blocks:
            x = block(x)
        
        # Layer norm
        x = self.encoder.norm(x)
        
        # Return CLS token (first token)
        return x[:, 0]  # Shape: [B, 384]


class ECGFusionModel(nn.Module):
    """
    Complete fusion model with aligned encoders + cross-attention + classifier
    """
    
    def __init__(
        self,
        encoder_1d: nn.Module,
        encoder_2d: nn.Module,
        feature_dim: int = 384,
        num_classes: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_encoders: bool = True
    ):
        super().__init__()
        
        # Wrap encoders to bypass classification heads
        self.encoder_1d = EncoderWithoutHead(encoder_1d)
        self.encoder_2d = EncoderWithoutHead(encoder_2d)
        self.freeze_encoders = freeze_encoders
        
        # Freeze encoders if specified
        if freeze_encoders:
            for param in self.encoder_1d.parameters():
                param.requires_grad = False
            for param in self.encoder_2d.parameters():
                param.requires_grad = False
        
        # Cross-attention: bidirectional
        self.cross_attn_1d_to_2d = CrossAttentionFusion(feature_dim, num_heads, dropout)
        self.cross_attn_2d_to_1d = CrossAttentionFusion(feature_dim, num_heads, dropout)
        
        # Layer norm
        self.norm = nn.LayerNorm(feature_dim)
        
        # Fusion
        fusion_dim = feature_dim * 2
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        print(f"\n[ECG Fusion Model] Created")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Fusion dim: {fusion_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  Num heads: {num_heads}")
        print(f"  Encoders frozen: {freeze_encoders}")
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
        """
        Args:
            x_1d: (B, C, L) - 1D ECG
            x_2d: (B, C, H, W) - 2D ECG representation
        Returns:
            logits: (B, num_classes)
        """
        # Extract features (EncoderWithoutHead ensures these are 384D embeddings)
        with torch.set_grad_enabled(not self.freeze_encoders):
            feat_1d = self.encoder_1d(x_1d)  # [B, 384]
            feat_2d = self.encoder_2d(x_2d)  # [B, 384]
        
        # Verify feature dimensions (should be 384, not 4!)
        assert feat_1d.shape[1] == 384, f"Expected 384D features, got {feat_1d.shape[1]}D"
        assert feat_2d.shape[1] == 384, f"Expected 384D features, got {feat_2d.shape[1]}D"
        
        # Bidirectional cross-attention
        feat_1d_attended = self.cross_attn_1d_to_2d(feat_1d, feat_2d)  # 1D queries 2D
        feat_2d_attended = self.cross_attn_2d_to_1d(feat_2d, feat_1d)  # 2D queries 1D
        
        # Residual connections
        feat_1d_fused = self.norm(feat_1d + feat_1d_attended)
        feat_2d_fused = self.norm(feat_2d + feat_2d_attended)
        
        # Concatenate
        fused = torch.cat([feat_1d_fused, feat_2d_fused], dim=1)  # [B, 768]
        
        # Classify
        logits = self.classifier(fused)  # [B, 4]
        
        return logits


def evaluate_model(
    model: ECGFusionModel,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path
):
    """Comprehensive evaluation on test set"""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix,
        precision_recall_fscore_support
    )
    
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for x_1d, x_2d, labels in tqdm(test_loader, desc="Test Evaluation"):
            x_1d = x_1d.to(device)
            x_2d = x_2d.to(device)
            labels = labels.to(device)
            
            logits = model(x_1d, x_2d)
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Overall metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    sensitivity = recall
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Specificity
    cm = confusion_matrix(all_labels, all_preds)
    specificity_per_class = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(spec)
    specificity = np.average(specificity_per_class, weights=cm.sum(axis=1))
    
    # AUC
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        auc_per_class = []
        for i in range(all_probs.shape[1]):
            binary_labels = (all_labels == i).astype(int)
            if len(np.unique(binary_labels)) > 1:
                auc_class = roc_auc_score(binary_labels, all_probs[:, i])
            else:
                auc_class = 0.0
            auc_per_class.append(auc_class)
    except Exception:
        auc = 0.0
        auc_per_class = [0.0] * all_probs.shape[1]
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    sensitivity_per_class = recall_per_class
    
    # Build results dictionary
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "auc_per_class": [float(x) for x in auc_per_class],
        "precision_per_class": [float(x) for x in precision_per_class],
        "recall_per_class": [float(x) for x in recall_per_class],
        "sensitivity_per_class": [float(x) for x in sensitivity_per_class],
        "specificity_per_class": [float(x) for x in specificity_per_class],
        "f1_per_class": [float(x) for x in f1_per_class],
        "confusion_matrix": cm.tolist()
    }
    
    # Save results
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print results
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Overall Metrics:")
    print(f"  Accuracy:    {accuracy*100:.2f}%")
    print(f"  Precision:   {precision:.4f}")
    print(f"  F1 Score:    {f1:.4f}")
    print(f"  AUC:         {auc:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'test_confusion_matrix.png', dpi=300)
        plt.close()
    except Exception:
        pass
    
    print("="*70)
    print(f"Results saved to: {output_dir / 'test_results.json'}")
    print("="*70)
    
    return results


def train_fusion_model(
    model: ECGFusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    output_dir: Path,
    patience: int = 10
):
    """Train the fusion model"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    epochs_without_improvement = 0
    best_val_acc = 0.0
    
    print("\n" + "="*70)
    print("PHASE 2: FUSION MODEL TRAINING")
    print("="*70)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x_1d, x_2d, labels) in enumerate(pbar):
            x_1d = x_1d.to(device)
            x_2d = x_2d.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(x_1d, x_2d)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_1d, x_2d, labels in tqdm(val_loader, desc="Validation"):
                x_1d = x_1d.to(device)
                x_2d = x_2d.to(device)
                labels = labels.to(device)
                
                logits = model(x_1d, x_2d)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print("="*70)
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Val F1:     {val_f1:.4f}")
        
        ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': history
        }
        torch.save(ckpt, output_dir / f'checkpoint_epoch_{epoch}.pth')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(ckpt, output_dir / 'best_fusion_model.pth')
            print(f"  💾 Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Training Complete! Best Val Acc: {best_val_acc:.2f}%")
    
    # Test evaluation
    if test_loader is not None:
        best_checkpoint = torch.load(output_dir / 'best_fusion_model.pth', map_location=device)
        model.load_state_dict(best_checkpoint['model_state_dict'])
        evaluate_model(model, test_loader, device, output_dir)
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aligned-encoder-1d', required=True)
    parser.add_argument('--aligned-encoder-2d', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--representation-type', default='stft')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--feature-dim', type=int, default=384)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze-encoders', action='store_true')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("PHASE 2: CROSS-ATTENTION FUSION")
    print("="*70)
    
    # Load data
    try:
        from src.data.fusion_dataset import ECGFusionDataModule
    except ImportError:
        from fusion_dataset import ECGFusionDataModule
    
    data_module = ECGFusionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        representation_type=args.representation_type
    )
    data_module.setup()
    
    # Load encoders
    print("\n[Loading Encoders]")
    encoder_1d_state = torch.load(args.aligned_encoder_1d, map_location=device)
    encoder_2d_state = torch.load(args.aligned_encoder_2d, map_location=device)
    
    try:
        from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
        from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
    except ImportError:
        from ecg_transformer_1d import create_ecg_transformer_1d_small
        from models.ecg_transformer_2d import create_ecg_transformer_2d_small
    
    encoder_1d = create_ecg_transformer_1d_small()
    encoder_2d = create_ecg_transformer_2d_small(representation_type='mel_spectrogram')
    
    encoder_1d.load_state_dict(encoder_1d_state, strict=False)
    encoder_2d.load_state_dict(encoder_2d_state, strict=False)
    
    encoder_1d = encoder_1d.to(device)
    encoder_2d = encoder_2d.to(device)
    
    print("  ✓ Encoders loaded")
    
    # Create fusion model (EncoderWithoutHead wrapper handles head removal)
    model = ECGFusionModel(
        encoder_1d=encoder_1d,
        encoder_2d=encoder_2d,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_encoders=args.freeze_encoders
    )
    model = model.to(device)
    
    # Train
    train_fusion_model(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        test_loader=data_module.test_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        output_dir=output_dir,
        patience=args.patience
    )
    
    print("\n✓ Phase 2 Complete!")


if __name__ == "__main__":
    main()
