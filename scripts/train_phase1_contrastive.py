#!/usr/bin/env python3
"""
Phase 1: Contrastive Alignment (FIXED)
Saves encoders WITHOUT classification heads for Phase 2
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

project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
for path in [project_root, src_dir, src_dir / 'data', src_dir / 'models']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class FeatureExtractor(nn.Module):
    """Extracts features WITHOUT classification head"""
    
    def __init__(self, transformer):
        super().__init__()
        self.patch_embed = transformer.patch_embed
        self.blocks = transformer.blocks
        self.norm = transformer.norm
    
    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token [B, 384]


class ContrastiveAlignmentModel(nn.Module):
    """Contrastive learning for 1D-2D alignment"""
    
    def __init__(
        self,
        encoder_1d: nn.Module,
        encoder_2d: nn.Module,
        dim: int = 384,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.encoder_1d = encoder_1d
        self.encoder_2d = encoder_2d
        
        # Projection heads
        self.proj_1d = nn.Sequential(
            nn.Linear(dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.proj_2d = nn.Sequential(
            nn.Linear(dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        self.temperature = temperature
        
        print(f"\n[Contrastive Model] Created")
        print(f"  Feature dim: {dim}")
        print(f"  Projection dim: {projection_dim}")
        print(f"  Temperature: {temperature}")
    
    def forward(self, x_1d, x_2d):
        feat_1d = self.encoder_1d(x_1d)  # [B, 384]
        feat_2d = self.encoder_2d(x_2d)  # [B, 384]
        
        z_1d = F.normalize(self.proj_1d(feat_1d), dim=1)
        z_2d = F.normalize(self.proj_2d(feat_2d), dim=1)
        
        return z_1d, z_2d
    
    def contrastive_loss(self, z_1d, z_2d):
        batch_size = z_1d.shape[0]
        logits = torch.matmul(z_1d, z_2d.T) / self.temperature
        labels = torch.arange(batch_size).to(z_1d.device)
        
        loss_1d = F.cross_entropy(logits, labels)
        loss_2d = F.cross_entropy(logits.T, labels)
        
        return (loss_1d + loss_2d) / 2


def train(model, train_loader, val_loader, num_epochs, lr, device, output_dir, patience=10):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Only train projection heads, fine-tune encoders slowly
    encoder_params = list(model.encoder_1d.parameters()) + list(model.encoder_2d.parameters())
    proj_params = list(model.proj_1d.parameters()) + list(model.proj_2d.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': lr * 0.1, 'weight_decay': 0.01},
        {'params': proj_params, 'lr': lr, 'weight_decay': 0.01}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {'train_loss': [], 'val_loss': [], 'alignment_gap': []}
    best_gap = 0.0
    patience_count = 0
    
    print("\n" + "="*70)
    print("PHASE 1: CONTRASTIVE ALIGNMENT")
    print("="*70)
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for x_1d, x_2d, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_1d, x_2d = x_1d.to(device), x_2d.to(device)
            
            optimizer.zero_grad()
            z_1d, z_2d = model(x_1d, x_2d)
            loss = model.contrastive_loss(z_1d, z_2d)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0.0
        all_pos_sim = []
        all_neg_sim = []
        
        with torch.no_grad():
            for x_1d, x_2d, _ in val_loader:
                x_1d, x_2d = x_1d.to(device), x_2d.to(device)
                
                z_1d, z_2d = model(x_1d, x_2d)
                loss = model.contrastive_loss(z_1d, z_2d)
                val_loss += loss.item()
                
                # Alignment metrics
                sim = torch.matmul(z_1d, z_2d.T)
                pos_sim = torch.diagonal(sim).mean().item()
                neg_sim = sim[~torch.eye(sim.shape[0], dtype=bool, device=device)].mean().item()
                
                all_pos_sim.append(pos_sim)
                all_neg_sim.append(neg_sim)
        
        val_loss /= len(val_loader)
        avg_pos_sim = np.mean(all_pos_sim)
        avg_neg_sim = np.mean(all_neg_sim)
        alignment_gap = avg_pos_sim - avg_neg_sim
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['alignment_gap'].append(alignment_gap)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Pos Sim: {avg_pos_sim:.4f} | Neg Sim: {avg_neg_sim:.4f}")
        print(f"  Alignment Gap: {alignment_gap:.4f} (target: >0.5)")
        
        # Save best
        if alignment_gap > best_gap:
            best_gap = alignment_gap
            patience_count = 0
            
            # Save FULL encoder modules (without heads)
            torch.save(model.encoder_1d, output_dir / 'aligned_encoder_1d_full.pth')
            torch.save(model.encoder_2d, output_dir / 'aligned_encoder_2d_full.pth')
            
            # Also save state dicts for compatibility
            torch.save(model.encoder_1d.state_dict(), output_dir / 'aligned_encoder_1d.pth')
            torch.save(model.encoder_2d.state_dict(), output_dir / 'aligned_encoder_2d.pth')
            
            print(f"  💾 Saved best model (Gap: {alignment_gap:.4f})")
        else:
            patience_count += 1
        
        if patience_count >= patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    with open(output_dir / 'alignment_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Best Alignment Gap: {best_gap:.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-1d-path', required=True)
    parser.add_argument('--model-2d-path', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--representation-type', default='stft')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--projection-dim', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--output-dir', default='outputs/contrastive_alignment_stft')
    parser.add_argument('--device', default='cuda')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    
    print("="*70)
    print("PHASE 1: CONTRASTIVE ALIGNMENT (FIXED)")
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
    
    # Load pre-trained models
    print("\n[Loading Models]")
    ckpt_1d = torch.load(args.model_1d_path, map_location=device, weights_only=False)
    ckpt_2d = torch.load(args.model_2d_path, map_location=device, weights_only=False)
    
    try:
        from ecg_transformer_1d import create_ecg_transformer_1d_small
        from models.ecg_transformer_2d import create_ecg_transformer_2d_small
    except ImportError:
        from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
        from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
    
    model_1d = create_ecg_transformer_1d_small()
    model_2d = create_ecg_transformer_2d_small(representation_type=args.representation_type)
    
    def _safe_load_state_dict(model, state_dict, name="model"):
        """Try strict load; on failure attempt non-strict and report missing/unexpected keys."""
        try:
            model.load_state_dict(state_dict)
            print(f"  ✓ Loaded {name} (strict=True)")
        except RuntimeError as e:
            print(f"  ⚠️  Strict load failed for {name}: {e}")
            try:
                res = model.load_state_dict(state_dict, strict=False)
                missing = getattr(res, 'missing_keys', None)
                unexpected = getattr(res, 'unexpected_keys', None)
                print(f"  ✓ Loaded {name} with strict=False. Missing keys: {len(missing) if missing is not None else 'unknown'}, Unexpected keys: {len(unexpected) if unexpected is not None else 'unknown'}")
                if missing:
                    print(f"    Missing keys (sample): {missing[:10]}")
                if unexpected:
                    print(f"    Unexpected keys (sample): {unexpected[:10]}")
            except Exception as e2:
                print(f"  ❌ Failed to load {name} even with strict=False: {e2}")
                raise

    _safe_load_state_dict(model_1d, ckpt_1d['model_state_dict'], name='1D model')
    _safe_load_state_dict(model_2d, ckpt_2d['model_state_dict'], name='2D model')
    
    # Extract features WITHOUT heads
    encoder_1d = FeatureExtractor(model_1d).to(device)
    encoder_2d = FeatureExtractor(model_2d).to(device)
    
    print("  ✓ Encoders created (WITHOUT heads)")
    
    # Verify dimensions
    with torch.no_grad():
        x_1d, x_2d, _ = next(iter(data_module.train_loader))
        feat_1d = encoder_1d(x_1d[:1].to(device))
        feat_2d = encoder_2d(x_2d[:1].to(device))
        print(f"  1D features: {feat_1d.shape}")
        print(f"  2D features: {feat_2d.shape}")
        
        assert feat_1d.shape[1] == 384, f"Expected 384D, got {feat_1d.shape[1]}D"
        assert feat_2d.shape[1] == 384, f"Expected 384D, got {feat_2d.shape[1]}D"
        print("  ✓ Feature dimensions verified: 384D")
    
    # Create contrastive model
    model = ContrastiveAlignmentModel(
        encoder_1d=encoder_1d,
        encoder_2d=encoder_2d,
        dim=384,
        projection_dim=args.projection_dim,
        temperature=args.temperature
    ).to(device)
    
    # Train
    train(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        device=device,
        output_dir=output_dir,
        patience=args.patience
    )
    
    print("\n✓ Phase 1 Complete!")
    print(f"Encoders saved to: {output_dir}")
    print("\nNext: Run Phase 2 fusion training")


if __name__ == "__main__":
    main()
# Phase 1: Contrastive Alignment of 1D and 2D ECG Features
# Aligns features from different modalities using InfoNCE contrastive loss

# Based on: "Cross-modal attention fusion for depression detection"
# Adapted for ECG temporal (1D) and spectral (2D) alignment
# UPDATED: Now supports STFT (default), Mel, and CWT representations
# """

# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import numpy as np
# from pathlib import Path
# from tqdm import tqdm
# import json
# import sys

# # Ensure imports work regardless of current working directory
# project_root = Path(__file__).resolve().parent.parent
# src_dir = project_root / 'src'
# if str(project_root) not in sys.path:
#     sys.path.insert(0, str(project_root))
# if str(src_dir) not in sys.path:
#     sys.path.insert(0, str(src_dir))

# data_dir = src_dir / 'data'
# models_dir = src_dir / 'models'
# if str(data_dir) not in sys.path:
#     sys.path.insert(0, str(data_dir))
# if str(models_dir) not in sys.path:
#     sys.path.insert(0, str(models_dir))


# class ContrastiveAlignmentModel(nn.Module):
#     """
#     Contrastive learning model to align 1D and 2D ECG features
    
#     Uses InfoNCE loss: positive pairs are (1D, 2D) from same sample
#     This ensures both modalities learn aligned representations
#     """
    
#     def __init__(
#         self,
#         encoder_1d: nn.Module,
#         encoder_2d: nn.Module,
#         dim_1d: int = 384,
#         dim_2d: int = 512,
#         projection_dim: int = 128,
#         temperature: float = 0.07
#     ):
#         """
#         Args:
#             encoder_1d: Pre-trained 1D encoder
#             encoder_2d: Pre-trained 2D encoder
#             dim_1d: Dimension of 1D features
#             dim_2d: Dimension of 2D features
#             projection_dim: Dimension of shared projection space
#             temperature: Temperature for InfoNCE loss
#         """
#         super().__init__()
        
#         self.encoder_1d = encoder_1d
#         self.encoder_2d = encoder_2d
        
#         # Projection heads to map to shared space
#         self.proj_1d = nn.Sequential(
#             nn.LayerNorm(dim_1d),
#             nn.Linear(dim_1d, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, projection_dim)
#         )
        
#         self.proj_2d = nn.Sequential(
#             nn.LayerNorm(dim_2d),
#             nn.Linear(dim_2d, 256),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(256, projection_dim)
#         )
        
#         self.temperature = temperature
        
#         print(f"\n[ContrastiveAlignment] Created")
#         print(f"  1D dim: {dim_1d} → Projection: {projection_dim}")
#         print(f"  2D dim: {dim_2d} → Projection: {projection_dim}")
#         print(f"  Temperature: {temperature}")
    
#     def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
#         """
#         Forward pass
        
#         Args:
#             x_1d: (B, C, L) - 1D ECG signal
#             x_2d: (B, C, H, W) - 2D ECG representation
        
#         Returns:
#             z_1d: (B, projection_dim) - Projected 1D features
#             z_2d: (B, projection_dim) - Projected 2D features
#         """
#         # Extract features from encoders
#         feat_1d = self.encoder_1d(x_1d)
#         feat_2d = self.encoder_2d(x_2d)
        
#         # Pool if sequence output
#         if feat_1d.dim() == 3:
#             feat_1d = feat_1d.mean(dim=1)
#         if feat_2d.dim() == 3:
#             feat_2d = feat_2d.mean(dim=1)
        
#         # Project to shared space
#         z_1d = self.proj_1d(feat_1d)
#         z_2d = self.proj_2d(feat_2d)
        
#         # L2 normalize (crucial for contrastive learning)
#         z_1d = F.normalize(z_1d, dim=1)
#         z_2d = F.normalize(z_2d, dim=1)
        
#         return z_1d, z_2d
    
#     def contrastive_loss(self, z_1d: torch.Tensor, z_2d: torch.Tensor):
#         """
#         InfoNCE contrastive loss
        
#         Positive pairs: (z_1d[i], z_2d[i]) - same sample
#         Negative pairs: all other combinations
        
#         Args:
#             z_1d: (B, D) - Normalized 1D projections
#             z_2d: (B, D) - Normalized 2D projections
        
#         Returns:
#             loss: Scalar contrastive loss
#         """
#         batch_size = z_1d.shape[0]
        
#         # Compute similarity matrix (cosine similarity)
#         logits = torch.matmul(z_1d, z_2d.T) / self.temperature
        
#         # Labels: diagonal elements are positive pairs
#         labels = torch.arange(batch_size).to(z_1d.device)
        
#         # Symmetric loss (1D→2D and 2D→1D)
#         loss_1d_to_2d = F.cross_entropy(logits, labels)
#         loss_2d_to_1d = F.cross_entropy(logits.T, labels)
        
#         loss = (loss_1d_to_2d + loss_2d_to_1d) / 2
        
#         return loss
    
#     def compute_alignment_metrics(self, z_1d: torch.Tensor, z_2d: torch.Tensor):
#         """
#         Compute alignment quality metrics
        
#         Returns:
#             dict with alignment metrics
#         """
#         batch_size = z_1d.shape[0]
        
#         # Cosine similarity matrix
#         sim_matrix = torch.matmul(z_1d, z_2d.T)
        
#         # Positive pair similarities (diagonal)
#         positive_sim = torch.diagonal(sim_matrix).mean().item()
        
#         # Negative pair similarities (off-diagonal)
#         mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_1d.device)
#         negative_sim = sim_matrix[mask].mean().item()
        
#         # Alignment gap (should be positive and large)
#         alignment_gap = positive_sim - negative_sim
        
#         return {
#             'positive_sim': positive_sim,
#             'negative_sim': negative_sim,
#             'alignment_gap': alignment_gap
#         }


# def train_contrastive_alignment(
#     model: ContrastiveAlignmentModel,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     num_epochs: int,
#     learning_rate: float,
#     device: torch.device,
#     output_dir: Path,
#     patience: int = 10
# ):
#     """
#     Train contrastive alignment
    
#     Args:
#         model: ContrastiveAlignmentModel
#         train_loader: Training data loader
#         val_loader: Validation data loader
#         num_epochs: Number of epochs
#         learning_rate: Learning rate
#         device: Device to train on
#         output_dir: Output directory
#         patience: Early stopping patience
    
#     Returns:
#         Trained model
#     """
#     # Fine-tune encoders with lower learning rate + train projection heads
#     encoder_params = list(model.encoder_1d.parameters()) + list(model.encoder_2d.parameters())
#     projection_params = list(model.proj_1d.parameters()) + list(model.proj_2d.parameters())

#     optimizer = torch.optim.Adam([
#         {'params': encoder_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.001},
#         {'params': projection_params, 'lr': learning_rate, 'weight_decay': 0.01}
#     ])
    
#     # Scheduler
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
#     # Training history
#     history = {
#         'train_loss': [],
#         'val_loss': [],
#         'positive_sim': [],
#         'negative_sim': [],
#         'alignment_gap': []
#     }
    
#     best_alignment_gap = 0.0
#     patience_counter = 0
    
#     print("\n" + "="*70)
#     print("PHASE 1: CONTRASTIVE ALIGNMENT TRAINING")
#     print("="*70)
    
#     for epoch in range(1, num_epochs + 1):
#         # Training
#         model.train()
#         train_loss = 0.0
        
#         for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
#             x_1d, x_2d, _ = batch
            
#             x_1d = x_1d.to(device)
#             x_2d = x_2d.to(device)
            
#             optimizer.zero_grad()
            
#             # Forward pass
#             z_1d, z_2d = model(x_1d, x_2d)
            
#             # Contrastive loss
#             loss = model.contrastive_loss(z_1d, z_2d)
            
#             # Backward pass
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
        
#         train_loss /= len(train_loader)
        
#         # Validation
#         model.eval()
#         val_loss = 0.0
#         all_metrics = []
        
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validation"):
#                 x_1d, x_2d, _ = batch
                
#                 x_1d = x_1d.to(device)
#                 x_2d = x_2d.to(device)
                
#                 # Forward pass
#                 z_1d, z_2d = model(x_1d, x_2d)
                
#                 # Loss
#                 loss = model.contrastive_loss(z_1d, z_2d)
#                 val_loss += loss.item()
                
#                 # Alignment metrics
#                 metrics = model.compute_alignment_metrics(z_1d, z_2d)
#                 all_metrics.append(metrics)
        
#         val_loss /= len(val_loader)
        
#         # Average metrics
#         avg_positive_sim = np.mean([m['positive_sim'] for m in all_metrics])
#         avg_negative_sim = np.mean([m['negative_sim'] for m in all_metrics])
#         avg_alignment_gap = np.mean([m['alignment_gap'] for m in all_metrics])
        
#         # Update scheduler
#         scheduler.step()
        
#         # Save history
#         history['train_loss'].append(train_loss)
#         history['val_loss'].append(val_loss)
#         history['positive_sim'].append(avg_positive_sim)
#         history['negative_sim'].append(avg_negative_sim)
#         history['alignment_gap'].append(avg_alignment_gap)
        
#         # Print results
#         print(f"\n{'='*70}")
#         print(f"Epoch {epoch}/{num_epochs}")
#         print(f"  Train Loss: {train_loss:.4f}")
#         print(f"  Val Loss:   {val_loss:.4f}")
#         print(f"  Positive Similarity: {avg_positive_sim:.4f} (should be high ~0.8-1.0)")
#         print(f"  Negative Similarity: {avg_negative_sim:.4f} (should be low ~0.0-0.3)")
#         print(f"  Alignment Gap:       {avg_alignment_gap:.4f} (should be large >0.5)")
        
#         # Save best model
#         if avg_alignment_gap > best_alignment_gap:
#             best_alignment_gap = avg_alignment_gap
#             patience_counter = 0
            
#             # Save checkpoint
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'alignment_gap': avg_alignment_gap,
#                 'history': history
#             }, output_dir / 'best_alignment.pth')
            
#             # Save aligned encoders separately for Phase 2
#             torch.save(model.encoder_1d.state_dict(), output_dir / 'aligned_encoder_1d.pth')
#             torch.save(model.encoder_2d.state_dict(), output_dir / 'aligned_encoder_2d.pth')
#             torch.save(model.proj_1d.state_dict(), output_dir / 'projection_1d.pth')
#             torch.save(model.proj_2d.state_dict(), output_dir / 'projection_2d.pth')
            
#             print(f"  💾 Saved best model (Alignment Gap: {avg_alignment_gap:.4f})")
#         else:
#             patience_counter += 1
        
#         # Early stopping
#         if patience_counter >= patience:
#             print(f"\n⚠️  Early stopping triggered (patience={patience})")
#             break
        
#         print("="*70)
    
#     # Save final history
#     with open(output_dir / 'alignment_history.json', 'w') as f:
#         json.dump(history, f, indent=2)
    
#     print(f"\n{'='*70}")
#     print(f"✓ Phase 1 Complete!")
#     print(f"  Best Alignment Gap: {best_alignment_gap:.4f}")
#     print(f"  Models saved to: {output_dir}")
#     print(f"{'='*70}\n")
    
#     return model


# def main():
#     parser = argparse.ArgumentParser(
#         description='Phase 1: Contrastive Alignment of 1D and 2D ECG Features'
#     )
    
#     # Model paths
#     parser.add_argument('--model-1d-path', type=str, required=True,
#                        help='Path to pre-trained 1D model')
#     parser.add_argument('--model-2d-path', type=str, required=True,
#                        help='Path to pre-trained 2D model')
    
#     # Data
#     parser.add_argument('--data-dir', type=str, required=True,
#                        help='Path to processed data directory')
#     parser.add_argument('--representation-type', type=str, default='stft',
#                        choices=['stft', 'mel', 'cwt'],
#                        help='2D representation type (default: stft)')
#     parser.add_argument('--batch-size', type=int, default=64,
#                        help='Batch size (larger is better for contrastive learning)')
#     parser.add_argument('--num-workers', type=int, default=4,
#                        help='Number of data loading workers')
    
#     # Model config
#     parser.add_argument('--dim-1d', type=int, default=384,
#                        help='Dimension of 1D features')
#     parser.add_argument('--dim-2d', type=int, default=512,
#                        help='Dimension of 2D features')
#     parser.add_argument('--projection-dim', type=int, default=128,
#                        help='Dimension of projection space')
#     parser.add_argument('--temperature', type=float, default=0.5,
#                        help='Temperature for InfoNCE loss')
    
#     # Training
#     parser.add_argument('--num-epochs', type=int, default=20,
#                        help='Number of epochs for alignment')
#     parser.add_argument('--learning-rate', type=float, default=1e-3,
#                        help='Learning rate')
#     parser.add_argument('--patience', type=int, default=10,
#                        help='Early stopping patience')
    
#     # Output
#     parser.add_argument('--output-dir', type=str, default='outputs/contrastive_alignment',
#                        help='Output directory')
    
#     # Device
#     parser.add_argument('--device', type=str, default='cuda',
#                        help='Device to use')
    
#     args = parser.parse_args()
    
#     # Setup
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     output_dir = Path(args.output_dir)
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     print("="*70)
#     print("PHASE 1: CONTRASTIVE ALIGNMENT")
#     print("="*70)
#     print(f"Device: {device}")
#     print(f"Output: {output_dir}")
#     print(f"Representation: {args.representation_type}")
#     print(f"Projection dim: {args.projection_dim}")
#     print(f"Temperature: {args.temperature}")
#     print("="*70)
    
#     # Load data
#     print("\n[Loading Data]")
    
#     data_module = None
    
#     # Try different import paths
#     try:
#         from src.data.fusion_dataset import ECGFusionDataModule
#         print("  ✓ Imported from src.data.fusion_dataset")
#         data_module = ECGFusionDataModule(
#             data_dir=args.data_dir,
#             batch_size=args.batch_size,
#             num_workers=args.num_workers,
#             representation_type=args.representation_type
#         )
#     except ImportError as e:
#         print(f"  ⚠️  Cannot import from src.data.fusion_dataset: {e}")
    
#     if data_module is None:
#         try:
#             from src.datasets.fusion_dataset import ECGFusionDataModule
#             print("  ✓ Imported from src.datasets.fusion_dataset")
#             data_module = ECGFusionDataModule(
#                 data_dir=args.data_dir,
#                 batch_size=args.batch_size,
#                 num_workers=args.num_workers,
#                 representation_type=args.representation_type
#             )
#         except ImportError as e:
#             print(f"  ⚠️  Cannot import from src.datasets.fusion_dataset: {e}")
    
#     if data_module is None:
#         try:
#             from fusion_dataset import ECGFusionDataModule
#             print("  ✓ Imported from fusion_dataset")
#             data_module = ECGFusionDataModule(
#                 data_dir=args.data_dir,
#                 batch_size=args.batch_size,
#                 num_workers=args.num_workers,
#                 representation_type=args.representation_type
#             )
#         except ImportError as e:
#             print(f"  ⚠️  Cannot import fusion_dataset: {e}")
    
#     if data_module is None:
#         print("\n❌ ERROR: Cannot import fusion_dataset from any location!")
#         print("\nPlease make sure fusion_dataset.py is in:")
#         print("  - src/data/fusion_dataset.py")
#         sys.exit(1)
    
#     try:
#         data_module.setup()
#         train_loader = data_module.train_loader
#         val_loader = data_module.val_loader
#     except Exception as e:
#         print(f"\n❌ ERROR setting up data module: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)
    
#     print(f"  ✓ Train batches: {len(train_loader)}")
#     print(f"  ✓ Val batches: {len(val_loader)}")
    
#     # Load pre-trained models
#     print("\n[Loading Pre-trained Models]")
#     print(f"  1D: {args.model_1d_path}")
#     print(f"  2D: {args.model_2d_path}")
    
#     checkpoint_1d = torch.load(args.model_1d_path, map_location=device, weights_only=False)
#     checkpoint_2d = torch.load(args.model_2d_path, map_location=device, weights_only=False)
    
#     # Load 1D model
#     if 'model_state_dict' in checkpoint_1d:
#         config = checkpoint_1d['config']
        
#         try:
#             from ecg_transformer_1d import (
#                 create_ecg_transformer_1d_small,
#                 create_ecg_transformer_1d_base,
#                 create_ecg_transformer_1d_large
#             )
            
#             model_size = config.get('model_size', 'small')
#             if model_size == 'small':
#                 model_1d = create_ecg_transformer_1d_small()
#             elif model_size == 'base':
#                 model_1d = create_ecg_transformer_1d_base()
#             elif model_size == 'large':
#                 model_1d = create_ecg_transformer_1d_large()
#             else:
#                 model_1d = create_ecg_transformer_1d_small()
            
#             model_1d.load_state_dict(checkpoint_1d['model_state_dict'])
#             print(f"  ✓ Loaded 1D model: {model_size}")
            
#         except Exception as e:
#             print(f"  ❌ Error loading 1D model: {e}")
#             return
#     else:
#         print(f"  ❌ Unknown 1D checkpoint structure")
#         return
    
#     # Load 2D model
#     if 'model_state_dict' in checkpoint_2d:
#         config = checkpoint_2d['config']
        
#         try:
#             from models.ecg_transformer_2d import (
#                 create_ecg_transformer_2d_small,
#                 create_ecg_transformer_2d_base,
#                 create_ecg_transformer_2d_large
#             )
            
#             # Detect representation type from checkpoint
#             state_dict = checkpoint_2d['model_state_dict']
#             representation_type = config.get('representation_type', 'mel_spectrogram')
            
#             print(f"  2D representation from checkpoint: {representation_type}")
            
#             model_size = config.get('model_size', 'small')
            
#             if model_size == 'small':
#                 model_2d = create_ecg_transformer_2d_small(
#                     representation_type=representation_type,
#                     representation_params=None
#                 )
#             elif model_size == 'base':
#                 model_2d = create_ecg_transformer_2d_base(
#                     representation_type=representation_type,
#                     representation_params=None
#                 )
#             elif model_size == 'large':
#                 model_2d = create_ecg_transformer_2d_large(
#                     representation_type=representation_type,
#                     representation_params=None
#                 )
#             else:
#                 model_2d = create_ecg_transformer_2d_small(
#                     representation_type=representation_type,
#                     representation_params=None
#                 )
            
#             model_2d.load_state_dict(checkpoint_2d['model_state_dict'])
#             print(f"  ✓ Loaded 2D model: {model_size}")
            
#         except Exception as e:
#             print(f"  ❌ Error loading 2D model: {e}")
#             import traceback
#             traceback.print_exc()
#             return
#     else:
#         print(f"  ❌ Unknown 2D checkpoint structure")
#         return
    
#     model_1d = model_1d.to(device)
#     model_2d = model_2d.to(device)
    
#     # Extract encoders
#     print(f"\n[Extracting Encoders]")
    
#     # For 1D model
#     if hasattr(model_1d, 'encoder'):
#         encoder_1d = model_1d.encoder
#     else:
#         class EncoderWrapper(nn.Module):
#             def __init__(self, model):
#                 super().__init__()
#                 self.patch_embed = model.patch_embed
#                 self.blocks = model.blocks
#                 self.norm = model.norm
            
#             def forward(self, x):
#                 x = self.patch_embed(x)
#                 for block in self.blocks:
#                     x = block(x)
#                 x = self.norm(x)
#                 x = x.mean(dim=1)
#                 return x
        
#         encoder_1d = EncoderWrapper(model_1d)
#         print(f"  1D: Wrapped encoder")
    
#     # For 2D model
#     if hasattr(model_2d, 'encoder'):
#         encoder_2d = model_2d.encoder
#     else:
#         class EncoderWrapper2D(nn.Module):
#             def __init__(self, model):
#                 super().__init__()
#                 self.patch_embed = model.patch_embed
#                 self.blocks = model.blocks
#                 self.norm = model.norm
            
#             def forward(self, x):
#                 x = self.patch_embed(x)
#                 for block in self.blocks:
#                     x = block(x)
#                 x = self.norm(x)
#                 x = x.mean(dim=1)
#                 return x
        
#         encoder_2d = EncoderWrapper2D(model_2d)
#         print(f"  2D: Wrapped encoder")
    
#     encoder_1d = encoder_1d.to(device)
#     encoder_2d = encoder_2d.to(device)
    
#     # Detect actual dimensions
#     print(f"\n[Detecting Encoder Dimensions]")
#     try:
#         with torch.no_grad():
#             first_batch = next(iter(train_loader))
#             x_1d, x_2d, _ = first_batch
#             x_1d = x_1d.to(device)
#             x_2d = x_2d.to(device)
            
#             feat_1d = encoder_1d(x_1d)
#             feat_2d = encoder_2d(x_2d)
            
#             if feat_1d.dim() == 3:
#                 args.dim_1d = feat_1d.shape[-1]
#             elif feat_1d.dim() == 2:
#                 args.dim_1d = feat_1d.shape[-1]
            
#             if feat_2d.dim() == 3:
#                 args.dim_2d = feat_2d.shape[-1]
#             elif feat_2d.dim() == 2:
#                 args.dim_2d = feat_2d.shape[-1]
            
#             print(f"  1D: {x_1d.shape} → {feat_1d.shape} (dim={args.dim_1d})")
#             print(f"  2D: {x_2d.shape} → {feat_2d.shape} (dim={args.dim_2d})")
            
#     except Exception as e:
#         print(f"  ⚠️  Using default dimensions: 1D={args.dim_1d}, 2D={args.dim_2d}")
#         print(f"  Error: {e}")
    
#     # Create contrastive alignment model
#     print("\n[Creating Contrastive Alignment Model]")
    
#     model = ContrastiveAlignmentModel(
#         encoder_1d=encoder_1d,
#         encoder_2d=encoder_2d,
#         dim_1d=args.dim_1d,
#         dim_2d=args.dim_2d,
#         projection_dim=args.projection_dim,
#         temperature=args.temperature
#     )
#     model = model.to(device)
    
#     # Train
#     trained_model = train_contrastive_alignment(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         num_epochs=args.num_epochs,
#         learning_rate=args.learning_rate,
#         device=device,
#         output_dir=output_dir,
#         patience=args.patience
#     )
    
#     print("\n✓ Phase 1 Complete!")
#     print(f"\nNext step: Run Phase 2 with aligned encoders:")
#     print(f"  python scripts/train_phase2_cross_attention.py \\")
#     print(f"    --aligned-encoder-1d {output_dir}/aligned_encoder_1d.pth \\")
#     print(f"    --aligned-encoder-2d {output_dir}/aligned_encoder_2d.pth \\")
#     print(f"    --data-dir {args.data_dir}")


# if __name__ == "__main__":
#     main()
# # #!/usr/bin/env python3
# # """
# # Phase 1: Contrastive Alignment of 1D and 2D ECG Features
# # Aligns features from different modalities using InfoNCE contrastive loss

# # Based on: "Cross-modal attention fusion for depression detection"
# # Adapted for ECG temporal (1D) and spectral (2D) alignment
# # """

# # import argparse
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader
# # import numpy as np
# # from pathlib import Path
# # from tqdm import tqdm
# # import json
# # import sys

# # # Ensure imports work regardless of current working directory: add project root and src to sys.path
# # project_root = Path(__file__).resolve().parent.parent
# # src_dir = project_root / 'src'
# # if str(project_root) not in sys.path:
# #     sys.path.insert(0, str(project_root))
# # if str(src_dir) not in sys.path:
# #     sys.path.insert(0, str(src_dir))
# # # Also add common subfolders for convenience
# # data_dir = src_dir / 'data'
# # models_dir = src_dir / 'models'
# # if str(data_dir) not in sys.path:
# #     sys.path.insert(0, str(data_dir))
# # if str(models_dir) not in sys.path:
# #     sys.path.insert(0, str(models_dir))


# # class ContrastiveAlignmentModel(nn.Module):
# #     """
# #     Contrastive learning model to align 1D and 2D ECG features
    
# #     Uses InfoNCE loss: positive pairs are (1D, 2D) from same sample
# #     This ensures both modalities learn aligned representations
# #     """
    
# #     def __init__(
# #         self,
# #         encoder_1d: nn.Module,
# #         encoder_2d: nn.Module,
# #         dim_1d: int = 384,
# #         dim_2d: int = 512,
# #         projection_dim: int = 128,
# #         temperature: float = 0.07
# #     ):
# #         """
# #         Args:
# #             encoder_1d: Pre-trained 1D encoder
# #             encoder_2d: Pre-trained 2D encoder
# #             dim_1d: Dimension of 1D features
# #             dim_2d: Dimension of 2D features
# #             projection_dim: Dimension of shared projection space
# #             temperature: Temperature for InfoNCE loss
# #         """
# #         super().__init__()
        
# #         self.encoder_1d = encoder_1d
# #         self.encoder_2d = encoder_2d
        
# #         # Projection heads to map to shared space
# #         self.proj_1d = nn.Sequential(
# #             nn.LayerNorm(dim_1d),
# #             nn.Linear(dim_1d, 256),
# #             nn.ReLU(),
# #             nn.Dropout(0.1),
# #             nn.Linear(256, projection_dim)
# #         )
        
# #         self.proj_2d = nn.Sequential(
# #             nn.LayerNorm(dim_2d),
# #             nn.Linear(dim_2d, 256),
# #             nn.ReLU(),
# #             nn.Dropout(0.1),
# #             nn.Linear(256, projection_dim)
# #         )
        
# #         self.temperature = temperature
        
# #         print(f"\n[ContrastiveAlignment] Created")
# #         print(f"  1D dim: {dim_1d} → Projection: {projection_dim}")
# #         print(f"  2D dim: {dim_2d} → Projection: {projection_dim}")
# #         print(f"  Temperature: {temperature}")
    
# #     def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
# #         """
# #         Forward pass
        
# #         Args:
# #             x_1d: (B, C, L) - 1D ECG signal
# #             x_2d: (B, C, H, W) - 2D ECG representation
        
# #         Returns:
# #             z_1d: (B, projection_dim) - Projected 1D features
# #             z_2d: (B, projection_dim) - Projected 2D features
# #         """
# #         # Extract features from encoders
# #         feat_1d = self.encoder_1d(x_1d)  # (B, N, dim_1d) or (B, dim_1d)
# #         feat_2d = self.encoder_2d(x_2d)  # (B, M, dim_2d) or (B, dim_2d)
        
# #         # Pool if sequence output
# #         if feat_1d.dim() == 3:  # (B, N, D)
# #             feat_1d = feat_1d.mean(dim=1)  # (B, dim_1d)
# #         if feat_2d.dim() == 3:  # (B, M, D)
# #             feat_2d = feat_2d.mean(dim=1)  # (B, dim_2d)
        
# #         # Project to shared space
# #         z_1d = self.proj_1d(feat_1d)  # (B, projection_dim)
# #         z_2d = self.proj_2d(feat_2d)  # (B, projection_dim)
        
# #         # L2 normalize (crucial for contrastive learning)
# #         z_1d = F.normalize(z_1d, dim=1)
# #         z_2d = F.normalize(z_2d, dim=1)
        
# #         return z_1d, z_2d
    
# #     def contrastive_loss(self, z_1d: torch.Tensor, z_2d: torch.Tensor):
# #         """
# #         InfoNCE contrastive loss
        
# #         Positive pairs: (z_1d[i], z_2d[i]) - same sample
# #         Negative pairs: all other combinations
        
# #         Args:
# #             z_1d: (B, D) - Normalized 1D projections
# #             z_2d: (B, D) - Normalized 2D projections
        
# #         Returns:
# #             loss: Scalar contrastive loss
# #         """
# #         batch_size = z_1d.shape[0]
        
# #         # Compute similarity matrix (cosine similarity)
# #         logits = torch.matmul(z_1d, z_2d.T) / self.temperature  # (B, B)
        
# #         # Labels: diagonal elements are positive pairs
# #         labels = torch.arange(batch_size).to(z_1d.device)
        
# #         # Symmetric loss (1D→2D and 2D→1D)
# #         loss_1d_to_2d = F.cross_entropy(logits, labels)
# #         loss_2d_to_1d = F.cross_entropy(logits.T, labels)
        
# #         loss = (loss_1d_to_2d + loss_2d_to_1d) / 2
        
# #         return loss
    
# #     def compute_alignment_metrics(self, z_1d: torch.Tensor, z_2d: torch.Tensor):
# #         """
# #         Compute alignment quality metrics
        
# #         Returns:
# #             dict with alignment metrics
# #         """
# #         batch_size = z_1d.shape[0]
        
# #         # Cosine similarity matrix
# #         sim_matrix = torch.matmul(z_1d, z_2d.T)  # (B, B)
        
# #         # Positive pair similarities (diagonal)
# #         positive_sim = torch.diagonal(sim_matrix).mean().item()
        
# #         # Negative pair similarities (off-diagonal)
# #         mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_1d.device)
# #         negative_sim = sim_matrix[mask].mean().item()
        
# #         # Alignment gap (should be positive and large)
# #         alignment_gap = positive_sim - negative_sim
        
# #         return {
# #             'positive_sim': positive_sim,
# #             'negative_sim': negative_sim,
# #             'alignment_gap': alignment_gap
# #         }


# # def train_contrastive_alignment(
# #     model: ContrastiveAlignmentModel,
# #     train_loader: DataLoader,
# #     val_loader: DataLoader,
# #     num_epochs: int,
# #     learning_rate: float,
# #     device: torch.device,
# #     output_dir: Path,
# #     patience: int = 10
# # ):
# #     """
# #     Train contrastive alignment
    
# #     Args:
# #         model: ContrastiveAlignmentModel
# #         train_loader: Training data loader
# #         val_loader: Validation data loader
# #         num_epochs: Number of epochs
# #         learning_rate: Learning rate
# #         device: Device to train on
# #         output_dir: Output directory
# #         patience: Early stopping patience
    
# #     Returns:
# #         Trained model
# #     """
# #     # Optimizer (only train projection heads, freeze encoders)
# #     # Fine-tune encoders with lower learning rate + train projection heads
# #     encoder_params = list(model.encoder_1d.parameters()) + list(model.encoder_2d.parameters())
# #     projection_params = list(model.proj_1d.parameters()) + list(model.proj_2d.parameters())

# #     # Use different learning rates: encoders at 10% of main LR, projections at full LR
# #     optimizer = torch.optim.Adam([
# #         {'params': encoder_params, 'lr': learning_rate * 0.1, 'weight_decay': 0.001},  # Fine-tune encoders slowly
# #         {'params': projection_params, 'lr': learning_rate, 'weight_decay': 0.01}      # Train projections normally
# #     ])
    
# #     # Scheduler
# #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
# #     # Training history
# #     history = {
# #         'train_loss': [],
# #         'val_loss': [],
# #         'positive_sim': [],
# #         'negative_sim': [],
# #         'alignment_gap': []
# #     }
    
# #     best_alignment_gap = 0.0
# #     patience_counter = 0
    
# #     print("\n" + "="*70)
# #     print("PHASE 1: CONTRASTIVE ALIGNMENT TRAINING")
# #     print("="*70)
    
# #     for epoch in range(1, num_epochs + 1):
# #         # ====================================================================
# #         # Training
# #         # ====================================================================
# #         model.train()
# #         train_loss = 0.0
        
# #         for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")):
# #             x_1d, x_2d, _ = batch  # We don't need labels for contrastive learning
            
# #             x_1d = x_1d.to(device)
# #             x_2d = x_2d.to(device)
            
# #             optimizer.zero_grad()
            
# #             # Forward pass
# #             z_1d, z_2d = model(x_1d, x_2d)
            
# #             # Contrastive loss
# #             loss = model.contrastive_loss(z_1d, z_2d)
            
# #             # Backward pass
# #             loss.backward()
# #             optimizer.step()
            
# #             train_loss += loss.item()
        
# #         train_loss /= len(train_loader)
        
# #         # ====================================================================
# #         # Validation
# #         # ====================================================================
# #         model.eval()
# #         val_loss = 0.0
# #         all_metrics = []
        
# #         with torch.no_grad():
# #             for batch in tqdm(val_loader, desc="Validation"):
# #                 x_1d, x_2d, _ = batch
                
# #                 x_1d = x_1d.to(device)
# #                 x_2d = x_2d.to(device)
                
# #                 # Forward pass
# #                 z_1d, z_2d = model(x_1d, x_2d)
                
# #                 # Loss
# #                 loss = model.contrastive_loss(z_1d, z_2d)
# #                 val_loss += loss.item()
                
# #                 # Alignment metrics
# #                 metrics = model.compute_alignment_metrics(z_1d, z_2d)
# #                 all_metrics.append(metrics)
        
# #         val_loss /= len(val_loader)
        
# #         # Average metrics
# #         avg_positive_sim = np.mean([m['positive_sim'] for m in all_metrics])
# #         avg_negative_sim = np.mean([m['negative_sim'] for m in all_metrics])
# #         avg_alignment_gap = np.mean([m['alignment_gap'] for m in all_metrics])
        
# #         # Update scheduler
# #         scheduler.step()
        
# #         # Save history
# #         history['train_loss'].append(train_loss)
# #         history['val_loss'].append(val_loss)
# #         history['positive_sim'].append(avg_positive_sim)
# #         history['negative_sim'].append(avg_negative_sim)
# #         history['alignment_gap'].append(avg_alignment_gap)
        
# #         # Print results
# #         print(f"\n{'='*70}")
# #         print(f"Epoch {epoch}/{num_epochs}")
# #         print(f"  Train Loss: {train_loss:.4f}")
# #         print(f"  Val Loss:   {val_loss:.4f}")
# #         print(f"  Positive Similarity: {avg_positive_sim:.4f} (should be high ~0.8-1.0)")
# #         print(f"  Negative Similarity: {avg_negative_sim:.4f} (should be low ~0.0-0.3)")
# #         print(f"  Alignment Gap:       {avg_alignment_gap:.4f} (should be large >0.5)")
        
# #         # Save best model
# #         if avg_alignment_gap > best_alignment_gap:
# #             best_alignment_gap = avg_alignment_gap
# #             patience_counter = 0
            
# #             # Save checkpoint
# #             torch.save({
# #                 'epoch': epoch,
# #                 'model_state_dict': model.state_dict(),
# #                 'optimizer_state_dict': optimizer.state_dict(),
# #                 'alignment_gap': avg_alignment_gap,
# #                 'history': history
# #             }, output_dir / 'best_alignment.pth')
            
# #             # Save aligned encoders separately for Phase 2
# #             torch.save(model.encoder_1d.state_dict(), output_dir / 'aligned_encoder_1d.pth')
# #             torch.save(model.encoder_2d.state_dict(), output_dir / 'aligned_encoder_2d.pth')
# #             torch.save(model.proj_1d.state_dict(), output_dir / 'projection_1d.pth')
# #             torch.save(model.proj_2d.state_dict(), output_dir / 'projection_2d.pth')
            
# #             print(f"  💾 Saved best model (Alignment Gap: {avg_alignment_gap:.4f})")
# #         else:
# #             patience_counter += 1
        
# #         # Early stopping
# #         if patience_counter >= patience:
# #             print(f"\n⚠️  Early stopping triggered (patience={patience})")
# #             break
        
# #         print("="*70)
    
# #     # Save final history
# #     with open(output_dir / 'alignment_history.json', 'w') as f:
# #         json.dump(history, f, indent=2)
    
# #     print(f"\n{'='*70}")
# #     print(f"✓ Phase 1 Complete!")
# #     print(f"  Best Alignment Gap: {best_alignment_gap:.4f}")
# #     print(f"  Models saved to: {output_dir}")
# #     print(f"{'='*70}\n")
    
# #     return model


# # def main():
# #     parser = argparse.ArgumentParser(
# #         description='Phase 1: Contrastive Alignment of 1D and 2D ECG Features'
# #     )
    
# #     # Model paths
# #     parser.add_argument('--model-1d-path', type=str, required=True,
# #                        help='Path to pre-trained 1D model')
# #     parser.add_argument('--model-2d-path', type=str, required=True,
# #                        help='Path to pre-trained 2D model')
    
# #     # Data
# #     parser.add_argument('--data-dir', type=str, required=True,
# #                        help='Path to processed data directory')
# #     parser.add_argument('--batch-size', type=int, default=64,
# #                        help='Batch size (larger is better for contrastive learning)')
# #     parser.add_argument('--num-workers', type=int, default=4,
# #                        help='Number of data loading workers')
    
# #     # Model config
# #     parser.add_argument('--dim-1d', type=int, default=384,
# #                        help='Dimension of 1D features')
# #     parser.add_argument('--dim-2d', type=int, default=512,
# #                        help='Dimension of 2D features')
# #     parser.add_argument('--projection-dim', type=int, default=128,
# #                        help='Dimension of projection space')
# #     parser.add_argument('--temperature', type=float, default=0.5,
# #                        help='Temperature for InfoNCE loss')
    
# #     # Training
# #     parser.add_argument('--num-epochs', type=int, default=20,
# #                        help='Number of epochs for alignment')
# #     parser.add_argument('--learning-rate', type=float, default=1e-3,
# #                        help='Learning rate')
# #     parser.add_argument('--patience', type=int, default=10,
# #                        help='Early stopping patience')
    
# #     # Output
# #     parser.add_argument('--output-dir', type=str, default='outputs/contrastive_alignment',
# #                        help='Output directory')
    
# #     # Device
# #     parser.add_argument('--device', type=str, default='cuda',
# #                        help='Device to use')
    
# #     args = parser.parse_args()
    
# #     # Setup
# #     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# #     output_dir = Path(args.output_dir)
# #     output_dir.mkdir(parents=True, exist_ok=True)
    
# #     print("="*70)
# #     print("PHASE 1: CONTRASTIVE ALIGNMENT")
# #     print("="*70)
# #     print(f"Device: {device}")
# #     print(f"Output: {output_dir}")
# #     print(f"Projection dim: {args.projection_dim}")
# #     print(f"Temperature: {args.temperature}")
# #     print("="*70)
    
# #     # Load data
# #     print("\n[Loading Data]")
    
# #     # Import dataset - try multiple import paths
# #     data_module = None
    
# #     # Try import path 1: from src.data
# #     try:
# #         from src.data.fusion_dataset import ECGFusionDataModule
# #         print("  ✓ Imported from src.data.fusion_dataset")
# #         data_module = ECGFusionDataModule(
# #             data_dir=args.data_dir,
# #             batch_size=args.batch_size,
# #             num_workers=args.num_workers,
# #             representation_type='cwt'  # Use mel to match your 2D checkpoint
# #         )
# #     except ImportError as e:
# #         print(f"  ⚠️  Cannot import from src.data.fusion_dataset: {e}")
    
# #     # Try import path 2: from src.datasets
# #     if data_module is None:
# #         try:
# #             from src.datasets.fusion_dataset import ECGFusionDataModule
# #             print("  ✓ Imported from src.datasets.fusion_dataset")
# #             data_module = ECGFusionDataModule(
# #                 data_dir=args.data_dir,
# #                 batch_size=args.batch_size,
# #                 num_workers=args.num_workers,
# #                 representation_type='cwt'
# #             )
# #         except ImportError as e:
# #             print(f"  ⚠️  Cannot import from src.datasets.fusion_dataset: {e}")
    
# #     # Try import path 3: direct import
# #     if data_module is None:
# #         try:
# #             from fusion_dataset import ECGFusionDataModule
# #             print("  ✓ Imported from fusion_dataset")
# #             data_module = ECGFusionDataModule(
# #                 data_dir=args.data_dir,
# #                 batch_size=args.batch_size,
# #                 num_workers=args.num_workers,
# #                 representation_type='cwt'
# #             )
# #         except ImportError as e:
# #             print(f"  ⚠️  Cannot import fusion_dataset: {e}")
    
# #     if data_module is None:
# #         print("\n❌ ERROR: Cannot import fusion_dataset from any location!")
# #         print("\nPlease make sure fusion_dataset.py is in one of:")
# #         print("  - src/data/fusion_dataset.py")
# #         print("  - src/datasets/fusion_dataset.py")
# #         print("\nOr add the correct path to sys.path")
# #         sys.exit(1)
    
# #     try:
# #         data_module.setup()
# #         train_loader = data_module.train_loader
# #         val_loader = data_module.val_loader
# #     except Exception as e:
# #         print(f"\n❌ ERROR setting up data module: {e}")
# #         import traceback
# #         traceback.print_exc()
# #         sys.exit(1)
    
# #     print(f"  ✓ Train batches: {len(train_loader)}")
# #     print(f"  ✓ Val batches: {len(val_loader)}")
    
# #     # Load pre-trained models
# #     print("\n[Loading Pre-trained Models]")
# #     print(f"  1D: {args.model_1d_path}")
# #     print(f"  2D: {args.model_2d_path}")
    
# #     checkpoint_1d = torch.load(args.model_1d_path, map_location=device, weights_only=False)
# #     checkpoint_2d = torch.load(args.model_2d_path, map_location=device, weights_only=False)
    
# #     print(f"  1D checkpoint keys: {list(checkpoint_1d.keys())}")
# #     print(f"  2D checkpoint keys: {list(checkpoint_2d.keys())}")
    
# #     # Handle different checkpoint formats
# #     if isinstance(checkpoint_1d, nn.Module):
# #         model_1d = checkpoint_1d
# #         print(f"  1D: Direct model")
# #     elif 'model' in checkpoint_1d:
# #         model_1d = checkpoint_1d['model']
# #         print(f"  1D: Extracted from 'model' key")
# #     elif 'model_state_dict' in checkpoint_1d:
# #         print(f"  ⚠️  1D checkpoint has 'model_state_dict' - reconstructing from config")
        
# #         if 'config' not in checkpoint_1d:
# #             print(f"  ❌ No config found in checkpoint!")
# #             return
        
# #         config = checkpoint_1d['config']
# #         print(f"  1D Config: {config}")
        
# #         # Import model class
# #         try:
# #             sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'models'))
            
# #             from ecg_transformer_1d import (
# #                 create_ecg_transformer_1d_small,
# #                 create_ecg_transformer_1d_base,
# #                 create_ecg_transformer_1d_large
# #             )
            
# #             # Map model size to constructor
# #             model_size = config.get('model_size', 'small')
# #             if model_size == 'small':
# #                 model_1d = create_ecg_transformer_1d_small()
# #             elif model_size == 'base':
# #                 model_1d = create_ecg_transformer_1d_base()
# #             elif model_size == 'large':
# #                 model_1d = create_ecg_transformer_1d_large()
# #             else:
# #                 print(f"  Unknown model size: {model_size}, using small")
# #                 model_1d = create_ecg_transformer_1d_small()
            
# #             print(f"  ✓ Created 1D model: {model_size}")
            
# #             # Load state dict
# #             model_1d.load_state_dict(checkpoint_1d['model_state_dict'])
# #             print(f"  ✓ Loaded 1D state dict")
            
# #         except Exception as e:
# #             print(f"  ❌ Error loading 1D model: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return
# #     else:
# #         print(f"  Unknown 1D checkpoint structure: {checkpoint_1d.keys()}")
# #         return
    
# #     # Same for 2D
# #     if isinstance(checkpoint_2d, nn.Module):
# #         model_2d = checkpoint_2d
# #         print(f"  2D: Direct model")
# #     elif 'model' in checkpoint_2d:
# #         model_2d = checkpoint_2d['model']
# #         print(f"  2D: Extracted from 'model' key")
# #     elif 'model_state_dict' in checkpoint_2d:
# #         print(f"  ⚠️  2D checkpoint has 'model_state_dict' - reconstructing from config")
        
# #         if 'config' not in checkpoint_2d:
# #             print(f"  ❌ No config found in checkpoint!")
# #             return
        
# #         config = checkpoint_2d['config']
# #         print(f"  2D Config: {config}")
        
# #         try:
# #             # Add src/models to path to handle relative imports
# #             models_dir = Path(__file__).parent.parent / 'src' / 'models'
# #             if str(models_dir) not in sys.path:
# #                 sys.path.insert(0, str(models_dir))
            
# #             # Also add src to path
# #             src_dir = Path(__file__).parent.parent / 'src'
# #             if str(src_dir) not in sys.path:
# #                 sys.path.insert(0, str(src_dir))
            
# #             # Now import should work
# #             from models.ecg_transformer_2d import (
# #                 create_ecg_transformer_2d_small,
# #                 create_ecg_transformer_2d_base,
# #                 create_ecg_transformer_2d_large
# #             )
            
# #             # CRITICAL: Detect representation type from state_dict keys
# #             state_dict = checkpoint_2d['model_state_dict']
            
# #             # Check what representation was used during training
# #             if any('mel_spectrogram' in k for k in state_dict.keys()):
# #                 representation_type = 'mel_spectrogram'
# #                 representation_params = None  # Use defaults
# #                 print(f"  Detected representation: mel_spectrogram")
# #             elif any('cwt' in k for k in state_dict.keys()):
# #                 representation_type = 'cwt'
# #                 representation_params = {
# #                     'wavelet': 'morl',
# #                     'scales': 128,
# #                     'sampling_rate': 250
# #                 }
# #                 print(f"  Detected representation: cwt")
# #             else:
# #                 # Default to mel_spectrogram (most common)
# #                 representation_type = 'mel_spectrogram'
# #                 representation_params = None
# #                 print(f"  No representation detected, using default: mel_spectrogram")
            
# #             # Map model size to constructor
# #             model_size = config.get('model_size', 'small')
            
# #             if model_size == 'small':
# #                 model_2d = create_ecg_transformer_2d_small(
# #                     representation_type=representation_type,
# #                     representation_params=representation_params
# #                 )
# #             elif model_size == 'base':
# #                 model_2d = create_ecg_transformer_2d_base(
# #                     representation_type=representation_type,
# #                     representation_params=representation_params
# #                 )
# #             elif model_size == 'large':
# #                 model_2d = create_ecg_transformer_2d_large(
# #                     representation_type=representation_type,
# #                     representation_params=representation_params
# #                 )
# #             else:
# #                 print(f"  Unknown model size: {model_size}, using small")
# #                 model_2d = create_ecg_transformer_2d_small(
# #                     representation_type=representation_type,
# #                     representation_params=representation_params
# #                 )
            
# #             print(f"  ✓ Created 2D model: {model_size}")
            
# #             # Load state dict
# #             model_2d.load_state_dict(checkpoint_2d['model_state_dict'])
# #             print(f"  ✓ Loaded 2D state dict")
            
# #         except Exception as e:
# #             print(f"  ❌ Error loading 2D model: {e}")
# #             import traceback
# #             traceback.print_exc()
# #             return
# #     else:
# #         print(f"  Unknown 2D checkpoint structure: {checkpoint_2d.keys()}")
# #         return
    
# #     model_1d = model_1d.to(device)
# #     model_2d = model_2d.to(device)
    
# #     # Extract encoders (remove classification heads)
# #     print(f"\n[Extracting Encoders]")
    
# #     # For 1D model
# #     if hasattr(model_1d, 'encoder'):
# #         encoder_1d = model_1d.encoder
# #         print(f"  1D: Found 'encoder' attribute")
# #     elif hasattr(model_1d, 'features'):
# #         encoder_1d = model_1d.features
# #         print(f"  1D: Found 'features' attribute")
# #     else:
# #         # Check what children the model has
# #         children = list(model_1d.named_children())
# #         print(f"  1D model children: {[name for name, _ in children]}")
        
# #         # Find all non-classifier modules
# #         encoder_modules = []
# #         for name, module in children:
# #             if name not in ['head', 'classifier', 'fc', 'output']:
# #                 encoder_modules.append((name, module))
        
# #         if len(encoder_modules) == 1:
# #             # Single encoder module
# #             encoder_1d = encoder_modules[0][1]
# #             print(f"  1D: Using single encoder module '{encoder_modules[0][0]}'")
# #         elif len(encoder_modules) > 1:
# #             # Multiple modules - create a proper wrapper that handles ModuleList
# #             class EncoderWrapper(nn.Module):
# #                 def __init__(self, model):
# #                     super().__init__()
# #                     self.patch_embed = model.patch_embed
# #                     self.blocks = model.blocks  # ModuleList - don't iterate in init
# #                     self.norm = model.norm
                
# #                 def forward(self, x):
# #                     x = self.patch_embed(x)
# #                     # Iterate through blocks during forward pass
# #                     for block in self.blocks:
# #                         x = block(x)
# #                     x = self.norm(x)
# #                     x = x.mean(dim=1)  # Global average pooling
# #                     return x
            
# #             encoder_1d = EncoderWrapper(model_1d)
# #             print(f"  1D: Wrapped {len(encoder_modules)} modules into encoder")
# #         else:
# #             # Last resort - use whole model but remove last layer
# #             print(f"  1D: Warning - using full model as encoder")
# #             encoder_1d = model_1d
    
# #     # For 2D model (same logic)
# #     if hasattr(model_2d, 'encoder'):
# #         encoder_2d = model_2d.encoder
# #         print(f"  2D: Found 'encoder' attribute")
# #     elif hasattr(model_2d, 'features'):
# #         encoder_2d = model_2d.features
# #         print(f"  2D: Found 'features' attribute")
# #     else:
# #         children = list(model_2d.named_children())
# #         print(f"  2D model children: {[name for name, _ in children]}")
        
# #         encoder_modules = []
# #         for name, module in children:
# #             if name not in ['head', 'classifier', 'fc', 'output']:
# #                 encoder_modules.append((name, module))
        
# #         if len(encoder_modules) == 1:
# #             encoder_2d = encoder_modules[0][1]
# #             print(f"  2D: Using single encoder module '{encoder_modules[0][0]}'")
# #         elif len(encoder_modules) > 1:
# #             # Multiple modules - create a proper wrapper that handles ModuleList
# #             class EncoderWrapper2D(nn.Module):
# #                 def __init__(self, model):
# #                     super().__init__()
# #                     # Skip ecg_to_2d since dataset already provides 2D data
# #                     self.patch_embed = model.patch_embed
# #                     self.blocks = model.blocks
# #                     self.norm = model.norm
                
# #                 def forward(self, x):
# #                     # x is already 2D from dataset (B, n_leads, freq, time)
# #                     x = self.patch_embed(x)
# #                     for block in self.blocks:
# #                         x = block(x)
# #                     x = self.norm(x)
# #                     x = x.mean(dim=1)
# #                     return x
            
# #             encoder_2d = EncoderWrapper2D(model_2d)
# #             print(f"  2D: Wrapped {len(encoder_modules)} modules into encoder")
# #         else:
# #             print(f"  2D: Warning - using full model as encoder")
# #             encoder_2d = model_2d
    
# #     # Test encoders with dummy data
# #     print(f"\n[Testing Encoders]")
    
# #     # Instead of testing with hardcoded dummy data, infer dimensions from model's expected input
# #     # Or skip testing and infer from first batch
# #     print(f"  Skipping dummy test, will infer dimensions from actual data batch")
    
# #     # Get actual dimensions from the model's configuration
# #     # For 1D: check patch_embed to infer expected dimensions
# #     # For 2D: same
    
# #     # Default dimensions (will be updated after first batch)
# #     actual_dim_1d = args.dim_1d
# #     actual_dim_2d = args.dim_2d
    
# #     print(f"  Using default dimensions: 1D={actual_dim_1d}, 2D={actual_dim_2d}")
# #     print(f"  (Will be verified on first training batch)")
    
# #     encoder_1d = encoder_1d.to(device)
# #     encoder_2d = encoder_2d.to(device)
    
# #     print(f"  ✓ Encoders extracted successfully")
    
# #     # Create contrastive alignment model
# #     print("\n[Creating Contrastive Alignment Model]")
    
# #     # Detect actual encoder output dimensions from first batch
# #     print(f"  Detecting encoder output dimensions...")
# #     try:
# #         with torch.no_grad():
# #             first_batch = next(iter(train_loader))
# #             x_1d, x_2d, _ = first_batch
# #             x_1d = x_1d.to(device)
# #             x_2d = x_2d.to(device)
            
# #             feat_1d = encoder_1d(x_1d)
# #             feat_2d = encoder_2d(x_2d)
            
# #             if feat_1d.dim() == 3:
# #                 actual_dim_1d = feat_1d.shape[-1]
# #             elif feat_1d.dim() == 2:
# #                 actual_dim_1d = feat_1d.shape[-1]
# #             else:
# #                 actual_dim_1d = args.dim_1d
            
# #             if feat_2d.dim() == 3:
# #                 actual_dim_2d = feat_2d.shape[-1]
# #             elif feat_2d.dim() == 2:
# #                 actual_dim_2d = feat_2d.shape[-1]
# #             else:
# #                 actual_dim_2d = args.dim_2d
            
# #             print(f"  1D: {x_1d.shape} → {feat_1d.shape} (dim={actual_dim_1d})")
# #             print(f"  2D: {x_2d.shape} → {feat_2d.shape} (dim={actual_dim_2d})")
            
# #             args.dim_1d = actual_dim_1d
# #             args.dim_2d = actual_dim_2d
            
# #     except Exception as e:
# #         print(f"  ⚠️  Using default dimensions: 1D={args.dim_1d}, 2D={args.dim_2d}")
    
# #     model = ContrastiveAlignmentModel(
# #         encoder_1d=encoder_1d,
# #         encoder_2d=encoder_2d,
# #         dim_1d=args.dim_1d,
# #         dim_2d=args.dim_2d,
# #         projection_dim=args.projection_dim,
# #         temperature=args.temperature
# #     )
# #     model = model.to(device)
    
# #     # Train
# #     trained_model = train_contrastive_alignment(
# #         model=model,
# #         train_loader=train_loader,
# #         val_loader=val_loader,
# #         num_epochs=args.num_epochs,
# #         learning_rate=args.learning_rate,
# #         device=device,
# #         output_dir=output_dir,
# #         patience=args.patience
# #     )
    
# #     print("\n✓ Phase 1 Complete!")
# #     print(f"\nNext step: Run Phase 2 with aligned encoders:")
# #     print(f"  python scripts/train_phase2_cross_attention.py \\")
# #     print(f"    --aligned-encoder-1d {output_dir}/aligned_encoder_1d.pth \\")
# #     print(f"    --aligned-encoder-2d {output_dir}/aligned_encoder_2d.pth \\")
# #     print(f"    --data-dir {args.data_dir}")


# # if __name__ == "__main__":
# #     main()