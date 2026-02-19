############################################################################################3
# #!/usr/bin/env python3
# """
# Phase 2: Cross-Attention Fusion for ECG Classification
# Stable "fair" training: label smoothing, warm restarts scheduler,
# gradient accumulation, grad clipping. Includes saliency overlays & gallery.

# This version preserves core hyperparameters (LR, optimizer type, architecture).
# """

# import argparse
# from pathlib import Path
# import sys
# import json
# import numpy as np
# from tqdm import tqdm
# from pathlib import Path
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# # plotting
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# # --- add at top of file ---
# import numpy as np
# try:
#     from scipy.signal import savgol_filter
#     _HAS_SAVGOL = True
# except Exception:
#     _HAS_SAVGOL = False

# def _moving_avg(x, k=5):
#     k = max(1, int(k))
#     if k == 1: 
#         return np.array(x, dtype=float)
#     w = np.ones(k) / k
#     return np.convolve(np.array(x, dtype=float), w, mode="same")

# def _smooth_series(y, method="ma", ma_window=5, sg_window=9, sg_poly=2):
#     y = np.array(y, dtype=float)
#     if method == "sg" and _HAS_SAVGOL and len(y) >= sg_window and sg_window % 2 == 1:
#         return savgol_filter(y, window_length=sg_window, polyorder=sg_poly, mode="interp")
#     else:
#         # default to moving average
#         return _moving_avg(y, k=ma_window)


# # smoothing for 1D saliency
# from scipy.ndimage import gaussian_filter1d

# # --------------------------
# # Project paths
# # --------------------------
# project_root = Path(__file__).resolve().parent.parent
# src_dir = project_root / 'src'
# for path in [project_root, src_dir, src_dir / 'data', src_dir / 'models']:
#     if str(path) not in sys.path:
#         sys.path.insert(0, str(path))

# import math

# def _is_norm_or_bias(name: str) -> bool:
#     n = name.lower()
#     return ('.bias' in n) or ('norm' in n) or ('bn' in n) or ('layernorm' in n) or ('ln' in n)

# def make_param_groups(model, enc_lr, head_lr, enc_wd, head_wd):
#     enc_decay, enc_nodc, head_decay, head_nodc = [], [], [], []
#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         is_enc = n.startswith('encoder_1d') or n.startswith('encoder_2d')
#         no_decay = _is_norm_or_bias(n)
#         if is_enc:
#             (enc_nodc if no_decay else enc_decay).append(p)
#         else:
#             (head_nodc if no_decay else head_decay).append(p)

#     return [
#         {'params': enc_decay,  'lr': enc_lr,  'weight_decay': enc_wd},
#         {'params': enc_nodc,   'lr': enc_lr,  'weight_decay': 0.0},
#         {'params': head_decay, 'lr': head_lr, 'weight_decay': head_wd},
#         {'params': head_nodc,  'lr': head_lr, 'weight_decay': 0.0},
#     ]

# def make_warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
#     def lr_lambda(epoch_idx: int):
#         # linear warm-up to 1.0
#         if epoch_idx < warmup_epochs:
#             return float(epoch_idx + 1) / float(max(1, warmup_epochs))
#         # cosine decay from 1.0 -> 0.0
#         progress = (epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
#         return 0.5 * (1.0 + math.cos(math.pi * progress))
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# # ==========================
# #  Model Components
# # ==========================
# class CrossAttentionFusion(nn.Module):
#     """Cross-attention module to fuse 1D and 2D features"""
#     def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.dim = dim
#         self.head_dim = dim // num_heads
#         self.q_proj = nn.Linear(dim, dim)
#         self.k_proj = nn.Linear(dim, dim)
#         self.v_proj = nn.Linear(dim, dim)
#         self.out_proj = nn.Linear(dim, dim)
#         self.dropout = nn.Dropout(dropout)
#         self.scale = self.head_dim ** -0.5

#     def forward(self, query: torch.Tensor, key_value: torch.Tensor):
#         B = query.shape[0]
#         if query.dim() == 2: query = query.unsqueeze(1)
#         if key_value.dim() == 2: key_value = key_value.unsqueeze(1)

#         Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         K = self.k_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
#         V = self.v_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

#         attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
#         attn = F.softmax(attn, dim=-1)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, V)
#         out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
#         out = self.out_proj(out).squeeze(1)
#         return out


# class EncoderWithoutHead(nn.Module):
#     """Extract CLS token features directly from transformer (bypass classifier)"""
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
#         if not (hasattr(encoder, 'patch_embed') and hasattr(encoder, 'blocks') and hasattr(encoder, 'norm')):
#             raise ValueError("Encoder must have patch_embed, blocks, and norm attributes")
#         print("  ✓ Wrapped encoder (bypasses classification head)")

#     def forward(self, x):
#         x = self.encoder.patch_embed(x)
#         for block in self.encoder.blocks:
#             x = block(x)
#         x = self.encoder.norm(x)
#         return x[:, 0]  # [B, 384]


# class ECGFusionModel(nn.Module):
#     """Aligned encoders + bidirectional cross-attention + classifier"""
#     def __init__(
#         self,
#         encoder_1d: nn.Module,
#         encoder_2d: nn.Module,
#         feature_dim: int = 384,
#         num_classes: int = 4,
#         num_heads: int = 8,
#         dropout: float = 0.1,
#         freeze_encoders: bool = True
#     ):
#         super().__init__()
#         self.encoder_1d = EncoderWithoutHead(encoder_1d)
#         self.encoder_2d = EncoderWithoutHead(encoder_2d)
#         self.freeze_encoders = freeze_encoders

#         if freeze_encoders:
#             for p in self.encoder_1d.parameters(): p.requires_grad = False
#             for p in self.encoder_2d.parameters(): p.requires_grad = False

#         self.cross_attn_1d_to_2d = CrossAttentionFusion(feature_dim, num_heads, dropout)
#         self.cross_attn_2d_to_1d = CrossAttentionFusion(feature_dim, num_heads, dropout)
#         self.norm = nn.LayerNorm(feature_dim)

#         fusion_dim = feature_dim * 2
#         self.classifier = nn.Sequential(
#             nn.Linear(fusion_dim, 512),
#             nn.LayerNorm(512),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes)
#         )

#         print("\n[ECG Fusion Model] Created")
#         print(f"  Feature dim: {feature_dim}")
#         print(f"  Fusion dim: {fusion_dim}")
#         print(f"  Num classes: {num_classes}")
#         print(f"  Num heads: {num_heads}")
#         print(f"  Encoders frozen: {freeze_encoders}")

#     def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
#         with torch.set_grad_enabled(not self.freeze_encoders):
#             f1 = self.encoder_1d(x_1d)  # [B, 384]
#             f2 = self.encoder_2d(x_2d)  # [B, 384]

#         assert f1.shape[1] == 384 and f2.shape[1] == 384, "Expected 384D embeddings"

#         f1_att = self.cross_attn_1d_to_2d(f1, f2)
#         f2_att = self.cross_attn_2d_to_1d(f2, f1)

#         f1_fused = self.norm(f1 + f1_att)
#         f2_fused = self.norm(f2 + f2_att)

#         fused = torch.cat([f1_fused, f2_fused], dim=1)  # [B, 768]
#         logits = self.classifier(fused)
#         return logits

# # ==========================
# #  Training Curves
# # ==========================
# def _smooth_series(y, sg_window=9, sg_poly=2, ma_window=0):
#     """Savitzky–Golay + (optional) moving average with edge padding."""
#     y = np.asarray(y, dtype=float)
#     n = len(y)
#     # Savitzky–Golay (use odd window and clamp to length)
#     if n >= 3:
#         win = min(sg_window if sg_window % 2 == 1 else sg_window + 1, n if n % 2 == 1 else n - 1)
#         win = max(3, win)  # at least 3, odd
#         poly = min(sg_poly, max(1, win - 2))
#         y = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
#     # Moving average (optional), with edge padding (no zero-pad drop)
#     if ma_window and ma_window > 1 and n > 1:
#         k = min(ma_window, n if n % 2 == 1 else n - 1)
#         pad = k // 2
#         ypad = np.pad(y, (pad, pad), mode="edge")
#         kernel = np.ones(k, dtype=float) / k
#         y = np.convolve(ypad, kernel, mode="valid")
#     return y

# def _tight_ylim(values, pad_rel=0.15, abs_min_pad=0.1, is_accuracy=False):
#     v = np.asarray(values, dtype=float)
#     lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
#     if is_accuracy:
#         # accuracy tends to be compressed near the top
#         pad = max(abs_min_pad, (hi - lo) * pad_rel)
#     else:
#         pad = max(0.01, (hi - lo) * 0.10)
#     if hi - lo < 1e-6:  # degenerate case
#         return lo - pad, hi + pad
#     return lo - pad, hi + pad

# def plot_history(history, out_dir: Path, sg_window=9, sg_poly=2, ma_window=0):
#     """
#     Create smoothed paper-ready plots:
#       - loss_clean.png
#       - accuracy_clean.png
#       - f1_clean.png (if val_f1 present)
#     Avoids end-point drop and uses tight y-limits.
#     """
#     out_dir.mkdir(parents=True, exist_ok=True)

#     train_loss = history.get('train_loss', [])
#     val_loss   = history.get('val_loss', [])
#     train_acc  = history.get('train_acc', [])
#     val_acc    = history.get('val_acc', [])
#     val_f1     = history.get('val_f1', [])

#     # Use common length across curves to avoid tail mismatch artifacts
#     n = min(len(train_loss), len(val_loss), len(train_acc), len(val_acc))
#     if n == 0:
#         print("[plot_history] Nothing to plot (history empty).")
#         return
#     train_loss, val_loss = train_loss[:n], val_loss[:n]
#     train_acc,  val_acc  = train_acc[:n],  val_acc[:n]
#     val_f1               = val_f1[:n] if len(val_f1) >= n else []

#     epochs = np.arange(n)

#     # Smooth
#     s_train_loss = _smooth_series(train_loss, sg_window, sg_poly, ma_window)
#     s_val_loss   = _smooth_series(val_loss,   sg_window, sg_poly, ma_window)
#     s_train_acc  = _smooth_series(train_acc,  sg_window, sg_poly, ma_window)
#     s_val_acc    = _smooth_series(val_acc,    sg_window, sg_poly, ma_window)
#     s_val_f1     = _smooth_series(val_f1,     sg_window, sg_poly, ma_window) if val_f1 else None

#     # Best validation accuracy epoch
#     best_idx = int(np.nanargmax(val_acc))
#     best_ep  = epochs[best_idx]

#     # ---- Loss
#     plt.figure(figsize=(10, 7))
#     plt.plot(epochs, s_train_loss, label='train_loss')
#     plt.plot(epochs, s_val_loss,   label='val_loss')
#     plt.scatter([best_ep], [s_val_loss[best_idx]], s=60, marker='*', zorder=5, label='best val acc epoch')
#     lo, hi = _tight_ylim(np.r_[s_train_loss, s_val_loss], pad_rel=0.12)
#     plt.ylim(lo, hi)
#     plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss')
#     plt.legend(); plt.tight_layout()
#     plt.savefig(out_dir / 'loss_clean.png', dpi=300); plt.close()

#     # ---- Accuracy
#     plt.figure(figsize=(10, 7))
#     plt.plot(epochs, s_train_acc, label='train_acc (smoothed)')
#     plt.plot(epochs, s_val_acc,   label='val_acc (smoothed)')
#     plt.scatter([best_ep], [s_val_acc[best_idx]], s=60, marker='*', zorder=5, label='best val acc epoch')
#     lo, hi = _tight_ylim(np.r_[s_train_acc, s_val_acc], pad_rel=0.12, is_accuracy=True)
#     plt.ylim(lo, hi)
#     plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Accuracy (Smoothed)')
#     plt.legend(); plt.tight_layout()
#     plt.savefig(out_dir / 'accuracy_clean.png', dpi=300); plt.close()

#     # ---- F1 (optional)
#     if s_val_f1 is not None and len(val_f1) > 0:
#         plt.figure(figsize=(10, 7))
#         plt.plot(epochs, s_val_f1, label='val_f1 (smoothed)')
#         lo, hi = _tight_ylim(s_val_f1, pad_rel=0.12)
#         plt.ylim(lo, hi)
#         plt.xlabel('Epoch'); plt.ylabel('Weighted F1'); plt.title('F1 (Smoothed)')
#         plt.legend(); plt.tight_layout()
#         plt.savefig(out_dir / 'f1_clean.png', dpi=300); plt.close()



# # ==========================
# #  Saliency utils
# # ==========================
# @torch.no_grad()
# def _softmax_probs(logits): return F.softmax(logits, dim=1)

# def _minmax_torch(x: torch.Tensor, eps: float = 1e-8):
#     x = x - x.min()
#     return x / (x.max() + eps)

# def compute_saliency(model: ECGFusionModel, x1: torch.Tensor, x2: torch.Tensor, target_idx: int, device):
#     model.eval()
#     x1 = x1.clone().detach().requires_grad_(True).to(device)
#     x2 = x2.clone().detach().requires_grad_(True).to(device)

#     logits = model(x1, x2)
#     target_logit = logits[0, target_idx]
#     model.zero_grad(set_to_none=True)
#     target_logit.backward()

#     s1 = x1.grad.abs().detach()[0].mean(dim=0)        # [T]
#     s2 = x2.grad.abs().detach()[0].mean(dim=0)        # [H, W]
#     return s1, s2, logits.detach().cpu()

# def save_saliency_samples(
#     model: ECGFusionModel,
#     loader: DataLoader,
#     device: torch.device,
#     out_dir: Path,
#     max_samples: int = 12,
#     smooth_sigma: float = 3.0,
#     saliency_threshold: float = 0.7,
#     compare_classes=(1, 2),
#     class_names=None
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     print(f"[Saliency] Saving {max_samples} samples to {out_dir}")
#     if class_names is None:
#         class_names = {0: 'SB', 1: 'SR', 2: 'AFIB', 3: 'GSVT'}

#     collected_1d, collected_2d, metas = [], [], []
#     model.eval()
#     saved = 0
#     with torch.enable_grad():
#         for x_1d, x_2d, y in loader:
#             for i in range(x_1d.size(0)):
#                 xi1 = x_1d[i:i+1].to(device)
#                 xi2 = x_2d[i:i+1].to(device)
#                 yi = int(y[i].item())

#                 logits = model(xi1, xi2)
#                 pred = int(logits.softmax(1).argmax(1).item())
#                 s1, s2, _ = compute_saliency(model, xi1, xi2, pred, device)

#                 s1_np = s1.cpu().numpy()
#                 if smooth_sigma > 0: s1_np = gaussian_filter1d(s1_np, sigma=smooth_sigma)
#                 s1 = torch.from_numpy(s1_np)

#                 collected_1d.append(s1)
#                 collected_2d.append(s2.detach().cpu())
#                 metas.append((saved, yi, pred, x_1d[i:i+1].cpu(), x_2d[i:i+1].cpu()))
#                 saved += 1
#                 if saved >= max_samples: break
#             if saved >= max_samples: break

#     # Normalize across samples
#     maxlen = max(s.numel() for s in collected_1d)
#     pad_1d = []
#     for s in collected_1d:
#         if s.numel() < maxlen:
#             pad = torch.zeros(maxlen); pad[:s.numel()] = s; pad_1d.append(pad)
#         else:
#             pad_1d.append(s)
#     S1 = _minmax_torch(torch.stack(pad_1d, dim=0))
#     collected_1d = [S1[i, :metas[i][3].shape[-1]] for i in range(len(metas))]

#     H = max(s.shape[0] for s in collected_2d); W = max(s.shape[1] for s in collected_2d)
#     pad_2d = []
#     for s in collected_2d:
#         canvas = torch.zeros((H, W)); h, w = s.shape; canvas[:h, :w] = s; pad_2d.append(canvas)
#     S2 = _minmax_torch(torch.stack(pad_2d, dim=0))
#     collected_2d = [S2[i, :collected_2d[i].shape[0], :collected_2d[i].shape[1]] for i in range(len(metas))]

#     # Save per-sample overlays
#     for i, (_, yi, pred, xi1_cpu, xi2_cpu) in enumerate(metas):
#         s1 = collected_1d[i]; s2 = collected_2d[i]
#         sig = xi1_cpu[0, 0].numpy(); spec = xi2_cpu[0, 0].numpy()
#         thr = float(saliency_threshold); mask = (s1.numpy() >= thr).astype(float)

#         # 1D shaded
#         plt.figure(figsize=(12, 3))
#         plt.plot(sig, color='black', linewidth=1)
#         on=False; start=0
#         for t in range(len(mask)):
#             if mask[t] and not on: on=True; start=t
#             if (not mask[t] and on) or (on and t==len(mask)-1):
#                 end=t if not mask[t] else t+1
#                 plt.fill_between(np.arange(start, end), sig[start:end], 0, color='red', alpha=0.3)
#                 on=False
#         plt.title(f"1D ECG with Saliency (true={class_names.get(yi, yi)}, pred={class_names.get(pred, pred)})")
#         plt.xlabel("Time (samples)"); plt.ylabel("Amplitude")
#         plt.tight_layout(); plt.savefig(out_dir / f"saliency1d_{i:03d}_true{yi}_pred{pred}.png", dpi=300); plt.close()

#         # 2D overlay
#         plt.figure(figsize=(6, 5))
#         plt.imshow(spec, aspect='auto', origin='lower', cmap='gray')
#         plt.imshow(s2.numpy(), aspect='auto', origin='lower', cmap='jet', alpha=0.5)
#         plt.title(f"2D STFT + Saliency (true={class_names.get(yi, yi)}, pred={class_names.get(pred, pred)})")
#         plt.axis("off"); plt.tight_layout()
#         plt.savefig(out_dir / f"saliency2d_{i:03d}_true{yi}_pred{pred}.png", dpi=300); plt.close()

# # ==========================
# #  Evaluation
# # ==========================
# def evaluate_model(model, test_loader, device: torch.device, output_dir: Path):
#     from sklearn.metrics import (
#         accuracy_score, precision_score, recall_score,
#         f1_score, roc_auc_score, confusion_matrix,
#         precision_recall_fscore_support
#     )

#     print("\n" + "="*70)
#     print("EVALUATING ON TEST SET")
#     print("="*70)

#     model.eval()
#     all_preds, all_labels, all_probs = [], [], []
#     with torch.no_grad():
#         for x_1d, x_2d, labels in tqdm(test_loader, desc="Test Evaluation"):
#             x_1d, x_2d = x_1d.to(device), x_2d.to(device)
#             logits = model(x_1d, x_2d)
#             probs = F.softmax(logits, dim=1)
#             pred = probs.argmax(dim=1)
#             all_preds.extend(pred.cpu().numpy())
#             all_labels.extend(labels.numpy())
#             all_probs.extend(probs.cpu().numpy())

#     all_preds = np.array(all_preds); all_labels = np.array(all_labels); all_probs = np.array(all_probs)
#     acc = accuracy_score(all_labels, all_preds)
#     prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
#     rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

#     cm = confusion_matrix(all_labels, all_preds)
#     specificity_per_class = []
#     for i in range(cm.shape[0]):
#         tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
#         fp = cm[:,i].sum() - cm[i,i]
#         spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#         specificity_per_class.append(spec)
#     specificity = np.average(specificity_per_class, weights=cm.sum(axis=1))

#     try:
#         auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
#         auc_per_class = []
#         for i in range(all_probs.shape[1]):
#             binary = (all_labels == i).astype(int)
#             auc_i = roc_auc_score(binary, all_probs[:, i]) if len(np.unique(binary)) > 1 else 0.0
#             auc_per_class.append(auc_i)
#     except Exception:
#         auc, auc_per_class = 0.0, [0.0]*all_probs.shape[1]

#     precision_per_class, recall_per_class, f1_per_class, _ = \
#         precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

#     results = {
#         "accuracy": float(acc),
#         "precision": float(prec),
#         "sensitivity": float(rec),
#         "specificity": float(specificity),
#         "recall": float(rec),
#         "f1": float(f1),
#         "auc": float(auc),
#         "auc_per_class": [float(x) for x in auc_per_class],
#         "precision_per_class": [float(x) for x in precision_per_class],
#         "recall_per_class": [float(x) for x in recall_per_class],
#         "sensitivity_per_class": [float(x) for x in recall_per_class],
#         "specificity_per_class": [float(x) for x in specificity_per_class],
#         "f1_per_class": [float(x) for x in f1_per_class],
#         "confusion_matrix": cm.tolist()
#     }

#     with open(output_dir / 'test_results.json', 'w') as f:
#         json.dump(results, f, indent=2)

#     print("\n" + "="*70)
#     print("TEST RESULTS")
#     print("="*70)
#     print(f"Accuracy:  {acc*100:.2f}%")
#     print(f"Precision: {prec:.4f}")
#     print(f"F1:        {f1:.4f}")
#     print(f"AUC:       {auc:.4f}")
#     print("Confusion Matrix:\n", cm)
#     print("="*70)
#     print(f"Results saved to: {output_dir / 'test_results.json'}")
#     print("="*70)
#     return results

# # ==========================
# #  Training (STABLE)
# # ==========================
# def train_fusion_model(
#     model: ECGFusionModel,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     test_loader: DataLoader,
#     num_epochs: int,
#     learning_rate: float,             # kept for backwards compat but not used for groups
#     device: torch.device,
#     output_dir: Path,
#     patience: int = 10,
#     label_smoothing: float = 0.05,
#     grad_accum_steps: int = 1,
#     # === NEW: differential LRs & WD + warmup
#     encoder_lr: float = 3e-5,
#     head_lr: float = 1e-4,
#     encoder_wd: float = 0.01,
#     head_wd: float = 0.01,
#     lr_warmup_epochs: int = 3,
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # 1) Label smoothing
#     criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

#     # Trainable params
#     trainable = [p for p in model.parameters() if p.requires_grad]
#     print(f"\nTrainable parameters: {sum(p.numel() for p in trainable):,}")

#     # === NEW: build param groups (encoders vs fusion+classifier)
#     param_groups = make_param_groups(
#         model,
#         enc_lr=encoder_lr,
#         head_lr=head_lr,
#         enc_wd=encoder_wd,
#         head_wd=head_wd,
#     )
#     optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

#     # === NEW: linear warmup → cosine (step ONCE per epoch)
#     scheduler = make_warmup_cosine_scheduler(
#         optimizer,
#         warmup_epochs=lr_warmup_epochs,
#         total_epochs=num_epochs,
#     )

#     print("\nParameter groups:")
#     for i, g in enumerate(optimizer.param_groups):
#         n_params = sum(p.numel() for p in g['params'])
#         print(f"  group {i}: lr={g['lr']:.2e}, wd={g['weight_decay']}, #params={n_params:,}")

#     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
#     best_val_acc = 0.0
#     epochs_no_improve = 0

#     print("\n" + "="*70)
#     print("PHASE 2: FUSION MODEL TRAINING (STABLE)")
#     print("="*70)

#     for epoch in range(num_epochs):
#         model.train()
#         tr_loss, tr_correct, tr_total = 0.0, 0, 0
#         optimizer.zero_grad(set_to_none=True)

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
#         for bi, (x_1d, x_2d, labels) in enumerate(pbar):
#             x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)
#             logits = model(x_1d, x_2d)
#             loss = criterion(logits, labels) / grad_accum_steps
#             loss.backward()

#             # grad clip
#             nn.utils.clip_grad_norm_(trainable, max_norm=1.0)

#             # Gradient accumulation
#             if (bi + 1) % grad_accum_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)

#             tr_loss += loss.item() * grad_accum_steps
#             tr_correct += (logits.argmax(1) == labels).sum().item()
#             tr_total += labels.size(0)

#             pbar.set_postfix({
#                 'loss': f'{tr_loss/(bi+1):.4f}',
#                 'acc': f'{100.*tr_correct/tr_total:.2f}%'
#             })

#         # === NEW: step the warmup+cosine scheduler ONCE per epoch
#         scheduler.step()

#         train_loss = tr_loss / len(train_loader)
#         train_acc = 100.0 * tr_correct / tr_total

#         # Validation
#         model.eval()
#         val_loss, val_correct, val_total = 0.0, 0, 0
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for x_1d, x_2d, labels in tqdm(val_loader, desc="Validation"):
#                 x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)
#                 logits = model(x_1d, x_2d)
#                 loss = criterion(logits, labels)
#                 val_loss += loss.item()
#                 pred = logits.argmax(1)
#                 val_correct += (pred == labels).sum().item()
#                 val_total += labels.size(0)
#                 all_preds.extend(pred.cpu().numpy()); all_labels.extend(labels.cpu().numpy())

#         val_loss /= len(val_loader)
#         val_acc = 100.0 * val_correct / val_total
#         from sklearn.metrics import f1_score
#         val_f1 = f1_score(all_labels, all_preds, average='weighted')

#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)
#         history['val_f1'].append(val_f1)

#         print("="*70)
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print(f"  LR (enc/head): {optimizer.param_groups[0]['lr']:.2e} / {optimizer.param_groups[2]['lr']:.2e}")
#         print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
#         print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
#         print(f"  Val   F1:   {val_f1:.4f}")

#         # Checkpoints
#         ckpt = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'val_acc': val_acc,
#             'history': history
#         }
#         torch.save(ckpt, output_dir / f'checkpoint_epoch_{epoch}.pth')

#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             epochs_no_improve = 0
#             torch.save(ckpt, output_dir / 'best_fusion_model.pth')
#             print(f"  💾 Saved best model (Val Acc: {val_acc:.2f}%)")
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print(f"\n⚠️  Early stopping at epoch {epoch+1}")
#             break

#     with open(output_dir / 'training_history.json', 'w') as f:
#         json.dump(history, f, indent=2)

#     plot_history(history, output_dir)
#     print(f"\n✓ Training Complete! Best Val Acc: {best_val_acc:.2f}%")

#     # Evaluate best on test
#     if test_loader is not None:
#         best = torch.load(output_dir / 'best_fusion_model.pth', map_location=device)
#         model.load_state_dict(best['model_state_dict'])
#         evaluate_model(model, test_loader, device, output_dir)

#     return model

# # ==========================
# #  Main
# # ==========================
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--aligned-encoder-1d', required=True)
#     parser.add_argument('--aligned-encoder-2d', required=True)
#     parser.add_argument('--data-dir', required=True)
#     parser.add_argument('--representation-type', default='stft')
#     parser.add_argument('--batch-size', type=int, default=32)
#     parser.add_argument('--num-workers', type=int, default=4)
#     parser.add_argument('--feature-dim', type=int, default=384)
#     parser.add_argument('--num-classes', type=int, default=4)
#     parser.add_argument('--num-heads', type=int, default=8)
#     parser.add_argument('--dropout', type=float, default=0.1)

#     # freeze / unfreeze (still present)
#     parser.add_argument('--unfreeze-encoders', action='store_true')

#     parser.add_argument('--num-epochs', type=int, default=50)
#     parser.add_argument('--learning-rate', type=float, default=1e-4)
#     parser.add_argument('--patience', type=int, default=15)
#     parser.add_argument('--output-dir', required=True)
#     parser.add_argument('--device', default='cuda')

#     # stability knobs (fair)
#     parser.add_argument('--label-smoothing', type=float, default=0.05)
#     parser.add_argument('--grad-accum-steps', type=int, default=1)
#     parser.add_argument('--encoder-lr', type=float, default=3e-5,
#                     help='LR for encoder weights (1D & 2D)')
#     parser.add_argument('--head-lr', type=float, default=1e-4,
#                         help='LR for fusion + classifier')
#     parser.add_argument('--encoder-wd', type=float, default=0.01,
#                         help='Weight decay for encoders')
#     parser.add_argument('--head-wd', type=float, default=0.01,
#                         help='Weight decay for head')
#     parser.add_argument('--lr-warmup-epochs', type=int, default=3,
#                         help='Linear warmup epochs before cosine anneal')


#     # saliency options
#     parser.add_argument('--save-saliency', action='store_true')
#     parser.add_argument('--saliency-samples', type=int, default=12)
#     parser.add_argument('--saliency-threshold', type=float, default=0.7)
#     parser.add_argument('--smooth-sigma', type=float, default=3.0)
#     parser.add_argument('--save-saliency-gallery', action='store_true',
#                     help='Save curated saliency gallery (one panel per class + 2x2 sheet)')
#     parser.add_argument('--gallery-samples-per-class', type=int, default=1)
#     parser.add_argument('--gallery-threshold', type=float, default=0.7)
#     parser.add_argument('--gallery-sigma', type=int, default=7)


#     args = parser.parse_args()
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

#     print("="*70); print("PHASE 2: CROSS-ATTENTION FUSION (STABLE)"); print("="*70)
    

#     # data
#     print("\n[Setting up Fusion Data Module]")
#     try:
#         from src.data.fusion_dataset import ECGFusionDataModule
#     except ImportError:
#         from fusion_dataset import ECGFusionDataModule

#     data_module = ECGFusionDataModule(
#         data_dir=args.data_dir,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         representation_type=args.representation_type
#     )
#     data_module.setup()
#     print("  ✓ Data module ready")

#     # encoders
#     print("\n[Loading Encoders]")
#     enc1_state = torch.load(Path(args.aligned_encoder_1d), map_location=device)
#     enc2_state = torch.load(Path(args.aligned_encoder_2d), map_location=device)

#     try:
#         from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
#         from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
#     except ImportError:
#         from ecg_transformer_1d import create_ecg_transformer_1d_small
#         from models.ecg_transformer_2d import create_ecg_transformer_2d_small

#     encoder_1d = create_ecg_transformer_1d_small()
#     encoder_2d = create_ecg_transformer_2d_small(representation_type=args.representation_type)

#     def _extract_state_dict(ckpt):
#         if isinstance(ckpt, dict):
#             if 'model_state_dict' in ckpt: return ckpt['model_state_dict']
#             if 'state_dict' in ckpt: return ckpt['state_dict']
#             return ckpt
#         raise RuntimeError('Unrecognized checkpoint format')

#     encoder_1d.load_state_dict(_extract_state_dict(enc1_state), strict=False)
#     encoder_2d.load_state_dict(_extract_state_dict(enc2_state), strict=False)
#     encoder_1d, encoder_2d = encoder_1d.to(device), encoder_2d.to(device)
#     print("  ✓ Encoders loaded")

#     freeze = not args.unfreeze_encoders
#     model = ECGFusionModel(
#         encoder_1d=encoder_1d,
#         encoder_2d=encoder_2d,
#         feature_dim=args.feature_dim,
#         num_classes=args.num_classes,
#         num_heads=args.num_heads,
#         dropout=args.dropout,
#         freeze_encoders=freeze
#     ).to(device)

#     # train
#     train_fusion_model(
#         model=model,
#         train_loader=data_module.train_loader,
#         val_loader=data_module.val_loader,
#         test_loader=data_module.test_loader,
#         num_epochs=args.num_epochs,
#         learning_rate=args.learning_rate,      # kept but not used for groups
#         device=device,
#         output_dir=output_dir,
#         patience=args.patience,
#         label_smoothing=args.label_smoothing,
#         grad_accum_steps=args.grad_accum_steps,
#         # NEW:
#         encoder_lr=args.encoder_lr,
#         head_lr=args.head_lr,
#         encoder_wd=args.encoder_wd,
#         head_wd=args.head_wd,
#         lr_warmup_epochs=args.lr_warmup_epochs,
#     )


#     # saliency / overlays (optional)
#     if args.save_saliency_gallery:
#         gal_dir = output_dir / "saliency_gallery"
#         try:
#             save_saliency_samples(
#                 model=model,
#                 loader=data_module.test_loader,
#                 device=device,
#                 out_dir=gal_dir,
#                 class_names={0:'SB',1:'SR',2:'AFIB',3:'GSVT'},
#                 samples_per_class=args.gallery_samples_per_class,
#                 sigma=args.gallery_sigma,
#                 thr=args.gallery_threshold,
#                 representation_type=args.representation_type
#             )
#         except Exception as e:
#             print("[Warning] Failed to save saliency gallery:", repr(e))


#     print("\n✓ Phase 2 Complete!")

# if __name__ == "__main__":
#     main()

# # #!/usr/bin/env python3
# # """
# # Phase 2: Cross-Attention Fusion for ECG Classification
# # with Grad-based Saliency (1D & 2D), Gaussian smoothing, overlays,
# # across-sample normalization, and curated side-by-side class gallery.
# # """

# # import argparse
# # from pathlib import Path
# # import sys
# # import json
# # import numpy as np
# # from tqdm import tqdm

# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader

# # # plotting
# # import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt

# # # smoothing for 1D saliency
# # from scipy.ndimage import gaussian_filter1d

# # # --------------------------
# # # Project paths
# # # --------------------------
# # project_root = Path(__file__).resolve().parent.parent
# # src_dir = project_root / 'src'
# # for path in [project_root, src_dir, src_dir / 'data', src_dir / 'models']:
# #     if str(path) not in sys.path:
# #         sys.path.insert(0, str(path))


# # # --------------------------
# # # Utility helpers
# # # --------------------------
# # def np_minmax(x, eps=1e-12):
# #     x = x - np.min(x)
# #     den = np.max(x)
# #     if not np.isfinite(den) or den < eps:
# #         den = 1.0
# #     return x / den

# # def torch_minmax(x: torch.Tensor, eps: float = 1e-8):
# #     x = x - x.amin()
# #     d = x.amax().clamp_min(eps)
# #     return x / d

# # def interp1d_to_length(arr: np.ndarray, target_len: int) -> np.ndarray:
# #     """Linear resample 1D array to target_len."""
# #     if arr.ndim != 1:
# #         arr = arr.reshape(-1)
# #     if arr.size == target_len:
# #         return arr
# #     x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=True)
# #     x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
# #     return np.interp(x_new, x_old, arr)

# # def resize_2d_to(shape_src: np.ndarray, target_hw: tuple) -> np.ndarray:
# #     """Resize [H,W] saliency map to (H_t,W_t) using bilinear via torch."""
# #     h, w = shape_src.shape
# #     th, tw = target_hw
# #     if (h, w) == (th, tw):
# #         return shape_src
# #     t = torch.from_numpy(shape_src).float()[None, None, :, :]  # 1x1xH xW
# #     t2 = F.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)
# #     return t2[0, 0].cpu().numpy()


# # # --------------------------
# # # Modules
# # # --------------------------
# # class CrossAttentionFusion(nn.Module):
# #     """Cross-attention module to fuse 1D and 2D features"""
# #     def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
# #         super().__init__()
# #         assert dim % num_heads == 0
# #         self.num_heads = num_heads
# #         self.dim = dim
# #         self.head_dim = dim // num_heads

# #         self.q_proj = nn.Linear(dim, dim)
# #         self.k_proj = nn.Linear(dim, dim)
# #         self.v_proj = nn.Linear(dim, dim)
# #         self.out_proj = nn.Linear(dim, dim)
# #         self.dropout = nn.Dropout(dropout)
# #         self.scale = self.head_dim ** -0.5

# #     def forward(self, query: torch.Tensor, key_value: torch.Tensor):
# #         B = query.shape[0]
# #         if query.dim() == 2:
# #             query = query.unsqueeze(1)
# #         if key_value.dim() == 2:
# #             key_value = key_value.unsqueeze(1)

# #         Q = self.q_proj(query).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
# #         K = self.k_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
# #         V = self.v_proj(key_value).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

# #         attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
# #         attn = F.softmax(attn, dim=-1)
# #         attn = self.dropout(attn)

# #         out = torch.matmul(attn, V)
# #         out = out.transpose(1, 2).contiguous().view(B, -1, self.dim)
# #         out = self.out_proj(out).squeeze(1)
# #         return out


# # class EncoderWithoutHead(nn.Module):
# #     """Extract CLS token features directly from transformer (bypass classifier)"""
# #     def __init__(self, encoder):
# #         super().__init__()
# #         self.encoder = encoder
# #         if not (hasattr(encoder, 'patch_embed') and hasattr(encoder, 'blocks') and hasattr(encoder, 'norm')):
# #             raise ValueError("Encoder must have patch_embed, blocks, and norm attributes")
# #         print("  ✓ Wrapped encoder (bypasses classification head)")

# #     def forward(self, x):
# #         x = self.encoder.patch_embed(x)
# #         for block in self.encoder.blocks:
# #             x = block(x)
# #         x = self.encoder.norm(x)
# #         return x[:, 0]  # [B, 384]


# # class ECGFusionModel(nn.Module):
# #     """Aligned encoders + bidirectional cross-attention + classifier"""
# #     def __init__(
# #         self,
# #         encoder_1d: nn.Module,
# #         encoder_2d: nn.Module,
# #         feature_dim: int = 384,
# #         num_classes: int = 4,
# #         num_heads: int = 8,
# #         dropout: float = 0.1,
# #         freeze_encoders: bool = True
# #     ):
# #         super().__init__()
# #         self.encoder_1d = EncoderWithoutHead(encoder_1d)
# #         self.encoder_2d = EncoderWithoutHead(encoder_2d)
# #         self.freeze_encoders = freeze_encoders

# #         if freeze_encoders:
# #             for p in self.encoder_1d.parameters():
# #                 p.requires_grad = False
# #             for p in self.encoder_2d.parameters():
# #                 p.requires_grad = False

# #         self.cross_attn_1d_to_2d = CrossAttentionFusion(feature_dim, num_heads, dropout)
# #         self.cross_attn_2d_to_1d = CrossAttentionFusion(feature_dim, num_heads, dropout)
# #         self.norm = nn.LayerNorm(feature_dim)

# #         fusion_dim = feature_dim * 2
# #         self.classifier = nn.Sequential(
# #             nn.Linear(fusion_dim, 512),
# #             nn.LayerNorm(512),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(512, 256),
# #             nn.LayerNorm(256),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(256, num_classes)
# #         )

# #         print("\n[ECG Fusion Model] Created")
# #         print(f"  Feature dim: {feature_dim}")
# #         print(f"  Fusion dim: {fusion_dim}")
# #         print(f"  Num classes: {num_classes}")
# #         print(f"  Num heads: {num_heads}")
# #         print(f"  Encoders frozen: {freeze_encoders}")

# #     def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
# #         with torch.set_grad_enabled(not self.freeze_encoders):
# #             f1 = self.encoder_1d(x_1d)  # [B, 384]
# #             f2 = self.encoder_2d(x_2d)  # [B, 384]

# #         assert f1.shape[1] == 384 and f2.shape[1] == 384, "Expected 384D embeddings"

# #         f1_att = self.cross_attn_1d_to_2d(f1, f2)
# #         f2_att = self.cross_attn_2d_to_1d(f2, f1)

# #         f1_fused = self.norm(f1 + f1_att)
# #         f2_fused = self.norm(f2 + f2_att)

# #         fused = torch.cat([f1_fused, f2_fused], dim=1)  # [B, 768]
# #         logits = self.classifier(fused)
# #         return logits


# # # --------------------------
# # # Metrics / plots
# # # --------------------------
# # def plot_history(history, out_dir: Path):
# #     # Loss
# #     plt.figure(figsize=(10, 7))
# #     plt.plot(history['train_loss'], label='train_loss')
# #     plt.plot(history['val_loss'], label='val_loss')
# #     plt.title('Loss')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Loss')
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(out_dir / 'loss.png', dpi=300)
# #     plt.close()

# #     # Accuracy
# #     plt.figure(figsize=(10, 7))
# #     plt.plot(history['train_acc'], label='train_acc')
# #     plt.plot(history['val_acc'], label='val_acc')
# #     plt.title('Accuracy')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Accuracy (%)')
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(out_dir / 'accuracy.png', dpi=300)
# #     plt.close()

# #     # F1
# #     plt.figure(figsize=(10, 7))
# #     plt.plot(history['val_f1'], label='val_f1')
# #     plt.title('F1')
# #     plt.xlabel('Epoch')
# #     plt.ylabel('Weighted F1')
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(out_dir / 'f1.png', dpi=300)
# #     plt.close()


# # # --------------------------
# # # Saliency helpers
# # # --------------------------
# # def compute_saliency(model: ECGFusionModel, x1: torch.Tensor, x2: torch.Tensor, target_idx: int, device):
# #     """
# #     Gradients of target logit w.r.t. inputs.
# #     Returns saliency_1d [T], saliency_2d [H, W]
# #     """
# #     model.eval()
# #     x1 = x1.clone().detach().requires_grad_(True).to(device)  # [1, C, T] (expected)
# #     x2 = x2.clone().detach().requires_grad_(True).to(device)  # [1, C, H, W]

# #     logits = model(x1, x2)              # [1, num_classes]
# #     target_logit = logits[0, target_idx]
# #     model.zero_grad(set_to_none=True)
# #     target_logit.backward()

# #     # defensive aggregation over channels/time
# #     g1 = x1.grad.detach().abs()[0]                  # [C, T] or [T] if C==1
# #     if g1.ndim == 1:
# #         s1 = g1
# #     else:
# #         # max over channel -> [T]
# #         s1 = g1.max(dim=0)[0]

# #     g2 = x2.grad.detach().abs()[0]                  # [C, H, W] or [H, W] if C==1
# #     if g2.ndim == 2:
# #         s2 = g2
# #     else:
# #         s2 = g2.max(dim=0)[0]                       # [H, W]

# #     return s1.cpu(), s2.cpu(), logits.detach().cpu()


# # def save_saliency_samples(
# #     model: ECGFusionModel,
# #     loader: DataLoader,
# #     device: torch.device,
# #     out_dir: Path,
# #     max_samples: int = 12,
# #     smooth_sigma: float = 3.0,
# #     saliency_threshold: float = 0.7,
# #     compare_classes=(1, 2),
# #     class_names=None
# # ):
# #     """
# #     Saves:
# #       - Per-sample 1D overlay (ECG + red shaded high-saliency)
# #       - Per-sample 2D overlay (spectrogram + heatmap)
# #       - SR vs AFIB (or any two) comparison panels
# #       - Across-sample min-max normalization for comparability
# #     """
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #     print(f"[Saliency] Saving {max_samples} samples to {out_dir}")

# #     if class_names is None:
# #         class_names = {0: 'SB', 1: 'SR', 2: 'AFIB', 3: 'GSVT'}

# #     collected_1d = []
# #     collected_2d = []
# #     metas = []  # (idx, true, pred, x1_cpu, x2_cpu)

# #     model.eval()
# #     saved = 0
# #     with torch.enable_grad():
# #         for x_1d, x_2d, y in loader:
# #             for i in range(x_1d.size(0)):
# #                 xi1 = x_1d[i:i+1].to(device)
# #                 xi2 = x_2d[i:i+1].to(device)
# #                 yi = int(y[i].item())

# #                 logits = model(xi1, xi2)
# #                 pred = int(logits.argmax(1).item())
# #                 s1, s2, _ = compute_saliency(model, xi1, xi2, pred, device)

# #                 # smooth/normalize 1D
# #                 s1_np = s1.numpy()
# #                 if smooth_sigma > 0:
# #                     s1_np = gaussian_filter1d(s1_np, sigma=smooth_sigma)

# #                 collected_1d.append(torch.from_numpy(s1_np))
# #                 collected_2d.append(s2.detach().cpu())
# #                 metas.append((saved, yi, pred, xi1.detach().cpu(), xi2.detach().cpu()))
# #                 saved += 1
# #                 if saved >= max_samples:
# #                     break
# #             if saved >= max_samples:
# #                 break

# #     # across-sample min-max (1D)
# #     maxlen = max(s.numel() for s in collected_1d)
# #     pad_1d = []
# #     for s in collected_1d:
# #         if s.numel() < maxlen:
# #             pad = torch.zeros(maxlen)
# #             pad[:s.numel()] = s
# #             pad_1d.append(pad)
# #         else:
# #             pad_1d.append(s)
# #     S1 = torch.stack(pad_1d, dim=0)
# #     S1 = torch_minmax(S1)
# #     collected_1d = [S1[i, :metas[i][3].shape[-1]] for i in range(len(metas))]

# #     # across-sample min-max (2D)
# #     H = max(s.shape[0] for s in collected_2d)
# #     W = max(s.shape[1] for s in collected_2d)
# #     pad_2d = []
# #     for s in collected_2d:
# #         canvas = torch.zeros((H, W))
# #         h, w = s.shape
# #         canvas[:h, :w] = s
# #         pad_2d.append(canvas)
# #     S2 = torch.stack(pad_2d, dim=0)
# #     S2 = torch_minmax(S2)
# #     collected_2d = [S2[i, :collected_2d[i].shape[0], :collected_2d[i].shape[1]] for i in range(len(metas))]

# #     # save each sample overlays
# #     for i, (idx, yi, pred, xi1_cpu, xi2_cpu) in enumerate(metas):
# #         s1 = collected_1d[i].numpy()
# #         s2 = collected_2d[i].numpy()
# #         sig = xi1_cpu[0, 0].numpy() if xi1_cpu.ndim == 3 else xi1_cpu.squeeze().numpy()
# #         spec = xi2_cpu[0, 0].numpy() if xi2_cpu.ndim == 3 else xi2_cpu.squeeze().numpy()

# #         # --- match lengths robustly ---
# #         s1 = interp1d_to_length(s1, sig.shape[-1])
# #         s2 = resize_2d_to(s2, spec.shape)

# #         # 1D overlay
# #         thr = float(saliency_threshold)
# #         mask = (s1 >= thr).astype(float)

# #         plt.figure(figsize=(12, 3))
# #         plt.plot(sig, color='black', linewidth=1)
# #         on = False; start = 0
# #         for t in range(len(mask)):
# #             if mask[t] and not on:
# #                 on = True; start = t
# #             if (not mask[t] and on) or (on and t == len(mask)-1):
# #                 end = t if not mask[t] else t+1
# #                 xs = np.arange(start, end)
# #                 plt.fill_between(xs, sig[start:end], 0, color='red', alpha=0.3)
# #                 on = False

# #         plt.title(f"1D ECG + Saliency (true={class_names.get(yi, yi)}, pred={class_names.get(pred, pred)})")
# #         plt.xlabel("Time (samples)")
# #         plt.ylabel("Amplitude")
# #         plt.tight_layout()
# #         plt.savefig(out_dir / f"saliency1d_{i:03d}_true{yi}_pred{pred}.png", dpi=300)
# #         plt.close()

# #         # 2D overlay
# #         plt.figure(figsize=(6, 5))
# #         plt.imshow(spec, aspect='auto', origin='lower', cmap='gray')
# #         plt.imshow(s2,  aspect='auto', origin='lower', cmap='jet', alpha=0.5)
# #         plt.title(f"2D STFT + Saliency (true={class_names.get(yi, yi)}, pred={class_names.get(pred, pred)})")
# #         plt.axis("off")
# #         plt.tight_layout()
# #         plt.savefig(out_dir / f"saliency2d_{i:03d}_true{yi}_pred{pred}.png", dpi=300)
# #         plt.close()


# # --------- Curated gallery (one per class + 2x2 montage) ----------
# def _draw_pair(
#     ecg_1d, sal_1d, spec_2d, sal_2d,
#     cls_true, cls_pred, class_names, out_path,
#     sigma=7, thr=0.7
# ):
#     # normalize per-sample
#     s1 = np_minmax(sal_1d.copy())
#     s2 = np_minmax(sal_2d.copy())

#     # match sizes
#     s1 = interp1d_to_length(s1, len(ecg_1d))
#     s2 = resize_2d_to(s2, spec_2d.shape)

#     if sigma and sigma > 0:
#         s1 = gaussian_filter1d(s1, sigma=sigma)

#     mask1 = s1 >= thr
#     mask2 = s2 >= thr

#     fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

#     # 1D with overlay
#     ax = axes[0]
#     xs = np.arange(len(ecg_1d))
#     ax.plot(xs, ecg_1d, lw=1.2, c='k')
#     in_region = False; start = 0
#     for i, m in enumerate(mask1):
#         if m and not in_region:
#             in_region = True; start = i
#         if in_region and (not m or i == len(mask1)-1):
#             end = i if not m else i+1
#             ax.axvspan(start, end, color='r', alpha=0.20)
#             in_region = False
#     ax.set_title(f"1D ECG (true={class_names[cls_true]}, pred={class_names[cls_pred]})")
#     ax.set_xlabel("Time (samples)"); ax.set_ylabel("Amplitude")
#     ax2 = ax.twinx()
#     ax2.plot(xs, s1, lw=0.8)
#     ax2.set_ylim(0, 1.0)
#     ax2.set_ylabel("Saliency (norm.)", rotation=270, labelpad=12)

#     # 2D STFT + heat + contour
#     ax = axes[1]
#     ax.imshow(spec_2d, aspect='auto', origin='lower')
#     im = ax.imshow(s2, aspect='auto', origin='lower', alpha=0.45)
#     try:
#         ax.contour(mask2.astype(float), levels=[0.5], colors='red', linewidths=1.0)
#     except Exception:
#         pass
#     ax.set_title("2D STFT + Saliency")
#     ax.set_xlabel("Time bins"); ax.set_ylabel("Freq bins")
#     cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
#     cb.set_label("Saliency (norm.)")

#     plt.savefig(out_path, dpi=200)
#     plt.close(fig)


# def save_saliency_gallery(
#     model,
#     loader,
#     device,
#     out_dir: Path,
#     class_names=("SB","SR","AFIB","GSVT"),
#     samples_per_class=1,
#     sigma=7,
#     thr=0.7,
#     representation_type="stft"
# ):
#     out_dir.mkdir(parents=True, exist_ok=True)
#     n_classes = len(class_names)

#     # Collect pools of candidates (prefer correct, high-confidence)
#     pools = {c: [] for c in range(n_classes)}
#     with torch.no_grad():
#         for x1, x2, y in loader:
#             x1 = x1.to(device); x2 = x2.to(device)
#             logits = model(x1, x2)
#             probs = logits.softmax(dim=1)
#             preds = logits.argmax(dim=1)
#             for i in range(len(y)):
#                 yi = int(y[i].item())
#                 pi = float(probs[i, preds[i]].item())
#                 pools[yi].append((pi, (x1[i].detach().cpu(), x2[i].detach().cpu(), yi, int(preds[i].item()))))

#     # Fallback: if a class pool is empty, accept first occurrence regardless of correctness
#     for c in range(n_classes):
#         if len(pools[c]) == 0:
#             for x1, x2, y in loader:
#                 for i in range(len(y)):
#                     if int(y[i].item()) == c:
#                         pools[c].append((0.0, (x1[i], x2[i], c, -1)))
#                         break
#                 if len(pools[c]) > 0:
#                     break

#     chosen = []
#     for c in range(n_classes):
#         if len(pools[c]) == 0:
#             continue
#         pools[c].sort(key=lambda t: t[0], reverse=True)
#         for k in range(min(samples_per_class, len(pools[c]))):
#             chosen.append(pools[c][k][1])

#     panel_paths = []
#     # Compute saliency and save per-class pairs
#     for (x1_cpu, x2_cpu, y_true, _) in chosen:
#         x1 = x1_cpu.unsqueeze(0).to(device).requires_grad_(True)
#         x2 = x2_cpu.unsqueeze(0).to(device).requires_grad_(True)
#         logits = model(x1, x2)
#         pred = int(logits.argmax(dim=1).item())
#         target = pred
#         model.zero_grad(set_to_none=True)
#         logits[:, target].sum().backward()

#         g1 = x1.grad.detach().abs()[0]  # [C,T] or [T]
#         if g1.ndim == 1:
#             s1 = g1.cpu().numpy()
#         else:
#             s1 = g1.max(dim=0)[0].cpu().numpy()  # [T] after channel max

#         g2 = x2.grad.detach().abs()[0]  # [C,H,W] or [H,W]
#         if g2.ndim == 2:
#             s2 = g2.cpu().numpy()
#         else:
#             s2 = g2.max(dim=0)[0].cpu().numpy()

#         one_d = x1_cpu.squeeze().cpu().numpy()
#         if one_d.ndim == 2:  # [C,T] -> use first channel or mean over channels for plotting
#             one_d = one_d.mean(axis=0)
#         two_d = x2_cpu.squeeze().cpu().numpy()
#         if two_d.ndim == 3:  # [C,H,W] -> mean over channels
#             two_d = two_d.mean(axis=0)

#         out_path = out_dir / f"pair_{class_names[y_true]}_pred{class_names[pred] if 0<=pred<n_classes else pred}.png"
#         _draw_pair(one_d, s1, two_d, s2, y_true, pred, class_names, out_path, sigma=sigma, thr=thr)
#         panel_paths.append((y_true, str(out_path)))

#     # 2x2 montage
#     if len(panel_paths) >= 4:
#         panel_paths.sort(key=lambda t: t[0])
#         fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#         for i, ax in enumerate(axes.ravel()):
#             if i >= len(panel_paths):
#                 ax.axis('off'); continue
#             img = plt.imread(panel_paths[i][1])
#             ax.imshow(img); ax.axis('off')
#         plt.suptitle("Curated Saliency Gallery (1D + 2D overlays)", y=0.98, fontsize=14)
#         plt.savefig(out_dir / "gallery_2x2.png", dpi=200)
#         plt.close(fig)


# # --------------------------
# # Evaluation
# # --------------------------
# def evaluate_model(model, test_loader, device: torch.device, output_dir: Path):
#     from sklearn.metrics import (
#         accuracy_score, precision_score, recall_score,
#         f1_score, roc_auc_score, confusion_matrix,
#         precision_recall_fscore_support
#     )

#     print("\n" + "="*70)
#     print("EVALUATING ON TEST SET")
#     print("="*70)

#     model.eval()
#     all_preds, all_labels, all_probs = [], [], []

#     with torch.no_grad():
#         for x_1d, x_2d, labels in tqdm(test_loader, desc="Test Evaluation"):
#             x_1d, x_2d = x_1d.to(device), x_2d.to(device)
#             logits = model(x_1d, x_2d)
#             probs = F.softmax(logits, dim=1)
#             pred = probs.argmax(dim=1)

#             all_preds.extend(pred.cpu().numpy())
#             all_labels.extend(labels.numpy())
#             all_probs.extend(probs.cpu().numpy())

#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
#     all_probs = np.array(all_probs)

#     acc = accuracy_score(all_labels, all_preds)
#     prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
#     rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
#     f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

#     cm = confusion_matrix(all_labels, all_preds)
#     specificity_per_class = []
#     for i in range(cm.shape[0]):
#         tn = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
#         fp = cm[:,i].sum() - cm[i,i]
#         spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#         specificity_per_class.append(spec)
#     specificity = np.average(specificity_per_class, weights=cm.sum(axis=1))

#     try:
#         auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
#         auc_per_class = []
#         for i in range(all_probs.shape[1]):
#             binary = (all_labels == i).astype(int)
#             auc_i = roc_auc_score(binary, all_probs[:, i]) if len(np.unique(binary)) > 1 else 0.0
#             auc_per_class.append(auc_i)
#     except Exception:
#         auc, auc_per_class = 0.0, [0.0]*all_probs.shape[1]

#     precision_per_class, recall_per_class, f1_per_class, support = \
#         precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

#     results = {
#         "accuracy": float(acc),
#         "precision": float(prec),
#         "sensitivity": float(rec),
#         "specificity": float(specificity),
#         "recall": float(rec),
#         "f1": float(f1),
#         "auc": float(auc),
#         "auc_per_class": [float(x) for x in auc_per_class],
#         "precision_per_class": [float(x) for x in precision_per_class],
#         "recall_per_class": [float(x) for x in recall_per_class],
#         "sensitivity_per_class": [float(x) for x in recall_per_class],
#         "specificity_per_class": [float(x) for x in specificity_per_class],
#         "f1_per_class": [float(x) for x in f1_per_class],
#         "confusion_matrix": cm.tolist()
#     }

#     with open(output_dir / 'test_results.json', 'w') as f:
#         json.dump(results, f, indent=2)

#     print("\n" + "="*70)
#     print("TEST RESULTS")
#     print("="*70)
#     print(f"Accuracy:  {acc*100:.2f}%")
#     print(f"Precision: {prec:.4f}")
#     print(f"F1:        {f1:.4f}")
#     print(f"AUC:       {auc:.4f}")
#     print("Confusion Matrix:\n", cm)
#     print("="*70)
#     print(f"Results saved to: {output_dir / 'test_results.json'}")
#     print("="*70)

#     return results


# # --------------------------
# # Training
# # --------------------------
# def train_fusion_model(
#     model: ECGFusionModel,
#     train_loader: DataLoader,
#     val_loader: DataLoader,
#     test_loader: DataLoader,
#     num_epochs: int,
#     learning_rate: float,
#     device: torch.device,
#     output_dir: Path,
#     patience: int = 10,
#     grad_accum_steps: int = 1
# ):
#     output_dir.mkdir(parents=True, exist_ok=True)
#     # Use light label smoothing for stability
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
#     trainable = [p for p in model.parameters() if p.requires_grad]
#     print(f"\nTrainable parameters: {sum(p.numel() for p in trainable):,}")

#     optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
#     # Use warm restarts scheduler for stability
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

#     history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
#     best_val_acc = 0.0
#     epochs_no_improve = 0

#     print("\n" + "="*70)
#     print("PHASE 2: FUSION MODEL TRAINING")
#     print("="*70)

#     total_batches = len(train_loader)
#     for epoch in range(num_epochs):
#         # train
#         model.train()
#         tr_loss, tr_correct, tr_total = 0.0, 0, 0

#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
#         for bi, (x_1d, x_2d, labels) in enumerate(pbar):
#             x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)

#             # Zero grads at the start of each accumulation cycle
#             if (bi % grad_accum_steps) == 0:
#                 optimizer.zero_grad()

#             logits = model(x_1d, x_2d)
#             loss = criterion(logits, labels)

#             # scale loss for gradient accumulation
#             loss_scaled = loss / float(grad_accum_steps)
#             loss_scaled.backward()

#             # Step optimizer on accumulation boundary or on final batch
#             is_last_batch = (bi == (total_batches - 1))
#             if ((bi + 1) % grad_accum_steps == 0) or is_last_batch:
#                 nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
#                 optimizer.step()
#                 # step scheduler with fractional epoch for smoother warm restarts
#                 try:
#                     scheduler.step(epoch + (bi + 1) / float(max(1, total_batches)))
#                 except Exception:
#                     pass

#             tr_loss += loss.item()
#             tr_correct += (logits.argmax(1) == labels).sum().item()
#             tr_total += labels.size(0)
#             pbar.set_postfix({'loss': f'{tr_loss/(bi+1):.4f}', 'acc': f'{100.*tr_correct/tr_total:.2f}%'})

#         train_loss = tr_loss / len(train_loader)
#         train_acc = 100.0 * tr_correct / tr_total

#         # val
#         model.eval()
#         val_loss, val_correct, val_total = 0.0, 0, 0
#         all_preds, all_labels = [], []
#         with torch.no_grad():
#             for x_1d, x_2d, labels in tqdm(val_loader, desc="Validation"):
#                 x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)
#                 logits = model(x_1d, x_2d)
#                 loss = criterion(logits, labels)

#                 val_loss += loss.item()
#                 pred = logits.argmax(1)
#                 val_correct += (pred == labels).sum().item()
#                 val_total += labels.size(0)
#                 all_preds.extend(pred.cpu().numpy()); all_labels.extend(labels.cpu().numpy())

#         val_loss /= len(val_loader)
#         val_acc = 100.0 * val_correct / val_total

#         from sklearn.metrics import f1_score
#         val_f1 = f1_score(all_labels, all_preds, average='weighted')

#         history['train_loss'].append(train_loss)
#         history['train_acc'].append(train_acc)
#         history['val_loss'].append(val_loss)
#         history['val_acc'].append(val_acc)
#         history['val_f1'].append(val_f1)

#         print("="*70)
#         print(f"Epoch {epoch+1}/{num_epochs}")
#         print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
#         print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
#         print(f"  Val F1:     {val_f1:.4f}")

#         # save checkpoints
#         ckpt = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'val_acc': val_acc,
#             'history': history
#         }
#         torch.save(ckpt, output_dir / f'checkpoint_epoch_{epoch}.pth')
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             epochs_no_improve = 0
#             torch.save(ckpt, output_dir / 'best_fusion_model.pth')
#             print(f"  💾 Saved best model (Val Acc: {val_acc:.2f}%)")
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print(f"\n⚠️  Early stopping at epoch {epoch+1}")
#             break

#         scheduler.step()

#     with open(output_dir / 'training_history.json', 'w') as f:
#         json.dump(history, f, indent=2)

#     plot_history(history, output_dir)
#     print(f"\n✓ Training Complete! Best Val Acc: {best_val_acc:.2f}%")

#     # Evaluate best on test
#     if test_loader is not None:
#         best = torch.load(output_dir / 'best_fusion_model.pth', map_location=device)
#         model.load_state_dict(best['model_state_dict'])
#         evaluate_model(model, test_loader, device, output_dir)

#     return model


# # --------------------------
# # Main
# # --------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--aligned-encoder-1d', required=True)
#     parser.add_argument('--aligned-encoder-2d', required=True)
#     parser.add_argument('--data-dir', required=True)
#     parser.add_argument('--representation-type', default='stft')
#     parser.add_argument('--batch-size', type=int, default=32)
#     parser.add_argument('--num-workers', type=int, default=4)
#     parser.add_argument('--feature-dim', type=int, default=384)
#     parser.add_argument('--num-classes', type=int, default=4)
#     parser.add_argument('--num-heads', type=int, default=8)
#     parser.add_argument('--dropout', type=float, default=0.1)

#     # freezing / unfreezing (default frozen; --unfreeze-encoders to train them)
#     parser.add_argument('--unfreeze-encoders', action='store_true')

#     parser.add_argument('--num-epochs', type=int, default=50)
#     parser.add_argument('--learning-rate', type=float, default=1e-4)
#     parser.add_argument('--patience', type=int, default=15)
#     parser.add_argument('--output-dir', required=True)
#     parser.add_argument('--device', default='cuda')
#     parser.add_argument('--grad-accum-steps', type=int, default=1,
#                         help='Number of steps to accumulate gradients before optimizer.step()')

#     # saliency options
#     parser.add_argument('--save-saliency', action='store_true')
#     parser.add_argument('--saliency-samples', type=int, default=12)
#     parser.add_argument('--saliency-threshold', type=float, default=0.7)
#     parser.add_argument('--smooth-sigma', type=float, default=3.0)
#     parser.add_argument('--compare-classes', nargs=2, type=int, default=[1, 2],
#                         help="Two class ids to compare side-by-side (default SR vs AFIB)")
#     # curated gallery options (one per class + 2x2 montage)
#     parser.add_argument('--save-saliency-gallery', action='store_true',
#                         help='Save curated saliency gallery (one panel per class + 2x2 gallery)')
#     parser.add_argument('--gallery-samples-per-class', type=int, default=1,
#                         help='Samples per class for the gallery')
#     parser.add_argument('--gallery-threshold', type=float, default=0.7,
#                         help='Threshold for saliency overlays (0-1)')
#     parser.add_argument('--gallery-sigma', type=int, default=7,
#                         help='Gaussian smoothing sigma for 1D saliency in the gallery')

#     args = parser.parse_args()
#     device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
#     output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)

#     print("="*70)
#     print("PHASE 2: CROSS-ATTENTION FUSION")
#     print("="*70)

#     # data
#     print("\n[Setting up Fusion Data Module]")
#     try:
#         from src.data.fusion_dataset import ECGFusionDataModule
#     except ImportError:
#         from fusion_dataset import ECGFusionDataModule

#     data_module = ECGFusionDataModule(
#         data_dir=args.data_dir,
#         batch_size=args.batch_size,
#         num_workers=args.num_workers,
#         representation_type=args.representation_type
#     )
#     data_module.setup()
#     print("  ✓ Data module ready")

#     # encoders
#     print("\n[Loading Encoders]")
#     enc1_path = Path(args.aligned_encoder_1d)
#     enc2_path = Path(args.aligned_encoder_2d)

#     enc1_state = torch.load(enc1_path, map_location=device)
#     enc2_state = torch.load(enc2_path, map_location=device)

#     try:
#         from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
#         from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
#     except ImportError:
#         from ecg_transformer_1d import create_ecg_transformer_1d_small
#         from models.ecg_transformer_2d import create_ecg_transformer_2d_small

#     encoder_1d = create_ecg_transformer_1d_small()
#     encoder_2d = create_ecg_transformer_2d_small(representation_type=args.representation_type)

#     def _extract_state_dict(ckpt):
#         if isinstance(ckpt, dict):
#             if 'model_state_dict' in ckpt: return ckpt['model_state_dict']
#             if 'state_dict' in ckpt: return ckpt['state_dict']
#             return ckpt
#         raise RuntimeError('Unrecognized checkpoint format')

#     encoder_1d.load_state_dict(_extract_state_dict(enc1_state), strict=False)
#     encoder_2d.load_state_dict(_extract_state_dict(enc2_state), strict=False)
#     encoder_1d, encoder_2d = encoder_1d.to(device), encoder_2d.to(device)
#     print("  ✓ Encoders loaded")

#     freeze = not args.unfreeze_encoders
#     model = ECGFusionModel(
#         encoder_1d=encoder_1d,
#         encoder_2d=encoder_2d,
#         feature_dim=args.feature_dim,
#         num_classes=args.num_classes,
#         num_heads=args.num_heads,
#         dropout=args.dropout,
#         freeze_encoders=freeze
#     ).to(device)

#     # train
#     train_fusion_model(
#         model=model,
#         train_loader=data_module.train_loader,
#         val_loader=data_module.val_loader,
#         test_loader=data_module.test_loader,
#         num_epochs=args.num_epochs,
#         learning_rate=args.learning_rate,
#         device=device,
#         output_dir=output_dir,
#         patience=args.patience,
#         grad_accum_steps=args.grad_accum_steps
#     )

#     # optional: saliency dump (few random samples)
#     if args.save_saliency:
#         sal_dir = output_dir / "saliency"
#         save_saliency_samples(
#             model=model,
#             loader=data_module.test_loader,
#             device=device,
#             out_dir=sal_dir,
#             max_samples=args.saliency_samples,
#             smooth_sigma=args.smooth_sigma,
#             saliency_threshold=args.saliency_threshold,
#             compare_classes=tuple(args.compare_classes),
#             class_names={0:'SB',1:'SR',2:'AFIB',3:'GSVT'}
#         )

#     # curated gallery
#     if args.save_saliency_gallery:
#         gal_dir = output_dir / "saliency_gallery"
#         try:
#             save_saliency_gallery(
#                 model=model,
#                 loader=data_module.test_loader,
#                 device=device,
#                 out_dir=gal_dir,
#                 class_names={0:'SB',1:'SR',2:'AFIB',3:'GSVT'},
#                 samples_per_class=args.gallery_samples_per_class,
#                 sigma=args.gallery_sigma,
#                 thr=args.gallery_threshold,
#                 representation_type=args.representation_type
#             )
#             print(f"[Gallery] Saved to {gal_dir}")
#         except Exception as e:
#             print("[Warning] Failed to save saliency gallery:", repr(e))

#     print("\n✓ Phase 2 Complete!")


# if __name__ == "__main__":
#     main()


################################################################################3
#!/usr/bin/env python3
"""
Phase 2: Cross-Attention Fusion for ECG Classification
- Working Grad-CAM for 1D & 2D (token-level)
- Optional unfreeze of encoders with smaller LR
- Separate PNGs for loss.png, accuracy.png, f1.png
"""

import argparse
import math
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
for path in [project_root, src_dir, src_dir / 'data', src_dir / 'models']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------
class CrossAttentionFusion(nn.Module):
    """Cross-attention module to fuse 1D and 2D CLS features"""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

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
    Wrapper to extract transformer tokens (CLS + patches) bypassing any classifier.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        if not (hasattr(encoder, 'patch_embed') and hasattr(encoder, 'blocks') and hasattr(encoder, 'norm')):
            raise ValueError("Encoder must have patch_embed, blocks, and norm")
        print("  ✓ Wrapped encoder (bypasses classification head)")

    def forward(self, x, return_tokens: bool = False):
        # Tokens: (B, 1+N, D) for ViT-style
        x = self.encoder.patch_embed(x)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)  # (B, 1+N, D)
        if return_tokens:
            cls = x[:, 0]          # (B, D)
            return cls, x          # (B, D), (B, 1+N, D)
        return x[:, 0]             # (B, D)


class ECGFusionModel(nn.Module):
    """
    Fusion model: aligned encoders -> bidirectional cross-attn on CLS -> MLP head.
    Provides a forward_with_tokens() used by Grad-CAM.
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

        self.encoder_1d = EncoderWithoutHead(encoder_1d)
        self.encoder_2d = EncoderWithoutHead(encoder_2d)
        self.freeze_encoders = freeze_encoders

        if freeze_encoders:
            for p in self.encoder_1d.parameters(): p.requires_grad = False
            for p in self.encoder_2d.parameters(): p.requires_grad = False

        # Cross-attention on CLS↔CLS
        self.cross_attn_1d_to_2d = CrossAttentionFusion(feature_dim, num_heads, dropout)
        self.cross_attn_2d_to_1d = CrossAttentionFusion(feature_dim, num_heads, dropout)

        self.norm = nn.LayerNorm(feature_dim)
        fusion_dim = feature_dim * 2

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

        print("\n[ECG Fusion Model] Created")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Fusion dim: {fusion_dim}")
        print(f"  Num classes: {num_classes}")
        print(f"  Num heads: {num_heads}")
        print(f"  Encoders frozen: {freeze_encoders}")

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
        with torch.set_grad_enabled(not self.freeze_encoders):
            cls1 = self.encoder_1d(x_1d)  # (B, D)
            cls2 = self.encoder_2d(x_2d)  # (B, D)

        feat_1d_att = self.cross_attn_1d_to_2d(cls1, cls2)
        feat_2d_att = self.cross_attn_2d_to_1d(cls2, cls1)

        feat_1d_fused = self.norm(cls1 + feat_1d_att)
        feat_2d_fused = self.norm(cls2 + feat_2d_att)

        fused = torch.cat([feat_1d_fused, feat_2d_fused], dim=1)  # (B, 2D)
        logits = self.classifier(fused)                           # (B, C)
        return logits

    def forward_with_tokens(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
        """
        Returns logits and (cls, tokens) for both streams, with tokens grad-enabled.
        Used for Grad-CAM to produce per-token importance maps.
        """
        # Get tokens (detach & requires_grad so they act as grad roots even if encoders are frozen)
        cls1, tok1 = self.encoder_1d(x_1d, return_tokens=True)
        cls2, tok2 = self.encoder_2d(x_2d, return_tokens=True)

        tok1 = tok1.detach().requires_grad_(True)
        tok2 = tok2.detach().requires_grad_(True)
        cls1 = tok1[:, 0]
        cls2 = tok2[:, 0]

        feat_1d_att = self.cross_attn_1d_to_2d(cls1, cls2)
        feat_2d_att = self.cross_attn_2d_to_1d(cls2, cls1)

        feat_1d_fused = self.norm(cls1 + feat_1d_att)
        feat_2d_fused = self.norm(cls2 + feat_2d_att)

        fused = torch.cat([feat_1d_fused, feat_2d_fused], dim=1)
        logits = self.classifier(fused)

        return logits, (cls1, tok1), (cls2, tok2)


# -----------------------------------------------------------------------------
# Metrics & plotting
# -----------------------------------------------------------------------------
def evaluate_model(model: ECGFusionModel, test_loader: DataLoader, device: torch.device, output_dir: Path):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
        precision_recall_fscore_support
    )

    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70)

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

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

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    specificity_per_class = []
    for i in range(cm.shape[0]):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(spec)
    specificity = np.average(specificity_per_class, weights=cm.sum(axis=1))

    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        auc_per_class = []
        for i in range(all_probs.shape[1]):
            binary = (all_labels == i).astype(int)
            if len(np.unique(binary)) > 1:
                auc_i = roc_auc_score(binary, all_probs[:, i])
            else:
                auc_i = 0.0
            auc_per_class.append(auc_i)
    except Exception:
        auc = 0.0
        auc_per_class = [0.0] * all_probs.shape[1]

    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "auc_per_class": [float(x) for x in auc_per_class],
        "precision_per_class": [float(x) for x in precision_per_class],
        "recall_per_class": [float(x) for x in recall_per_class],
        "sensitivity_per_class": [float(x) for x in recall_per_class],
        "specificity_per_class": [float(x) for x in specificity_per_class],
        "f1_per_class": [float(x) for x in f1_per_class],
        "confusion_matrix": cm.tolist()
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)

    # Confusion matrix heatmap
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


# -----------------------------------------------------------------------------
# Grad-CAM (token-based)
# -----------------------------------------------------------------------------
def _normalize_cam(t: torch.Tensor):
    t = t - t.min()
    denom = t.max().clamp_min(1e-12)
    return t / denom

def _infer_patch_grid_2d(x_2d: torch.Tensor, patch_embed) -> tuple[int, int]:
    # Try the model's own metadata first
    if hasattr(patch_embed, 'grid_size'):
        gs = patch_embed.grid_size
        if isinstance(gs, (tuple, list)) and len(gs) == 2:
            return int(gs[0]), int(gs[1])
    if hasattr(patch_embed, 'num_patches'):
        n = int(patch_embed.num_patches)
        s = int(math.sqrt(n))
        return s, max(1, n // max(1, s))
    # Fallback (adjust if you know your grid)
    return 8, 8

def generate_gradcam_samples(model: ECGFusionModel,
                             loader: DataLoader,
                             device: torch.device,
                             out_dir: Path,
                             num_samples: int = 12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    for x_1d, x_2d, labels in loader:
        if saved >= num_samples:
            break
        x_1d = x_1d.to(device)
        x_2d = x_2d.to(device)
        labels = labels.to(device)
        B = x_1d.size(0)

        for i in range(B):
            if saved >= num_samples:
                break

            xi1 = x_1d[i:i+1]
            xi2 = x_2d[i:i+1]
            yi = int(labels[i])

            logits, (_, tok1), (_, tok2) = model.forward_with_tokens(xi1, xi2)
            probs = F.softmax(logits, dim=1)
            pred = int(probs.argmax(dim=1))

            score = logits[0, pred]
            model.zero_grad(set_to_none=True)
            tok1.retain_grad(); tok2.retain_grad()
            score.backward()

            # ----- 1D Grad-CAM -----
            if tok1.grad is not None and tok1.size(1) > 1:
                grad1 = tok1.grad[0, 1:, :]       # (N1, D) ignore CLS
                act1  = tok1.detach()[0, 1:, :]   # (N1, D)
                w1 = grad1.mean(dim=1, keepdim=True)   # (N1, 1)
                cam1 = (act1 * w1).sum(dim=1)          # (N1,)
                cam1 = _normalize_cam(cam1)

                N1 = cam1.numel()
                xs = np.linspace(0, xi1.shape[-1], num=N1, endpoint=False)
                plt.figure(figsize=(10, 2.6))
                plt.plot(xs, cam1.cpu().numpy())
                plt.title(f'1D Grad-CAM (true={yi}, pred={pred})')
                plt.xlabel('Time (samples)')
                plt.ylabel('Importance')
                plt.tight_layout()
                plt.savefig(out_dir / f'gradcam1d_{saved:03d}_true{yi}_pred{pred}.png', dpi=300)
                plt.close()

            # ----- 2D Grad-CAM -----
            if tok2.grad is not None and tok2.size(1) > 1:
                grad2 = tok2.grad[0, 1:, :]
                act2  = tok2.detach()[0, 1:, :]
                w2 = grad2.mean(dim=1, keepdim=True)
                cam2 = (act2 * w2).sum(dim=1)
                cam2 = _normalize_cam(cam2)

                Hp, Wp = _infer_patch_grid_2d(xi2, model.encoder_2d.encoder.patch_embed)
                N2 = cam2.numel()
                if Hp * Wp != N2:
                    s = int(math.sqrt(N2))
                    Hp, Wp = s, max(1, N2 // max(1, s))
                heat = cam2.view(Hp, Wp).cpu().numpy()

                base = xi2[0, 0].detach().cpu().numpy()
                plt.figure(figsize=(6, 5))
                plt.imshow(base, aspect='auto', origin='lower')
                plt.imshow(heat, aspect='auto', origin='lower', alpha=0.5, cmap='jet')
                plt.title(f'2D Grad-CAM (true={yi}, pred={pred})')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(out_dir / f'gradcam2d_{saved:03d}_true{yi}_pred{pred}.png', dpi=300)
                plt.close()

            saved += 1


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_fusion_model(model: ECGFusionModel,
                       train_loader: DataLoader,
                       val_loader: DataLoader,
                       test_loader: DataLoader,
                       num_epochs: int,
                       learning_rate: float,
                       device: torch.device,
                       output_dir: Path,
                       patience: int = 10,
                       make_plots: bool = True):
    """
    Train the model. If encoders are unfrozen, use smaller LR on encoder params.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    # Build optimizer with separate LR for encoders if unfrozen
    enc_params = list(model.encoder_1d.parameters()) + list(model.encoder_2d.parameters())
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder_")]

    if any(p.requires_grad for p in enc_params) and not model.freeze_encoders:
        optimizer = torch.optim.AdamW(
            [
                {'params': enc_params, 'lr': min(learning_rate, 3e-5), 'weight_decay': 0.01},
                {'params': head_params, 'lr': learning_rate, 'weight_decay': 0.01},
            ]
        )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    epochs_without_improvement = 0
    best_val_acc = 0.0

    print("\n" + "="*70)
    print("PHASE 2: FUSION MODEL TRAINING")
    print("="*70)

    for epoch in range(num_epochs):
        # -------------------- Train --------------------
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # -------------------- Val --------------------
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_labels = [], []

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
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        print(f"  Val   F1:   {val_f1:.4f}")

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

    # Save history JSON
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ Training Complete! Best Val Acc: {best_val_acc:.2f}%")

    # Separate metric plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        xs = np.arange(len(history['train_loss']))
        # Loss
        plt.figure(figsize=(8, 6))
        plt.plot(xs, history['train_loss'], label='train_loss')
        plt.plot(xs, history['val_loss'], label='val_loss')
        plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
        plt.tight_layout(); plt.savefig(output_dir / 'loss.png', dpi=300); plt.close()

        # Accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(xs, history['train_acc'], label='train_acc')
        plt.plot(xs, history['val_acc'], label='val_acc')
        plt.title('Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.legend()
        plt.tight_layout(); plt.savefig(output_dir / 'accuracy.png', dpi=300); plt.close()

        # F1
        plt.figure(figsize=(8, 6))
        plt.plot(xs, history['val_f1'], label='val_f1')
        plt.title('F1'); plt.xlabel('Epoch'); plt.ylabel('F1 (weighted)')
        plt.legend(); plt.tight_layout(); plt.savefig(output_dir / 'f1.png', dpi=300); plt.close()
    except Exception:
        pass

    # Final test evaluation
    if test_loader is not None:
        best = torch.load(output_dir / 'best_fusion_model.pth', map_location=device)
        model.load_state_dict(best['model_state_dict'])
        evaluate_model(model, test_loader, device, output_dir)

    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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
    parser.add_argument('--freeze-encoders', action='store_true', help='(kept for compatibility)')
    parser.add_argument('--unfreeze-encoders', action='store_true', help='fine-tune encoders with small LR')
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda')

    # Grad-CAM
    parser.add_argument('--save-gradcam', action='store_true')
    parser.add_argument('--gradcam-samples', type=int, default=12)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("PHASE 2: CROSS-ATTENTION FUSION")
    print("="*70)

    # -------------------- Data --------------------
    print("\n[Setting up Fusion Data Module]")
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
    print("  ✓ Data module setup complete!")

    # -------------------- Encoders --------------------
    print("\n[Loading Encoders]")

    enc1_path = Path(args.aligned_encoder_1d)
    enc2_path = Path(args.aligned_encoder_2d)

    def _extract_state_dict(ckpt):
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt: return ckpt['model_state_dict']
            if 'state_dict' in ckpt:        return ckpt['state_dict']
            return ckpt
        raise RuntimeError('Unrecognized checkpoint format')

    # Try full modules first if *_full.pth
    if str(enc1_path).endswith('_full.pth') and str(enc2_path).endswith('_full.pth'):
        try:
            encoder_1d = torch.load(enc1_path, map_location=device)
            encoder_2d = torch.load(enc2_path, map_location=device)
            print("  ✓ Loaded full encoder modules from Phase 1")
        except Exception as e:
            print("  ✗ Failed to unpickle full modules:", repr(e))
            print("  → Falling back to state_dict loading.")
            # Fall back to state dicts by reading the sidecar .pth (without _full)
            raise RuntimeError(
                "Provide state_dict checkpoints (non-_full) or ensure class paths are importable."
            )
    else:
        # Load state dicts and instantiate fresh encoders
        enc1_state = torch.load(enc1_path, map_location=device)
        enc2_state = torch.load(enc2_path, map_location=device)
        enc1_state = _extract_state_dict(enc1_state)
        enc2_state = _extract_state_dict(enc2_state)

        try:
            from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
            from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
        except ImportError:
            from ecg_transformer_1d import create_ecg_transformer_1d_small
            from models.ecg_transformer_2d import create_ecg_transformer_2d_small

        encoder_1d = create_ecg_transformer_1d_small()
        encoder_2d = create_ecg_transformer_2d_small(representation_type=args.representation_type)

        encoder_1d.load_state_dict(enc1_state, strict=False)
        encoder_2d.load_state_dict(enc2_state, strict=False)
        print("  ✓ Loaded encoder state dicts and created models")

    encoder_1d = encoder_1d.to(device)
    encoder_2d = encoder_2d.to(device)

    # Freeze or unfreeze
    freeze_flag = not args.unfreeze_encoders

    # -------------------- Fusion model --------------------
    model = ECGFusionModel(
        encoder_1d=encoder_1d,
        encoder_2d=encoder_2d,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        dropout=args.dropout,
        freeze_encoders=freeze_flag
    ).to(device)

    # -------------------- Train --------------------
    model = train_fusion_model(
        model=model,
        train_loader=data_module.train_loader,
        val_loader=data_module.val_loader,
        test_loader=data_module.test_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device,
        output_dir=output_dir,
        patience=args.patience,
        make_plots=True
    )

    # -------------------- Grad-CAM --------------------
    if args.save_gradcam:
        best = torch.load(output_dir / 'best_fusion_model.pth', map_location=device)
        model.load_state_dict(best['model_state_dict'])
        grad_dir = output_dir / 'gradcam'
        print(f"\n[Grad-CAM] Saving {args.gradcam_samples} samples to {grad_dir}")
        generate_gradcam_samples(model, data_module.test_loader, device, grad_dir, num_samples=args.gradcam_samples)

    print("\n✓ Phase 2 Complete!")


if __name__ == "__main__":
    main()
######################################################################################################

"""
Phase 2: Cross-Attention Fusion for ECG Classification
COMPLETE WORKING VERSION
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
    
    # Load encoders - check if _full.pth or regular .pth
    print("\n[Loading Encoders]")
    
    encoder_1d_path = Path(args.aligned_encoder_1d)
    encoder_2d_path = Path(args.aligned_encoder_2d)
    
    # Check if these are _full.pth files (contain full module objects)
    if str(encoder_1d_path).endswith('_full.pth') and str(encoder_2d_path).endswith('_full.pth'):
        print("  Loading full encoder modules...")
        try:
            # Attempt to unpickle full module objects. This requires the exact class
            # definitions to be importable under the same module path that was used
            # when the object was saved. If that fails, we catch and print guidance.
            encoder_1d = torch.load(encoder_1d_path, map_location=device)
            encoder_2d = torch.load(encoder_2d_path, map_location=device)
            print("  ✓ Loaded full encoder modules from Phase 1")
        except Exception as e:
            print("  ✗ Failed to unpickle full encoder modules:", repr(e))
            print("\nThe checkpoint appears to be a pickled model object that depends on a class"
                  " (for example 'FeatureExtractor') that isn't importable in this process."
                  "\nTwo options to proceed:\n"
                  "  1) Re-save the encoders as state_dicts from the Phase 1 training script, e.g.:\n"
                  "       torch.save(model.state_dict(), 'encoder_1d.pth')\n"
                  "     Then pass those files to --aligned-encoder-1d/2d.\n"
                  "  2) Make the class definition available on PYTHONPATH so the object can be unpickled\n"
                  "     (not recommended when moving between environments).\n")
            raise RuntimeError("Cannot load pickled encoder modules; please provide state_dict checkpoints or re-save encoders so their class definitions are importable.") from e
    else:
        print("  Loading encoder state dicts...")
        # Load state dicts and apply to newly created models
        encoder_1d_state = torch.load(encoder_1d_path, map_location=device)
        encoder_2d_state = torch.load(encoder_2d_path, map_location=device)
        
        try:
            from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
            from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
        except ImportError:
            from ecg_transformer_1d import create_ecg_transformer_1d_small
            from models.ecg_transformer_2d import create_ecg_transformer_2d_small
        
        encoder_1d = create_ecg_transformer_1d_small()
        encoder_2d = create_ecg_transformer_2d_small(representation_type=args.representation_type)
        
        # Handle common checkpoint formats
        def _extract_state_dict(ckpt):
            if isinstance(ckpt, dict):
                if 'model_state_dict' in ckpt:
                    return ckpt['model_state_dict']
                if 'state_dict' in ckpt:
                    return ckpt['state_dict']
                # Assume it's already a bare state_dict
                return ckpt
            # Unknown format
            raise RuntimeError('Unrecognized checkpoint format for encoder state dict')

        encoder_1d_state = _extract_state_dict(encoder_1d_state)
        encoder_2d_state = _extract_state_dict(encoder_2d_state)

        encoder_1d.load_state_dict(encoder_1d_state, strict=False)
        encoder_2d.load_state_dict(encoder_2d_state, strict=False)
        print("  ✓ Loaded encoder state dicts and created models")
    
    encoder_1d = encoder_1d.to(device)
    encoder_2d = encoder_2d.to(device)
    
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
