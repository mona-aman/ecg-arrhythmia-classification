#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---- local imports (robust to cwd) ----
import sys
proj = Path(__file__).resolve().parent.parent
for p in [proj, proj/"src", proj/"src/data", proj/"src/models"]:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

# === saliency helpers (your versions) ===
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def _minmax_torch(x, eps=1e-8):
    x = x - x.min()
    return x / (x.max() + eps)

@torch.no_grad()
def _softmax_probs(logits): return F.softmax(logits, dim=1)

def compute_saliency(model, x1, x2, target_idx, device):
    # enable grads for inputs only
    model.eval()
    x1 = x1.clone().detach().requires_grad_(True).to(device)
    x2 = x2.clone().detach().requires_grad_(True).to(device)

    # forward WITHOUT no_grad (we need gradients)
    logits = model(x1, x2)
    target_logit = logits[0, target_idx]
    model.zero_grad(set_to_none=True)
    target_logit.backward()

    s1 = x1.grad.abs().detach()[0].mean(dim=0)   # [T]
    s2 = x2.grad.abs().detach()[0].mean(dim=0)   # [H, W]
    return s1, s2, logits.detach().cpu()

def save_saliency_samples(
    model,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_samples: int = 12,
    smooth_sigma: float = 3.0,
    saliency_threshold: float = 0.7,
    class_names=None
):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Saliency] Saving {max_samples} samples to {out_dir}")
    if class_names is None:
        class_names = {0:'SB',1:'SR',2:'AFIB',3:'GSVT'}

    collected_1d, collected_2d, metas = [], [], []
    model.eval()
    saved = 0

    # we need gradients through the forward
    with torch.enable_grad():
        for x_1d, x_2d, y in loader:
            bsz = x_1d.size(0)
            for i in range(bsz):
                xi1 = x_1d[i:i+1].to(device)
                xi2 = x_2d[i:i+1].to(device)
                yi  = int(y[i].item())

                logits = model(xi1, xi2)
                pred = int(logits.softmax(1).argmax(1).item())
                s1, s2, _ = compute_saliency(model, xi1, xi2, pred, device)

                # smooth + to tensor
                s1_np = s1.cpu().numpy()
                if smooth_sigma > 0: s1_np = gaussian_filter1d(s1_np, sigma=smooth_sigma)
                s1 = torch.from_numpy(s1_np)

                collected_1d.append(s1)
                collected_2d.append(s2.detach().cpu())
                metas.append((saved, yi, pred, x_1d[i:i+1].cpu(), x_2d[i:i+1].cpu()))
                saved += 1
                if saved >= max_samples: break
            if saved >= max_samples: break

    # normalize across samples
    maxlen = max(s.numel() for s in collected_1d)
    pad_1d = []
    for s in collected_1d:
        if s.numel() < maxlen:
            pad = torch.zeros(maxlen); pad[:s.numel()] = s; pad_1d.append(pad)
        else:
            pad_1d.append(s)
    S1 = _minmax_torch(torch.stack(pad_1d, dim=0))
    collected_1d = [S1[i, :metas[i][3].shape[-1]] for i in range(len(metas))]

    H = max(s.shape[0] for s in collected_2d); W = max(s.shape[1] for s in collected_2d)
    pad_2d = []
    for s in collected_2d:
        canvas = torch.zeros((H, W)); h, w = s.shape; canvas[:h, :w] = s; pad_2d.append(canvas)
    S2 = _minmax_torch(torch.stack(pad_2d, dim=0))
    collected_2d = [S2[i, :collected_2d[i].shape[0], :collected_2d[i].shape[1]] for i in range(len(metas))]

    # write pngs
    for i, (_, yi, pred, xi1_cpu, xi2_cpu) in enumerate(metas):
        s1 = collected_1d[i]; s2 = collected_2d[i]
        sig  = xi1_cpu[0, 0].numpy()
        spec = xi2_cpu[0, 0].numpy()

        thr = float(saliency_threshold); mask = (s1.numpy() >= thr).astype(float)

        # 1D shaded regions
        plt.figure(figsize=(12, 3))
        plt.plot(sig, linewidth=1)
        on=False; start=0
        for t in range(len(mask)):
            if mask[t] and not on: on=True; start=t
            if (not mask[t] and on) or (on and t==len(mask)-1):
                end=t if not mask[t] else t+1
                plt.fill_between(np.arange(start, end), sig[start:end], 0, alpha=0.3)
                on=False
        plt.title(f"1D Saliency (true={class_names.get(yi, yi)}, pred={class_names.get(pred, pred)})")
        plt.xlabel("Time"); plt.ylabel("Amplitude")
        plt.tight_layout(); plt.savefig(out_dir / f"saliency1d_{i:03d}_t{yi}_p{pred}.png", dpi=300); plt.close()

        # 2D heat overlay
        plt.figure(figsize=(6, 5))
        plt.imshow(spec, aspect='auto', origin='lower', cmap='gray')
        plt.imshow(s2.numpy(), aspect='auto', origin='lower', alpha=0.5)  # default colormap
        plt.title(f"2D STFT + Saliency (true={class_names.get(yi, yi)}, pred={class_names.get(pred, pred)})")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(out_dir / f"saliency2d_{i:03d}_t{yi}_p{pred}.png", dpi=300); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Fusion checkpoint: best_fusion_model.pth")
    ap.add_argument("--aligned-encoder-1d", required=True)
    ap.add_argument("--aligned-encoder-2d", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--representation-type", default="stft")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=7.0)
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--gallery", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data module ----
    print("[Setting up Fusion Data Module]")
    try:
        from src.data.fusion_dataset import ECGFusionDataModule
    except ImportError:
        from fusion_dataset import ECGFusionDataModule

    dm = ECGFusionDataModule(
        data_dir=args.data_dir,
        batch_size=32,
        num_workers=4,
        representation_type=args.representation_type
    )
    dm.setup()
    print("\n  ✓ Data module ready")

    # ---- Build fusion model skeleton ----
    try:
        from src.models.fusion_model import ECGFusionModel
    except ImportError:
        from fusion_model import ECGFusionModel

    # Wrap encoders (heads bypassed)
    # If you already have helpers that return encoder-only modules, import and use them
    aligned_1d = torch.load(args.aligned_encoder_1d, map_location=device)
    aligned_2d = torch.load(args.aligned_encoder_2d, map_location=device)

    # If files are state_dicts, adapt:
    if isinstance(aligned_1d, dict) and "state_dict" not in aligned_1d:
        # You have to reconstruct the encoder class then load_state_dict.
        # Assuming create_* helpers exist and produce the same encoder shapes:
        try:
            from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
            from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
        except ImportError:
            from ecg_transformer_1d import create_ecg_transformer_1d_small
            from models.ecg_transformer_2d import create_ecg_transformer_2d_small

        enc1 = create_ecg_transformer_1d_small()
        enc2 = create_ecg_transformer_2d_small(representation_type=args.representation_type)

        # FeatureExtractor like in your Phase 1 file:
        import torch.nn as nn
        class FeatureExtractor(nn.Module):
            def __init__(self, transformer):
                super().__init__()
                self.patch_embed = transformer.patch_embed
                self.blocks = transformer.blocks
                self.norm = transformer.norm
            def forward(self, x):
                x = self.patch_embed(x)
                for block in self.blocks: x = block(x)
                x = self.norm(x)
                return x[:,0]

        enc1 = FeatureExtractor(enc1); enc2 = FeatureExtractor(enc2)
        enc1.load_state_dict(aligned_1d); enc2.load_state_dict(aligned_2d)
    else:
        # full modules were saved
        enc1, enc2 = aligned_1d, aligned_2d

    enc1.to(device).eval()
    enc2.to(device).eval()

    model = ECGFusionModel(
        encoder_1d=enc1,
        encoder_2d=enc2,
        feature_dim=384,
        fusion_dim=768,
        num_classes=4,
        num_heads=8,
        dropout=0.2
    ).to(device).eval()

    # load classifier/fusion head from ckpt
    ckpt = torch.load(args.ckpt, map_location=device)
    if "model_state_dict" in ckpt:
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if missing or unexpected:
            print(f"[load_state_dict] missing={len(missing)} unexpected={len(unexpected)}")
    else:
        # full model was saved
        model.load_state_dict(ckpt, strict=False)

    # ---- save saliency ----
    save_saliency_samples(
        model=model,
        loader=dm.test_loader,
        device=device,
        out_dir=out_dir,
        max_samples=args.samples,
        smooth_sigma=args.sigma,
        saliency_threshold=args.thr,
        class_names={0:'SB',1:'SR',2:'AFIB',3:'GSVT'}
    )

    if args.gallery:
        # FIX: build gallery path without mixing Path and "+"
        gallery_dir = out_dir.parent / f"{out_dir.name}_gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        # Simple gallery: copy/aggregate first N images (or you can montage with PIL)
        # Here we just print where individual images are.
        print(f"[Saliency] Gallery folder prepared at: {gallery_dir} (place your selections here)")

    print("✓ Done.")

if __name__ == "__main__":
    main()
