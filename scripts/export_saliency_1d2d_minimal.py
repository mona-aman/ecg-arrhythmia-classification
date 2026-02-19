#!/usr/bin/env python3
"""
Minimal, paper-ready saliency for stand-alone 1D and 2D ECG classifiers.
- No fusion dependency.
- One composite figure with AFIB (correct), SR (correct), and an optional 3rd panel.
- 1D: shaded salient regions on the raw trace
- 2D: STFT with heatmap saliency overlay

Usage example (adapt paths to your repo):
PYTHONPATH=.:src \
python scripts/export_saliency_1d2d_minimal.py \
  --ckpt-1d outputs/phase1_baselines/best_1d.pth \
  --ckpt-2d outputs/phase1_baselines/best_2d.pth \
  --data-dir data/processed_batched \
  --repr stft \
  --out outputs/fig_saliency_minimal.png \
  --sigma 7 --thr 0.7 --max-scan 2000
"""

import argparse, json, sys, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- Repo imports (adjust names if yours differ) ----------
# Expect these to exist already in your codebase:
#   - data module: ECGFusionDataModule (yields (x_1d, x_2d, y))
#   - 1D/2D model builders: create_ecg_transformer_1d_small/base, create_ecg_transformer_2d_small/base
try:
    from src.data.fusion_dataset import ECGFusionDataModule
except Exception:
    from src.data.fusion_dataset import ECGFusionDataModule  # fallback if you keep it flat

try:
    from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small as make_1d
except Exception:
    from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small as make_1d

try:
    from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small as make_2d
except Exception:
    from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small as make_2d

# ---------- Utils ----------
def _softmax_probs(logits):
    return F.softmax(logits, dim=1)

def _minmax_np(a, eps=1e-8):
    a = a - np.min(a)
    d = np.max(a) - np.min(a) + eps
    return a / d

def smooth1d(arr, sigma=7):
    # nearest-edge pad to avoid edge drop
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(arr, sigma=sigma, mode="nearest")
    except Exception:
        # cheap fallback: 1D conv with Gaussian kernel
        radius = int(3*sigma)
        xs = np.arange(-radius, radius+1, dtype=np.float32)
        kern = np.exp(-0.5*(xs/sigma)**2)
        kern /= kern.sum()
        pad_l = arr[0] * np.ones(radius, dtype=np.float32)
        pad_r = arr[-1] * np.ones(radius, dtype=np.float32)
        z = np.concatenate([pad_l, arr, pad_r])
        y = np.convolve(z, kern, mode="same")
        return y[radius:-radius]

@torch.no_grad()
def predict(model, x):
    return model(x).argmax(1)

def compute_saliency_1d(model, x1, target_idx, device):
    """Grad wrt 1D input → |grad| mean over channels."""
    model.eval()
    x = x1.clone().detach().requires_grad_(True).to(device)
    logits = model(x)
    logit = logits[0, target_idx]
    model.zero_grad(set_to_none=True)
    logit.backward()
    s = x.grad.detach().abs()[0].mean(dim=0)  # (T,)
    return s.cpu(), logits.detach().cpu()

def compute_saliency_2d(model, x2, target_idx, device):
    """Grad wrt 2D input → |grad| mean over leads."""
    model.eval()
    x = x2.clone().detach().requires_grad_(True).to(device)
    logits = model(x)
    logit = logits[0, target_idx]
    model.zero_grad(set_to_none=True)
    logit.backward()
    s = x.grad.detach().abs()[0].mean(dim=0)  # (H,W)
    return s.cpu(), logits.detach().cpu()

def load_ckpt_exact(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
    model.load_state_dict(sd)
    return model

def pick_examples(dm, model_1d, model_2d, class_names, want=("AFIB","SR"), max_scan=2000, device="cuda"):
    """
    Scan test set until we find:
      - one AFIB correctly classified by both 1D and 2D (or at least by its own model)
      - one SR correctly classified similarly
    Returns list of tuples: (idx, x1[None], x2[None], y_int, y_name)
    """
    name2id = {v:i for i,v in class_names.items()}
    wanted = list(want)
    got = {}
    picked = []

    it = 0
    for xb1, xb2, yb in dm.test_loader:
        for i in range(xb1.size(0)):
            it += 1
            if it > max_scan: break
            yi = int(yb[i].item())
            yname = class_names.get(yi, str(yi))

            # only search among desired classes
            if yname not in wanted: 
                continue

            x1i = xb1[i:i+1].to(device)
            x2i = xb2[i:i+1].to(device)

            # check correctness (relax: accept if correct on its own modality)
            p1 = int(predict(model_1d, x1i).item())
            p2 = int(predict(model_2d, x2i).item())
            ok = (p1 == yi) or (p2 == yi)
            if not ok: 
                continue

            if yname not in got:
                picked.append((len(picked), xb1[i:i+1].cpu(), xb2[i:i+1].cpu(), yi, yname))
                got[yname] = True
                if len(got) == len(wanted):
                    return picked

        if it > max_scan: 
            break

    return picked  # may be < len(wanted)

def panel_plot(panels, out_path, thr=0.7, sigma=7, titles=None):
    """
    panels: list of dicts with keys:
      'sig': (T,), 'sal1d': (T,), 'spec': (H,W), 'sal2d': (H,W), 'title': str
    Renders:
      Top: 1D with shaded saliency (>=thr of max)
      Middle: 2D spectrogram with saliency heatmap (alpha)
      Bottom: left blank (label: 'fusion (n/a)')
    """
    n = len(panels)
    fig_h = 2.1*n  # ~0.7 per row * 3 rows
    fig = plt.figure(figsize=(10, fig_h*1.5))

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(nrows=3*n, ncols=1, hspace=0.25)

    for i, P in enumerate(panels):
        sig = P["sig"]
        sal1d = P["sal1d"]
        spec = P["spec"]
        sal2d = P["sal2d"]
        title = P.get("title","")

        # normalize & smooth 1D saliency
        s1 = sal1d.numpy()
        s1 = smooth1d(s1, sigma=sigma)
        s1 = _minmax_np(s1)
        mask = s1 >= thr*np.max(s1)

        # Top row: 1D with shaded saliency
        ax1 = fig.add_subplot(gs[3*i+0, 0])
        ax1.plot(sig, linewidth=1)
        on = False; start = 0
        for t in range(len(mask)):
            if mask[t] and not on:
                on=True; start=t
            if (not mask[t] and on) or (on and t==len(mask)-1):
                end = t if not mask[t] else t+1
                ax1.fill_between(np.arange(start, end), sig[start:end], 0, alpha=0.3)
                on=False
        ax1.set_ylabel("1D")
        ax1.set_title(title, fontsize=11)

        # Middle row: 2D with heat overlay
        ax2 = fig.add_subplot(gs[3*i+1, 0])
        ax2.imshow(spec, aspect='auto', origin='lower', cmap='gray')
        hm = sal2d.numpy()
        hm = _minmax_np(hm)
        ax2.imshow(hm, aspect='auto', origin='lower', cmap='jet', alpha=0.45)
        ax2.set_ylabel("2D STFT")
        ax2.set_xticks([]); ax2.set_yticks([])

        # Bottom row: reserved / optional
        ax3 = fig.add_subplot(gs[3*i+2, 0])
        ax3.axis('off')
        ax3.text(0.01, 0.6, "fusion (n/a)", fontsize=9)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-1d", required=True)
    p.add_argument("--ckpt-2d", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--repr", default="stft", choices=["stft","mel","cwt"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="outputs/fig_saliency_minimal.png")
    p.add_argument("--sigma", type=float, default=7.0)
    p.add_argument("--thr", type=float, default=0.7)
    p.add_argument("--max-scan", type=int, default=2000)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_names = {0:"SB", 1:"SR", 2:"AFIB", 3:"GSVT"}

    # Data (we only need test split)
    dm = ECGFusionDataModule(
        data_dir=args.data_dir,
        batch_size=32,
        num_workers=4,
        representation_type=args.repr
    )
    dm.setup()

    # Models
    model_1d = make_1d().to(device).eval()
    model_2d = make_2d(representation_type=args.repr).to(device).eval()
    load_ckpt_exact(model_1d, args.ckpt_1d, device)
    load_ckpt_exact(model_2d, args.ckpt_2d, device)

    # Pick panels: AFIB + SR (correct)
    picked = pick_examples(dm, model_1d, model_2d, class_names, want=("AFIB","SR"),
                           max_scan=args.max_scan, device=device)
    if len(picked) == 0:
        print("Could not find matching examples; try increasing --max-scan or ensure class mapping.")
        sys.exit(1)

    # Build panels
    panels = []
    for idx, x1_cpu, x2_cpu, yi, yname in picked:
        x1 = x1_cpu.to(device)
        x2 = x2_cpu.to(device)

        # predictions for titles
        with torch.no_grad():
            p1 = int(predict(model_1d, x1).item())
            p2 = int(predict(model_2d, x2).item())

        # saliency target = model's own prediction (standard in saliency plots)
        s1, _ = compute_saliency_1d(model_1d, x1, p1, device)
        s2, _ = compute_saliency_2d(model_2d, x2, p2, device)

        sig = x1_cpu[0,0].numpy()
        spec = x2_cpu[0,0].numpy()

        title = f"{yname} (true) | 1D pred={yname if p1==yi else class_names[p1]} | 2D pred={yname if p2==yi else class_names[p2]}"
        panels.append(dict(sig=sig, sal1d=s1, spec=spec, sal2d=s2, title=title))

    # Render one composite figure
    panel_plot(panels, args.out, thr=args.thr, sigma=args.sigma)
    print(f"✓ Saved: {args.out}")

if __name__ == "__main__":
    main()
