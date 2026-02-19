#!/usr/bin/env python3
"""
Single-class, vertically-aligned saliency visualization for 3 models:
1) 1D-only, 2) 2D-only, 3) Stable Fusion

Layout per example (columns = models):
  Row 0: Column headers (titles only)
  Row 1: 1D waveform with bright saliency overlay (if model uses 1D)
  Row 2: 2D STFT with bright saliency overlay (if model uses 2D)
  Row 3: Modality weights bar chart with % labels (forced for solo models)
"""

import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ==== repo imports (adjust if your paths differ) ====
from src.data.fusion_dataset import ECGFusionDataModule
from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small as make_1d
from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small as make_2d
from src.models.fusion_simple import ECGFusionModelSimple


# ---------- utilities ----------
def _minmax(a, eps: float = 1e-8):
    a = a - a.min()
    d = a.max() - a.min() + eps
    return a / d

def smooth1d(arr, sigma=7):
    try:
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(arr, sigma=sigma, mode="nearest")
    except Exception:
        k = max(1, int(2 * sigma) + 1)
        pad = k // 2
        z = np.pad(arr, (pad, pad), mode='edge')
        kern = np.ones(k) / k
        return np.convolve(z, kern, mode='same')[pad:-pad]

def load_exact(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = obj.get("model_state_dict", obj)
    model.load_state_dict(sd, strict=False)
    return model


# ---------- saliency (input-gradient) ----------
def saliency_1d(model, x1, target_idx, device):
    model.eval()
    x = x1.clone().detach().requires_grad_(True).to(device)
    logits = model(x)
    logit = logits[0, target_idx]
    model.zero_grad(set_to_none=True)
    logit.backward()
    s = x.grad.detach().abs()[0].mean(dim=0)  # (T,)
    return s.cpu(), logits.detach().cpu()

def saliency_2d(model, x2, target_idx, device):
    model.eval()
    x = x2.clone().detach().requires_grad_(True).to(device)
    logits = model(x)
    logit = logits[0, target_idx]
    model.zero_grad(set_to_none=True)
    logit.backward()
    s = x.grad.detach().abs()[0].mean(dim=0)  # (H, W)
    return s.cpu(), logits.detach().cpu()

def saliency_fusion(model_f, x1, x2, target_idx, device):
    """
    Compute input-gradient saliency for both inputs on the fusion model.
    Returns (sal1d(T,), sal2d(H,W), logits, modality_weights[2])
    """
    model_f.eval()
    x1v = x1.clone().detach().requires_grad_(True).to(device)
    x2v = x2.clone().detach().requires_grad_(True).to(device)
    logits, gates = model_f(x1v, x2v, return_gates=True)  # gates softmax (B,2)
    logit = logits[0, target_idx]
    model_f.zero_grad(set_to_none=True)
    logit.backward()
    s1 = x1v.grad.detach().abs()[0].mean(dim=0).cpu()   # (T,)
    s2 = x2v.grad.detach().abs()[0].mean(dim=0).cpu()   # (H,W)
    return s1, s2, logits.detach().cpu(), gates[0].detach().cpu()


# ---------- example picker ----------
def find_example_for_class(
    dm, m1d, m2d, mf_stable, class_names, target_class_name,
    device, max_scan=10000, require_all_correct=True
):
    """Return a single (x1,x2,y,preds,gates_stable) for target class."""
    idx_of = {v: k for k, v in class_names.items()}
    target_idx = idx_of[target_class_name]
    scanned = 0

    for xb1, xb2, yb in dm.test_loader:
        b = xb1.size(0)
        for i in range(b):
            yi = int(yb[i].item())
            if yi != target_idx:
                continue

            x1 = xb1[i:i+1].to(device)
            x2 = xb2[i:i+1].to(device)

            with torch.no_grad():
                p1 = int(m1d(x1).argmax(1).item())
                p2 = int(m2d(x2).argmax(1).item())
                logits_st, gates_st = mf_stable(x1, x2, return_gates=True)
                pf_st = int(logits_st.argmax(1).item())

            if require_all_correct:
                ok = (p1 == yi) and (p2 == yi) and (pf_st == yi)
            else:
                ok = (pf_st == yi)

            if ok:
                return dict(
                    x1=xb1[i:i+1].cpu(), x2=xb2[i:i+1].cpu(), y=yi,
                    preds=(p1, p2, pf_st),
                    gates_stable=gates_st[0].cpu()
                )

        scanned += b
        if scanned >= max_scan:
            break

    return None


# ---------- plotting helpers ----------
def _bright_spec(spec, pct=(2, 98), gamma=0.7):
    """Log + percentile clip + gamma for a crisp background."""
    s = np.log10(np.maximum(spec, 1e-10))
    vmin, vmax = np.percentile(s, pct)
    s = np.clip(s, vmin, vmax)
    s = (s - vmin) / (vmax - vmin + 1e-8)
    s = np.power(s, gamma)
    return s

def plot_1d_saliency(ax, sig, saliency, time_axis, sigma=6, thr=0.75, title=""):
    s = _minmax(smooth1d(saliency.numpy(), sigma=sigma))

    # Main waveform
    ax.plot(time_axis, sig, linewidth=1.1, color="#1f2937")  # dark gray

    # Highlight contiguous high-saliency regions
    cutoff = thr * (s.max() if s.max() > 0 else 1.0)
    mask = s >= cutoff
    if mask.any():
        on = False; start = 0
        for t in range(len(mask)):
            if mask[t] and not on:
                on = True; start = t
            if (not mask[t] and on) or (on and t == len(mask) - 1):
                end = t if not mask[t] else t + 1
                ax.axvspan(time_axis[start],
                           time_axis[end - 1] if end > start else time_axis[end],
                           color="#ef4444", alpha=0.28, lw=0)
                on = False

    ax.set_xlim(time_axis[0], time_axis[-1])
    ax.set_ylabel("1D", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.grid(True, alpha=0.22, linewidth=0.6)
    if title:
        ax.set_title(title, fontsize=10, pad=6)

def plot_2d_saliency(ax, spec, saliency, time_axis, fs, title=""):
    H, W = spec.shape
    base = _bright_spec(spec, pct=(2, 98), gamma=0.7)
    ax.imshow(base, aspect='auto', origin='lower',
              extent=[time_axis[0], time_axis[-1], 0, H],
              cmap='magma', alpha=1.0)

    hm = _minmax(saliency.numpy())
    ax.imshow(hm, aspect='auto', origin='lower',
              extent=[time_axis[0], time_axis[-1], 0, H],
              cmap='turbo', alpha=0.55)

    ax.set_ylabel("2D STFT\n(freq bins)", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_yticks([0, H // 2, H])
    ax.grid(False)
    if title:
        ax.set_title(title, fontsize=10, pad=6)

def plot_weights(ax, gates, title=None):
    w = gates.numpy() * 100.0
    colors = ["#2563eb", "#ef4444"]  # 1D blue, 2D red
    bars = ax.bar([0, 1], w, color=colors, edgecolor='black', linewidth=0.8)
    for b, pct in zip(bars, w):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2,
                f"{pct:.0f}%", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["1D", "2D"], fontsize=10)
    ax.set_ylabel("Modality weight (%)", fontsize=9)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=10, pad=6)


# ---------- main figure ----------
def render_single_class(meta, sig, spec, sals, gates_st,
                        out_path, fs=250, thr=0.75, sigma=6,
                        class_names=None):
    """
    Build one figure: 3 columns (1D, 2D, Stable Fusion) × 4 rows
      Row 0: titles
      Row 1: 1D plot
      Row 2: 2D plot
      Row 3: modality weights
    """
    p1, p2, pf_st = meta["preds"]
    yi = meta["y"]
    time_axis = np.arange(len(sig)) / fs

    fig = plt.figure(figsize=(13.5, 9))
    gs = gridspec.GridSpec(
        nrows=4, ncols=3,
        height_ratios=[0.18, 1.2, 1.2, 0.9],
        hspace=0.42, wspace=0.30
    )

    # Column headers row
    headers = ["1D Model", "2D Model", "Stable Fusion"]
    for c, text in enumerate(headers):
        axh = fig.add_subplot(gs[0, c]); axh.axis("off")
        axh.text(0.5, 0.5, text, ha="center", va="center",
                 fontsize=14, fontweight="bold", transform=axh.transAxes)

    # Summary header (true/preds)
    fig.suptitle(
        f"True={class_names[yi]} | 1D={class_names[p1]} | 2D={class_names[p2]} | FUS={class_names[pf_st]}",
        fontsize=15, fontweight="bold", y=0.98
    )

    # Column 0: 1D-only
    ax = fig.add_subplot(gs[1, 0])
    plot_1d_saliency(ax, sig, sals["1d_solo"], time_axis, sigma=sigma, thr=thr,
                     title=f"Pred: {class_names[p1]}")
    ax = fig.add_subplot(gs[2, 0]); ax.axis("off")
    ax.text(0.5, 0.5, "No 2D for 1D model", ha="center", va="center",
            fontsize=10, color="#6b7280", transform=ax.transAxes)
    ax = fig.add_subplot(gs[3, 0])
    plot_weights(ax, gates=torch.tensor([1.0, 0.0]), title="Weights (forced)")

    # Column 1: 2D-only
    ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
    ax.text(0.5, 0.5, "No 1D for 2D model", ha="center", va="center",
            fontsize=10, color="#6b7280", transform=ax.transAxes)
    ax = fig.add_subplot(gs[2, 1])
    plot_2d_saliency(ax, spec, sals["2d_solo"], time_axis, fs,
                     title=f"Pred: {class_names[p2]}")
    ax = fig.add_subplot(gs[3, 1])
    plot_weights(ax, gates=torch.tensor([0.0, 1.0]), title="Weights (forced)")

    # Column 2: Stable Fusion
    ax = fig.add_subplot(gs[1, 2])
    plot_1d_saliency(ax, sig, sals["1d_st"], time_axis, sigma=sigma, thr=thr,
                     title=f"Pred: {class_names[pf_st]}")
    ax = fig.add_subplot(gs[2, 2])
    plot_2d_saliency(ax, spec, sals["2d_st"], time_axis, fs)
    ax = fig.add_subplot(gs[3, 2])
    plot_weights(ax, gates_st, title="Weights")

    # Minimal legend
    lg_ax = fig.add_axes([0.015, 0.01, 0.97, 0.08]); lg_ax.axis("off")
    patch = mpatches.Patch(color="#ef4444", alpha=0.28, label="High-saliency regions")
    lg_ax.legend(handles=[patch], loc="center", ncols=1, frameon=False, fontsize=11)

    # Save
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.99)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved: {out_path}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-1d", required=True)
    ap.add_argument("--ckpt-2d", required=True)
    ap.add_argument("--ckpt-fusion-stable", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--repr", default="stft", choices=["stft", "mel", "cwt"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="outputs/saliency/fig_single_class.png")
    ap.add_argument("--sigma", type=float, default=6.0)
    ap.add_argument("--thr", type=float, default=0.75)
    ap.add_argument("--max-scan", type=int, default=8000)
    ap.add_argument("--dim-1d", type=int, default=384)
    ap.add_argument("--dim-2d", type=int, default=384)
    ap.add_argument("--fs", type=int, default=250)
    ap.add_argument("--focus-class", default="AFIB", choices=["SB", "SR", "AFIB", "GSVT"])
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="accept example where not all three models are correct")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    class_names = {0: "SB", 1: "SR", 2: "AFIB", 3: "GSVT"}

    print(f"Loading data from {args.data_dir} ...")
    dm = ECGFusionDataModule(
        data_dir=args.data_dir, batch_size=32, num_workers=4,
        representation_type=args.repr
    )
    dm.setup()

    print("Loading models ...")
    model_1d = make_1d().to(device).eval()
    model_2d = make_2d(representation_type=args.repr).to(device).eval()
    load_exact(model_1d, args.ckpt_1d, device)
    load_exact(model_2d, args.ckpt_2d, device)

    fusion_stable = ECGFusionModelSimple(
        model_1d, model_2d,
        dim_1d=args.dim_1d, dim_2d=args.dim_2d,
        num_classes=4
    ).to(device).eval()
    load_exact(fusion_stable, args.ckpt_fusion_stable, device)

    print(f"Searching for a {args.focus_class} example ...")
    meta = find_example_for_class(
        dm, model_1d, model_2d, fusion_stable,
        class_names, args.focus_class, device,
        max_scan=args.max_scan, require_all_correct=not args.allow_mismatch
    )
    if meta is None:
        print("❌ No suitable example found. Try --allow-mismatch or increase --max-scan.")
        sys.exit(1)

    # Prepare inputs for saliency
    x1 = meta["x1"].to(device)  # shape (1, C, T)
    x2 = meta["x2"].to(device)  # shape (1, C, H, W)
    sig = meta["x1"][0, 0].numpy()        # for rendering 1D
    spec = meta["x2"][0, 0].numpy()       # for rendering 2D
    p1, p2, pf_st = meta["preds"]

    print("Computing saliency maps ...")
    # 1D model saliency at its own predicted class
    s1d_solo, _ = saliency_1d(model_1d, x1, target_idx=p1, device=device)
    # 2D model saliency at its own predicted class
    s2d_solo, _ = saliency_2d(model_2d, x2, target_idx=p2, device=device)
    # stable fusion saliency at its predicted class
    s1d_st, s2d_st, _, gates_st = saliency_fusion(fusion_stable, x1, x2, target_idx=pf_st, device=device)

    sals = {
        "1d_solo": s1d_solo,
        "2d_solo": s2d_solo,
        "1d_st":   s1d_st,
        "2d_st":   s2d_st,
    }

    render_single_class(
        meta, sig, spec, sals, meta["gates_stable"],
        out_path=args.out, fs=args.fs, thr=args.thr, sigma=args.sigma, class_names=class_names
    )


if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# Single-class, vertically-aligned saliency visualization for ALL 4 models:
# 1D-only, 2D-only, Stable Fusion, Tuned Fusion

# Layout per example (columns = models):
#   Row 0: Column headers (titles only)
#   Row 1: 1D waveform with bright saliency overlay (if model uses 1D)
#   Row 2: 2D STFT with bright saliency overlay (if model uses 2D)
#   Row 3: Modality weights bar chart with % labels
# """

# import argparse, sys
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# import matplotlib.patches as mpatches

# # ==== repo imports (adjust if your paths differ) ====
# from src.data.fusion_dataset import ECGFusionDataModule
# from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small as make_1d
# from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small as make_2d
# from src.models.fusion_simple import ECGFusionModelSimple


# # ---------- utilities ----------
# def _minmax(a, eps: float = 1e-8):
#     a = a - a.min()
#     d = a.max() - a.min() + eps
#     return a / d

# def smooth1d(arr, sigma=7):
#     try:
#         from scipy.ndimage import gaussian_filter1d
#         return gaussian_filter1d(arr, sigma=sigma, mode="nearest")
#     except Exception:
#         k = max(1, int(2 * sigma) + 1)
#         pad = k // 2
#         z = np.pad(arr, (pad, pad), mode='edge')
#         kern = np.ones(k) / k
#         return np.convolve(z, kern, mode='same')[pad:-pad]

# def load_exact(model, ckpt_path, device):
#     obj = torch.load(ckpt_path, map_location=device, weights_only=False)
#     sd = obj.get("model_state_dict", obj)
#     model.load_state_dict(sd, strict=False)
#     return model


# # ---------- saliency (input-gradient) ----------
# def saliency_1d(model, x1, target_idx, device):
#     model.eval()
#     x = x1.clone().detach().requires_grad_(True).to(device)
#     logits = model(x)
#     logit = logits[0, target_idx]
#     model.zero_grad(set_to_none=True)
#     logit.backward()
#     s = x.grad.detach().abs()[0].mean(dim=0)  # (T,)
#     return s.cpu(), logits.detach().cpu()

# def saliency_2d(model, x2, target_idx, device):
#     model.eval()
#     x = x2.clone().detach().requires_grad_(True).to(device)
#     logits = model(x)
#     logit = logits[0, target_idx]
#     model.zero_grad(set_to_none=True)
#     logit.backward()
#     s = x.grad.detach().abs()[0].mean(dim=0)  # (H, W)
#     return s.cpu(), logits.detach().cpu()

# def saliency_fusion(model_f, x1, x2, target_idx, device):
#     """
#     Compute input-gradient saliency for both inputs.
#     Returns (sal1d(T,), sal2d(H,W), logits, modality_weights[2])
#     """
#     model_f.eval()
#     x1v = x1.clone().detach().requires_grad_(True).to(device)
#     x2v = x2.clone().detach().requires_grad_(True).to(device)
#     logits, gates = model_f(x1v, x2v, return_gates=True)  # gates softmax (B,2)
#     logit = logits[0, target_idx]
#     model_f.zero_grad(set_to_none=True)
#     logit.backward()
#     s1 = x1v.grad.detach().abs()[0].mean(dim=0).cpu()   # (T,)
#     s2 = x2v.grad.detach().abs()[0].mean(dim=0).cpu()   # (H,W)
#     return s1, s2, logits.detach().cpu(), gates[0].detach().cpu()


# # ---------- example picker ----------
# def find_example_for_class(
#     dm, m1d, m2d, mf_stable, mf_tuned, class_names, target_class_name,
#     device, max_scan=10000, require_all_correct=True
# ):
#     """Return a single (x1,x2,y,preds,gates_stable,gates_tuned) for target class."""
#     idx_of = {v: k for k, v in class_names.items()}
#     target_idx = idx_of[target_class_name]
#     scanned = 0

#     for xb1, xb2, yb in dm.test_loader:
#         b = xb1.size(0)
#         for i in range(b):
#             yi = int(yb[i].item())
#             if yi != target_idx:
#                 continue

#             x1 = xb1[i:i+1].to(device)
#             x2 = xb2[i:i+1].to(device)

#             with torch.no_grad():
#                 p1 = int(m1d(x1).argmax(1).item())
#                 p2 = int(m2d(x2).argmax(1).item())
#                 logits_st, gates_st = mf_stable(x1, x2, return_gates=True)
#                 logits_tn, gates_tn = mf_tuned(x1, x2, return_gates=True)
#                 pf_st = int(logits_st.argmax(1).item())
#                 pf_tn = int(logits_tn.argmax(1).item())

#             if require_all_correct:
#                 ok = (p1 == yi) and (p2 == yi) and (pf_st == yi) and (pf_tn == yi)
#             else:
#                 ok = (pf_st == yi) or (pf_tn == yi)

#             if ok:
#                 return dict(
#                     x1=xb1[i:i+1].cpu(), x2=xb2[i:i+1].cpu(), y=yi,
#                     preds=(p1, p2, pf_st, pf_tn),
#                     gates_stable=gates_st[0].cpu(), gates_tuned=gates_tn[0].cpu()
#                 )

#         scanned += b
#         if scanned >= max_scan:
#             break

#     return None


# # ---------- plotting helpers ----------
# def _bright_spec(spec, pct=(2, 98), gamma=0.7):
#     """Log + percentile clip + gamma for a crisp background."""
#     s = np.log10(np.maximum(spec, 1e-10))
#     vmin, vmax = np.percentile(s, pct)
#     s = np.clip(s, vmin, vmax)
#     s = (s - vmin) / (vmax - vmin + 1e-8)
#     s = np.power(s, gamma)
#     return s

# def plot_1d_saliency(ax, sig, saliency, time_axis, sigma=6, thr=0.75, title=""):
#     s = _minmax(smooth1d(saliency.numpy(), sigma=sigma))

#     # Main waveform
#     ax.plot(time_axis, sig, linewidth=1.1, color="#1f2937")  # dark gray

#     # Highlight contiguous high-saliency regions
#     cutoff = thr * (s.max() if s.max() > 0 else 1.0)
#     mask = s >= cutoff
#     if mask.any():
#         on = False; start = 0
#         for t in range(len(mask)):
#             if mask[t] and not on:
#                 on = True; start = t
#             if (not mask[t] and on) or (on and t == len(mask) - 1):
#                 end = t if not mask[t] else t + 1
#                 ax.axvspan(time_axis[start],
#                            time_axis[end - 1] if end > start else time_axis[end],
#                            color="#ef4444", alpha=0.28, lw=0)
#                 on = False

#     ax.set_xlim(time_axis[0], time_axis[-1])
#     ax.set_ylabel("1D", fontsize=9)
#     ax.set_xlabel("Time (s)", fontsize=9)
#     ax.grid(True, alpha=0.22, linewidth=0.6)
#     if title:
#         ax.set_title(title, fontsize=10, pad=6)

# def plot_2d_saliency(ax, spec, saliency, time_axis, fs, title=""):
#     H, W = spec.shape
#     base = _bright_spec(spec, pct=(2, 98), gamma=0.7)
#     ax.imshow(base, aspect='auto', origin='lower',
#               extent=[time_axis[0], time_axis[-1], 0, H],
#               cmap='magma', alpha=1.0)

#     hm = _minmax(saliency.numpy())
#     ax.imshow(hm, aspect='auto', origin='lower',
#               extent=[time_axis[0], time_axis[-1], 0, H],
#               cmap='turbo', alpha=0.55)

#     ax.set_ylabel("2D STFT\n(freq bins)", fontsize=9)
#     ax.set_xlabel("Time (s)", fontsize=9)
#     ax.set_yticks([0, H // 2, H])
#     ax.grid(False)
#     if title:
#         ax.set_title(title, fontsize=10, pad=6)

# def plot_weights(ax, gates, title=None):
#     w = gates.numpy() * 100.0
#     colors = ["#2563eb", "#ef4444"]  # 1D blue, 2D red
#     bars = ax.bar([0, 1], w, color=colors, edgecolor='black', linewidth=0.8)
#     for b, pct in zip(bars, w):
#         ax.text(b.get_x() + b.get_width() / 2, b.get_height() / 2,
#                 f"{pct:.0f}%", ha="center", va="center",
#                 fontsize=12, color="white", fontweight="bold")
#     ax.set_ylim(0, 100)
#     ax.set_xticks([0, 1]); ax.set_xticklabels(["1D", "2D"], fontsize=10)
#     ax.set_ylabel("Modality weight (%)", fontsize=9)
#     ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
#     ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.3)
#     if title:
#         ax.set_title(title, fontsize=10, pad=6)


# # ---------- main figure ----------
# def render_single_class(meta, sig, spec, sals, gates_st, gates_tn,
#                         out_path, fs=250, thr=0.75, sigma=6,
#                         class_names=None):
#     """
#     Build one figure: 4 columns (1D, 2D, Stable, Tuned) × 4 rows
#       Row 0: titles
#       Row 1: 1D plot
#       Row 2: 2D plot
#       Row 3: weights
#     """
#     p1, p2, pf_st, pf_tn = meta["preds"]
#     yi = meta["y"]
#     time_axis = np.arange(len(sig)) / fs

#     fig = plt.figure(figsize=(16, 9))
#     gs = gridspec.GridSpec(
#         nrows=4, ncols=4,
#         height_ratios=[0.18, 1.2, 1.2, 0.9],
#         hspace=0.42, wspace=0.30
#     )

#     # Column headers row
#     headers = ["1D Model", "2D Model", "Stable Fusion", "Tuned Fusion"]
#     for c, text in enumerate(headers):
#         axh = fig.add_subplot(gs[0, c]); axh.axis("off")
#         axh.text(0.5, 0.5, text, ha="center", va="center",
#                  fontsize=14, fontweight="bold", transform=axh.transAxes)

#     # Summary header (true/preds)
#     fig.suptitle(
#         f"True={class_names[yi]} | 1D={class_names[p1]} | 2D={class_names[p2]} | "
#         f"FUS_stable={class_names[pf_st]} | FUS_tuned={class_names[pf_tn]}",
#         fontsize=15, fontweight="bold", y=0.98
#     )

#     # Column 0: 1D-only
#     ax = fig.add_subplot(gs[1, 0])
#     plot_1d_saliency(ax, sig, sals["1d_solo"], time_axis, sigma=sigma, thr=thr,
#                      title=f"Pred: {class_names[p1]}")
#     ax = fig.add_subplot(gs[2, 0]); ax.axis("off")
#     ax.text(0.5, 0.5, "No 2D for 1D model", ha="center", va="center",
#             fontsize=10, color="#6b7280", transform=ax.transAxes)
#     ax = fig.add_subplot(gs[3, 0])
#     plot_weights(ax, gates=torch.tensor([1.0, 0.0]), title="Weights (forced)")

#     # Column 1: 2D-only
#     ax = fig.add_subplot(gs[1, 1]); ax.axis("off")
#     ax.text(0.5, 0.5, "No 1D for 2D model", ha="center", va="center",
#             fontsize=10, color="#6b7280", transform=ax.transAxes)
#     ax = fig.add_subplot(gs[2, 1])
#     plot_2d_saliency(ax, spec, sals["2d_solo"], time_axis, fs,
#                      title=f"Pred: {class_names[p2]}")
#     ax = fig.add_subplot(gs[3, 1])
#     plot_weights(ax, gates=torch.tensor([0.0, 1.0]), title="Weights (forced)")

#     # Column 2: Stable Fusion
#     ax = fig.add_subplot(gs[1, 2])
#     plot_1d_saliency(ax, sig, sals["1d_st"], time_axis, sigma=sigma, thr=thr,
#                      title=f"Pred: {class_names[pf_st]}")
#     ax = fig.add_subplot(gs[2, 2])
#     plot_2d_saliency(ax, spec, sals["2d_st"], time_axis, fs)
#     ax = fig.add_subplot(gs[3, 2])
#     plot_weights(ax, gates_st, title="Weights")

#     # Column 3: Tuned Fusion
#     ax = fig.add_subplot(gs[1, 3])
#     plot_1d_saliency(ax, sig, sals["1d_tn"], time_axis, sigma=sigma, thr=thr,
#                      title=f"Pred: {class_names[pf_tn]}")
#     ax = fig.add_subplot(gs[2, 3])
#     plot_2d_saliency(ax, spec, sals["2d_tn"], time_axis, fs)
#     ax = fig.add_subplot(gs[3, 3])
#     plot_weights(ax, gates_tn, title="Weights")

#     # Minimal legend
#     lg_ax = fig.add_axes([0.015, 0.01, 0.97, 0.08]); lg_ax.axis("off")
#     patch = mpatches.Patch(color="#ef4444", alpha=0.28, label="High-saliency regions")
#     lg_ax.legend(handles=[patch], loc="center", ncols=1, frameon=False, fontsize=11)

#     # Final spacing + save
#     plt.subplots_adjust(top=0.92, bottom=0.08, left=0.06, right=0.99)
#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print(f"✓ Saved: {out_path}")


# # ---------- CLI ----------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt-1d", required=True)
#     ap.add_argument("--ckpt-2d", required=True)
#     ap.add_argument("--ckpt-fusion-stable", required=True)
#     ap.add_argument("--ckpt-fusion-tuned", required=True)
#     ap.add_argument("--data-dir", required=True)
#     ap.add_argument("--repr", default="stft", choices=["stft", "mel", "cwt"])
#     ap.add_argument("--device", default="cuda")
#     ap.add_argument("--out", default="outputs/saliency/fig_single_class.png")
#     ap.add_argument("--sigma", type=float, default=6.0)
#     ap.add_argument("--thr", type=float, default=0.75)
#     ap.add_argument("--max-scan", type=int, default=8000)
#     ap.add_argument("--dim-1d", type=int, default=384)
#     ap.add_argument("--dim-2d", type=int, default=384)
#     ap.add_argument("--fs", type=int, default=250)
#     ap.add_argument("--focus-class", default="AFIB", choices=["SB", "SR", "AFIB", "GSVT"])
#     ap.add_argument("--allow-mismatch", action="store_true",
#                     help="accept example where not all four models are correct")
#     args = ap.parse_args()

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     class_names = {0: "SB", 1: "SR", 2: "AFIB", 3: "GSVT"}

#     print(f"Loading data from {args.data_dir} ...")
#     dm = ECGFusionDataModule(
#         data_dir=args.data_dir, batch_size=32, num_workers=4,
#         representation_type=args.repr
#     )
#     dm.setup()

#     print("Loading models ...")
#     model_1d = make_1d().to(device).eval()
#     model_2d = make_2d(representation_type=args.repr).to(device).eval()
#     load_exact(model_1d, args.ckpt_1d, device)
#     load_exact(model_2d, args.ckpt_2d, device)

#     fusion_stable = ECGFusionModelSimple(model_1d, model_2d,
#                                          dim_1d=args.dim_1d, dim_2d=args.dim_2d,
#                                          num_classes=4).to(device).eval()
#     load_exact(fusion_stable, args.ckpt_fusion_stable, device)

#     fusion_tuned = ECGFusionModelSimple(model_1d, model_2d,
#                                         dim_1d=args.dim_1d, dim_2d=args.dim_2d,
#                                         num_classes=4).to(device).eval()
#     load_exact(fusion_tuned, args.ckpt_fusion_tuned, device)

#     print(f"Searching for a {args.focus_class} example ...")
#     meta = find_example_for_class(
#         dm, model_1d, model_2d, fusion_stable, fusion_tuned,
#         class_names, args.focus_class, device,
#         max_scan=args.max_scan, require_all_correct=not args.allow_mismatch
#     )
#     if meta is None:
#         print("❌ No suitable example found. Try --allow-mismatch or increase --max-scan.")
#         sys.exit(1)

#     x1 = meta["x1"].to(device)
#     x2 = meta["x2"].to(device)
#     yi = meta["y"]
#     p1, p2, pf_st, pf_tn = meta["preds"]

#     print("Computing saliency maps ...")
#     # 1D model saliency at its own predicted class
#     s1d_solo, _ = saliency_1d(model_1d, x1, target_idx=p1, device=device)
#     # 2D model saliency at its own predicted class
#     s2d_solo, _ = saliency_2d(model_2d, x2, target_idx=p2, device=device)
#     # stable fusion saliency at its predicted class
#     s1d_st, s2d_st, _, gates_st = saliency_fusion(fusion_stable, x1, x2, target_idx=pf_st, device=device)
#     # tuned fusion saliency at its predicted class
#     s1d_tn, s2d_tn, _, gates_tn = saliency_fusion(fusion_tuned, x1, x2, target_idx=pf_tn, device=device)

#     # Build for rendering
#     sig = meta["x1"][0, 0].numpy()
#     spec = meta["x2"][0, 0].numpy()
#     sals = {
#         "1d_solo": s1d_solo,
#         "2d_solo": s2d_solo,
#         "1d_st":   s1d_st,
#         "2d_st":   s2d_st,
#         "1d_tn":   s1d_tn,
#         "2d_tn":   s2d_tn,
#     }

#     render_single_class(
#         meta, sig, spec, sals, meta["gates_stable"], meta["gates_tuned"],
#         out_path=args.out, fs=args.fs, thr=args.thr, sigma=args.sigma, class_names=class_names
#     )


# if __name__ == "__main__":
#     main()

