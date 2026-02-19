#!/usr/bin/env python3
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def smooth_series(y, sg_window=5, sg_poly=2, ma_window=0):
    y = np.asarray(y, dtype=float)
    # Savitzky–Golay (only if long enough)
    if sg_window >= 3 and sg_window % 2 == 1 and len(y) >= sg_window:
        y_sg = savgol_filter(y, window_length=min(sg_window, len(y)-(1-len(y)%2)), polyorder=min(sg_poly, 3), mode="interp")
    else:
        y_sg = y.copy()
    # Moving average (optional)
    if ma_window and ma_window > 1:
        k = min(ma_window, len(y_sg))
        kernel = np.ones(k, dtype=float)/k
        y_ma = np.convolve(y_sg, kernel, mode="same")
    else:
        y_ma = y_sg
    return y_ma

def tight_ylim(y, pad=0.5, is_accuracy=False):
    y = np.asarray(y)
    lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
    if is_accuracy:
        # accuracy often needs very tight range
        pad_val = max(0.1, (hi - lo) * 0.15)
        return (lo - pad_val, hi + pad_val)
    return (lo - pad, hi + pad)

def plot_curves(hist, out_dir: Path, sg=7, poly=2, ma=0):
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loss = hist.get("train_loss", [])
    val_loss   = hist.get("val_loss",   [])
    train_acc  = hist.get("train_acc",  [])
    val_acc    = hist.get("val_acc",    [])
    val_f1     = hist.get("val_f1",     [])

    n = min(len(train_loss), len(val_loss), len(train_acc), len(val_acc))
    if n == 0:
        raise SystemExit("History is empty or keys missing.")
    # keep common prefix length to avoid mismatched ends
    train_loss, val_loss = train_loss[:n], val_loss[:n]
    train_acc,  val_acc  = train_acc[:n],  val_acc[:n]
    val_f1               = val_f1[:n] if len(val_f1) >= n else [np.nan]*n

    epochs = np.arange(n)

    # Smooth
    s_train_loss = smooth_series(train_loss, sg, poly, ma)
    s_val_loss   = smooth_series(val_loss,   sg, poly, ma)
    s_train_acc  = smooth_series(train_acc,  sg, poly, ma)
    s_val_acc    = smooth_series(val_acc,    sg, poly, ma)
    s_val_f1     = smooth_series(val_f1,     sg, poly, ma) if np.isfinite(val_f1).any() else None

    # Best val acc epoch
    best_idx = int(np.nanargmax(val_acc))
    best_ep  = epochs[best_idx]

    # ----- Loss
    plt.figure(figsize=(9,6))
    plt.plot(epochs, s_train_loss, label="train_loss")
    plt.plot(epochs, s_val_loss,   label="val_loss")
    plt.scatter([best_ep], [s_val_loss[best_idx]], s=40, marker="*", zorder=5, label="best val acc epoch")
    lo, hi = tight_ylim(np.r_[s_train_loss, s_val_loss], pad=0.01)
    plt.ylim(lo, hi)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"loss_clean.png", dpi=300); plt.close()

    # ----- Accuracy
    plt.figure(figsize=(9,6))
    plt.plot(epochs, s_train_acc, label="train_acc (smoothed)")
    plt.plot(epochs, s_val_acc,   label="val_acc (smoothed)")
    plt.scatter([best_ep], [s_val_acc[best_idx]], s=40, marker="*", zorder=5, label="best val acc epoch")
    lo, hi = tight_ylim(np.r_[s_train_acc, s_val_acc], is_accuracy=True)
    plt.ylim(lo, hi)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy (Smoothed)")
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir/"accuracy_clean.png", dpi=300); plt.close()

    # ----- F1 (if present)
    if s_val_f1 is not None and np.isfinite(s_val_f1).any():
        plt.figure(figsize=(9,6))
        plt.plot(epochs, s_val_f1, label="val_f1 (smoothed)")
        lo, hi = tight_ylim(s_val_f1, pad=0.01)
        plt.ylim(lo, hi)
        plt.xlabel("Epoch"); plt.ylabel("Weighted F1"); plt.title("F1 (Smoothed)")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir/"f1_clean.png", dpi=300); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to training_history.json")
    ap.add_argument("--out", required=True, help="Output folder for plots")
    ap.add_argument("--sg-window", type=int, default=7, help="Savitzky–Golay window (odd)")
    ap.add_argument("--sg-poly", type=int, default=2, help="Savitzky–Golay polyorder")
    ap.add_argument("--ma-window", type=int, default=0, help="Optional moving-average window")
    args = ap.parse_args()

    hist = json.loads(Path(args.history).read_text())
    plot_curves(hist, Path(args.out), sg=args.sg_window, poly=args.sg_poly, ma=args.ma_window)

if __name__ == "__main__":
    main()

# # save as scripts/plot_history_smooth.py and run:
# # python scripts/plot_history_smoothed.py /ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/outputs/fusion_stft_tuned/training_history.json

# import json, sys
# from pathlib import Path
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

# hist_path = Path(sys.argv[1])
# out_dir = hist_path.parent

# with open(hist_path) as f:
#     H = json.load(f)

# def ema(x, alpha=0.2):
#     y = np.zeros_like(x, dtype=float)
#     if len(x)==0: return y
#     y[0] = x[0]
#     for i in range(1,len(x)):
#         y[i] = alpha*x[i] + (1-alpha)*y[i-1]
#     return y

# def smooth(y):
#     y = np.array(y, dtype=float)
#     y_ema = ema(y, 0.2)
#     if len(y) >= 5:
#         y_sg = savgol_filter(y_ema, window_length=5, polyorder=2, mode="interp")
#     else:
#         y_sg = y_ema
#     return y_sg

# keys = ["train_loss","val_loss","train_acc","val_acc"]
# for k in keys:
#     if k not in H: 
#         raise SystemExit(f"Missing key {k} in history.")

# L_tr = smooth(H["train_loss"])
# L_va = smooth(H["val_loss"])
# A_tr = smooth(H["train_acc"])
# A_va = smooth(H["val_acc"])

# # Accuracy
# plt.figure(figsize=(12,8))
# plt.plot(A_tr, label="train_acc (smoothed)")
# plt.plot(A_va, label="val_acc (smoothed)")
# plt.title("Accuracy (Smoothed)")
# plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.legend(); plt.tight_layout()
# plt.savefig(out_dir/"accuracy_smoothed_v2.png", dpi=300); plt.close()

# # Loss
# plt.figure(figsize=(12,8))
# plt.plot(L_tr, label="train_loss (smoothed)")
# plt.plot(L_va, label="val_loss (smoothed)")
# plt.title("Loss (Smoothed)")
# plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.tight_layout()
# plt.savefig(out_dir/"loss_smoothed_v2.png", dpi=300); plt.close()

# # Optional: F1 if present
# if "val_f1" in H:
#     F1 = smooth(H["val_f1"])
#     plt.figure(figsize=(12,8))
#     plt.plot(F1, label="val_f1 (smoothed)")
#     plt.title("F1 (Smoothed)")
#     plt.xlabel("Epoch"); plt.ylabel("Weighted F1"); plt.legend(); plt.tight_layout()
#     plt.savefig(out_dir/"f1_smoothed_v2.png", dpi=300); plt.close()

# print("Saved:", out_dir/"accuracy_smoothed_v2.png", out_dir/"loss_smoothed_v2.png")

# # #!/usr/bin/env python3
# # import argparse, json
# # from pathlib import Path

# # import numpy as np
# # import matplotlib
# # matplotlib.use("Agg")
# # import matplotlib.pyplot as plt

# # try:
# #     # optional, for Savitzky–Golay smoothing
# #     from scipy.signal import savgol_filter
# #     HAVE_SAVGOL = True
# # except Exception:
# #     HAVE_SAVGOL = False


# # def moving_average(y, window=5):
# #     if window <= 1 or window is None:
# #         return np.asarray(y, dtype=float)
# #     window = int(window)
# #     if window > len(y):
# #         window = max(1, len(y) // 2 * 2 + 1)  # odd, <= len
# #     kernel = np.ones(window) / window
# #     return np.convolve(y, kernel, mode="same")


# # def smooth(y, method="movavg", window=5, poly=2):
# #     y = np.asarray(y, dtype=float)
# #     if method == "savgol":
# #         if not HAVE_SAVGOL:
# #             # fallback quietly if scipy not available
# #             return moving_average(y, window)
# #         # Savitzky–Golay requires odd window >= poly+2
# #         w = int(window) if int(window) % 2 == 1 else int(window) + 1
# #         w = max(w, poly + 3 if (poly + 3) % 2 == 1 else poly + 4)
# #         if w > len(y):
# #             w = len(y) if len(y) % 2 == 1 else len(y) - 1
# #         w = max(3, w)
# #         return savgol_filter(y, window_length=w, polyorder=min(poly, w - 1))
# #     # default: moving average
# #     return moving_average(y, window)


# # def plot_series(xs, series_dict, title, ylabel, outpath):
# #     plt.figure(figsize=(14, 10))
# #     for label, y in series_dict.items():
# #         plt.plot(xs, y, label=label)
# #     plt.title(title)
# #     plt.xlabel("Epoch")
# #     plt.ylabel(ylabel)
# #     plt.legend()
# #     plt.tight_layout()
# #     plt.savefig(outpath, dpi=300)
# #     plt.close()


# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--input", required=True, help="Path to training_history.json")
# #     ap.add_argument("--outdir", required=True, help="Where to save the figures")
# #     ap.add_argument("--method", choices=["movavg", "savgol"], default="movavg",
# #                     help="Smoothing method (default: movavg)")
# #     ap.add_argument("--window", type=int, default=5, help="Window size for smoothing (default: 5)")
# #     ap.add_argument("--poly", type=int, default=2, help="Poly order for Savitzky–Golay (default: 2)")
# #     args = ap.parse_args()

# #     in_path = Path(args.input)
# #     out_dir = Path(args.outdir)
# #     out_dir.mkdir(parents=True, exist_ok=True)

# #     with open(in_path, "r") as f:
# #         hist = json.load(f)

# #     # Expected keys (gracefully handle missing)
# #     train_loss = hist.get("train_loss", [])
# #     val_loss   = hist.get("val_loss", [])
# #     train_acc  = hist.get("train_acc", [])
# #     val_acc    = hist.get("val_acc", [])
# #     val_f1     = hist.get("val_f1", [])

# #     # convert to arrays + smooth
# #     xs = np.arange(len(train_loss))

# #     s_train_loss = smooth(train_loss, method=args.method, window=args.window, poly=args.poly)
# #     s_val_loss   = smooth(val_loss,   method=args.method, window=args.window, poly=args.poly)

# #     s_train_acc = smooth(train_acc, method=args.method, window=args.window, poly=args.poly)
# #     s_val_acc   = smooth(val_acc,   method=args.method, window=args.window, poly=args.poly)

# #     s_val_f1 = smooth(val_f1, method=args.method, window=args.window, poly=args.poly)

# #     # Plot 1: Loss
# #     plot_series(
# #         xs,
# #         {"train_loss (smoothed)": s_train_loss, "val_loss (smoothed)": s_val_loss},
# #         title="Loss (Smoothed)",
# #         ylabel="Loss",
# #         outpath=out_dir / "loss_smoothed.png"
# #     )

# #     # Plot 2: Accuracy
# #     plot_series(
# #         xs,
# #         {"train_acc (smoothed)": s_train_acc, "val_acc (smoothed)": s_val_acc},
# #         title="Accuracy (Smoothed)",
# #         ylabel="Accuracy (%)",
# #         outpath=out_dir / "accuracy_smoothed.png"
# #     )

# #     # Plot 3: F1
# #     xsf1 = np.arange(len(val_f1))
# #     plot_series(
# #         xsf1,
# #         {"val_f1 (smoothed)": s_val_f1},
# #         title="Validation F1 (Smoothed)",
# #         ylabel="Weighted F1",
# #         outpath=out_dir / "f1_smoothed.png"
# #     )

# #     print(f"[OK] Saved:")
# #     print(f" - {out_dir / 'loss_smoothed.png'}")
# #     print(f" - {out_dir / 'accuracy_smoothed.png'}")
# #     print(f" - {out_dir / 'f1_smoothed.png'}")


# # if __name__ == "__main__":
# #     main()
