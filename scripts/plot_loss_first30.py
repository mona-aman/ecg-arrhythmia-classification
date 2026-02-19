# save as: scripts/plot_loss_first30.py
import json
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def moving_average(x, k):
    """Centered moving average of length k (odd), with edge reflection."""
    if k <= 1 or len(x) == 0:
        return np.array(x, dtype=float)
    if k % 2 == 0:
        k += 1  # make odd for centering
    pad = k // 2
    x = np.asarray(x, dtype=float)
    # reflect-pad
    left  = x[1:pad+1][::-1] if pad > 0 else np.array([])
    right = x[-pad-1:-1][::-1] if pad > 0 else np.array([])
    xp = np.concatenate([left, x, right])
    w = np.ones(k) / k
    y = np.convolve(xp, w, mode="valid")
    return y  # same length as x

def ema(x, alpha):
    """Exponential moving average; alpha in (0,1]."""
    if alpha >= 1 or alpha <= 0 or len(x) == 0:
        return np.array(x, dtype=float)
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to training_history.json")
    ap.add_argument("--out", default=None, help="Output PNG path (default: next to history)")
    ap.add_argument("--epochs", type=int, default=30, help="Max epoch index to display (default: 30)")
    # smoothing options
    ap.add_argument("--smooth", choices=["ma", "ema", "none"], default="ma",
                    help="Smoothing type for overlay (moving average, EMA, or none)")
    ap.add_argument("--window", type=int, default=5, help="Window size for moving average (odd)")
    ap.add_argument("--alpha", type=float, default=0.2, help="Alpha for EMA (0<alpha<=1)")
    args = ap.parse_args()

    hist_path = Path(args.history)
    hist = json.loads(hist_path.read_text())

    train_loss = hist.get("train_loss", [])
    val_loss   = hist.get("val_loss", [])
    val_acc    = hist.get("val_acc", [])

    n = min(len(train_loss), len(val_loss))
    train_loss = np.array(train_loss[:n], dtype=float)
    val_loss   = np.array(val_loss[:n],   dtype=float)
    val_acc    = np.array(val_acc[:n],    dtype=float) if len(val_acc) >= n else np.full(n, np.nan)

    N = min(args.epochs, n)
    if N == 0:
        raise SystemExit("No epochs found in history.")

    # best raw val loss in first N epochs
    sub_val = val_loss[:N]
    best_idx = int(np.nanargmin(sub_val))
    best_val_loss = float(sub_val[best_idx])
    best_val_acc  = float(val_acc[best_idx]) if not np.isnan(val_acc[best_idx]) else float("nan")

    # smoothing overlays (computed on first N only)
    if args.smooth == "ma":
        train_s = moving_average(train_loss[:N], args.window)
        val_s   = moving_average(val_loss[:N],   args.window)
        sm_label = f"(smoothed, MA k={args.window})"
    elif args.smooth == "ema":
        train_s = ema(train_loss[:N], args.alpha)
        val_s   = ema(val_loss[:N],   args.alpha)
        sm_label = f"(smoothed, EMA α={args.alpha})"
    else:
        train_s = None
        val_s = None
        sm_label = None

    # plot
    epochs = np.arange(N)
    plt.figure(figsize=(9,6))
    # raw
    plt.plot(epochs, train_loss[:N], label="train_loss (raw)", alpha=0.6)
    plt.plot(epochs, val_loss[:N],   label="val_loss (raw)",  alpha=0.6)
    # smoothed overlays
    if train_s is not None:
        plt.plot(epochs, train_s, label=f"train_loss {sm_label}", linewidth=2.0)
        plt.plot(epochs, val_s,   label=f"val_loss {sm_label}",   linewidth=2.0)

    # mark best raw val point
    plt.scatter([best_idx], [best_val_loss], s=70, marker="*", zorder=5, label="best val (raw)")
    plt.axvline(best_idx, linestyle="--", alpha=0.35)

    # annotate
    plt.annotate(
        f"best @ epoch {best_idx}\nval_loss={best_val_loss:.4f}\nval_acc={best_val_acc:.2f}%",
        xy=(best_idx, best_val_loss),
        xytext=(best_idx + 0.6, best_val_loss),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85),
        arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7)
    )

    # limits
    all_y = np.r_[train_loss[:N], val_loss[:N]]
    ylo, yhi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
    pad = max(0.02, (yhi - ylo) * 0.15)
    plt.ylim(ylo - pad, yhi + pad)
    plt.xlim(-0.5, max(epochs)+0.5)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss (first {N} epochs) with best raw val loss marked")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out) if args.out else (hist_path.parent / "loss_first30_best_smoothed.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to: {out_path}")
    print(f"Best (first {N} epochs) at epoch {best_idx}: val_loss={best_val_loss:.6f}, val_acc={best_val_acc:.2f}%")

if __name__ == "__main__":
    main()

# # save as: scripts/plot_loss_first30.py
# import json
# from pathlib import Path
# import argparse
# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--history", required=True,
#                     help="Path to training_history.json")
#     ap.add_argument("--out", default=None,
#                     help="Output PNG path (optional). If omitted, saved next to history.")
#     ap.add_argument("--epochs", type=int, default=30,
#                     help="Max epoch index to display (default: 30)")
#     args = ap.parse_args()

#     hist_path = Path(args.history)
#     hist = json.loads(hist_path.read_text())

#     train_loss = hist.get("train_loss", [])
#     val_loss   = hist.get("val_loss", [])
#     val_acc    = hist.get("val_acc", [])

#     # Ensure equal length prefix
#     n = min(len(train_loss), len(val_loss))
#     train_loss = train_loss[:n]
#     val_loss   = val_loss[:n]
#     val_acc    = val_acc[:n] if len(val_acc) >= n else [np.nan]*n

#     # Limit to first N epochs (0..N-1)
#     N = min(args.epochs, n)
#     if N == 0:
#         raise SystemExit("No epochs found in history.")

#     # Argmin on val loss within [0, N)
#     sub_val_loss = np.array(val_loss[:N], dtype=float)
#     best_idx = int(np.nanargmin(sub_val_loss))
#     best_val_loss = float(sub_val_loss[best_idx])
#     best_val_acc  = float(val_acc[best_idx]) if not np.isnan(val_acc[best_idx]) else float("nan")

#     # Plot
#     epochs = np.arange(N)
#     plt.figure(figsize=(9,6))
#     plt.plot(epochs, train_loss[:N], label="train_loss")
#     plt.plot(epochs, val_loss[:N],   label="val_loss")
#     # mark best point on val curve
#     plt.scatter([best_idx], [best_val_loss], s=60, marker="*", zorder=5, label="best val loss")
#     # vertical helper line
#     plt.axvline(best_idx, linestyle="--", alpha=0.4)

#     # annotation
#     plt.annotate(
#         f"best @ epoch {best_idx}\nval_loss={best_val_loss:.4f}\nval_acc={best_val_acc:.2f}%",
#         xy=(best_idx, best_val_loss),
#         xytext=(best_idx+0.5, best_val_loss),
#         textcoords="data",
#         fontsize=10,
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
#         arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7)
#     )

#     plt.xlim(-0.5, max(epochs)+0.5)
#     # tighten y for readability
#     all_y = np.r_[train_loss[:N], val_loss[:N]]
#     ylo, yhi = float(np.nanmin(all_y)), float(np.nanmax(all_y))
#     pad = max(0.02, (yhi - ylo) * 0.15)
#     plt.ylim(ylo - pad, yhi + pad)

#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title(f"Loss (first {N} epochs) with best val loss marked")
#     plt.legend()
#     plt.tight_layout()

#     out_path = Path(args.out) if args.out else (hist_path.parent / "loss_first30_best.png")
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(out_path, dpi=300)
#     print(f"Saved figure to: {out_path}")
#     print(f"Best (first {N} epochs) at epoch {best_idx}: val_loss={best_val_loss:.6f}, val_acc={best_val_acc:.2f}%")

# if __name__ == "__main__":
#     main()
