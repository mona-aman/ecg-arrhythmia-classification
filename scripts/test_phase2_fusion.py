#!/usr/bin/env python3
"""
Test Phase 2: Fusion Model Evaluation
Comprehensive testing with metrics, plots, and per-class analysis
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import sys

# Add project paths
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
for path in [project_root, src_dir, src_dir / 'data', src_dir / 'models', project_root / 'scripts']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def compute_specificity(y_true, y_pred, num_classes=4):
    cm = confusion_matrix(y_true, y_pred)
    specificity_per_class = []
    for i in range(num_classes):
        tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(specificity)
    return specificity_per_class


def compute_metrics(y_true, y_pred, y_proba, class_names, output_dir):
    num_classes = len(class_names)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    specificity_per_class = compute_specificity(y_true, y_pred, num_classes)

    cm = confusion_matrix(y_true, y_pred)

    # ROC AUC per class
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    roc_auc_per_class = []
    if y_proba.shape[1] == num_classes:
        for i in range(num_classes):
            if len(np.unique(y_true_bin[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc_per_class.append(auc(fpr, tpr))
            else:
                roc_auc_per_class.append(0.0)
        roc_auc = float(np.mean(roc_auc_per_class))
    else:
        roc_auc = 0.0
        roc_auc_per_class = [0.0] * num_classes

    specificity = float(np.mean(specificity_per_class))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "sensitivity": float(recall),
        "specificity": specificity,
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc),
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class,
        "sensitivity_per_class": recall_per_class,
        "specificity_per_class": specificity_per_class,
        "f1_per_class": f1_per_class,
        "auc_per_class": roc_auc_per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }

    with open(output_dir / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Print brief summary
    print(f"Test Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")

    return metrics


def plot_confusion_matrix(cm, class_names, output_dir):
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix (Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_counts.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_proba, class_names, output_dir):
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Multi-Class')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curves(y_true, y_proba, class_names, output_dir):
    num_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        if len(np.unique(y_true_bin[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_proba[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, lw=2, label=f'{class_names[i]} (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Multi-Class')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(metrics, class_names, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(class_names))
    width = 0.6
    axes[0, 0].bar(x, metrics['precision_per_class'], width, color='steelblue')
    axes[0, 0].set_title('Precision per Class')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(class_names, rotation=45)
    axes[0, 1].bar(x, metrics['recall_per_class'], width, color='coral')
    axes[0, 1].set_title('Recall per Class')
    axes[1, 0].bar(x, metrics['f1_per_class'], width, color='mediumseagreen')
    axes[1, 0].set_title('F1 per Class')
    axes[1, 1].bar(x, metrics['specificity_per_class'], width, color='mediumpurple')
    axes[1, 1].set_title('Specificity per Class')
    plt.tight_layout()
    plt.savefig(output_dir / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def test_fusion_model(model, test_loader, device, output_dir, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    print("\n[Testing Fusion Model]")
    with torch.no_grad():
        for x_1d, x_2d, labels in tqdm(test_loader, desc="Testing"):
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

    metrics = compute_metrics(all_labels, all_preds, all_probs, class_names, output_dir)
    plot_confusion_matrix(confusion_matrix(all_labels, all_preds), class_names, output_dir)
    plot_roc_curves(all_labels, all_probs, class_names, output_dir)
    plot_precision_recall_curves(all_labels, all_probs, class_names, output_dir)
    plot_per_class_metrics(metrics, class_names, output_dir)

    # Also save a compact test_results.json for compatibility
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'num_samples': int(len(all_labels))
        }, f, indent=2)

    return metrics


def _load_model_or_state(raw, model_factory, model_name: str):
    # Similar helper as in training script
    if isinstance(raw, torch.nn.Module):
        return raw
    state = None
    if isinstance(raw, dict):
        for key in ('model_state_dict', 'state_dict'):
            if key in raw:
                state = raw[key]
                break
        if state is None:
            state = raw
    if state is None:
        raise RuntimeError(f"Unable to interpret saved {model_name} file format")
    model = model_factory()
    new_state = {}
    for k, v in state.items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        new_state[new_k] = v
    try:
        model.load_state_dict(new_state)
    except Exception as e:
        print(f"Warning: strict load failed for {model_name}: {e}")
        model.load_state_dict(new_state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description='Test Phase 2 Fusion Model')
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--aligned-encoder-1d', type=str, required=True)
    parser.add_argument('--aligned-encoder-2d', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--feature-dim', type=int, default=384)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('='*70)
    print('PHASE 2: FUSION MODEL TESTING')
    print('='*70)
    class_names = ['Normal', 'AF', 'Other', 'Noisy']

    # Load data module
    try:
        from src.data.fusion_dataset import ECGFusionDataModule
    except Exception:
        from fusion_dataset import ECGFusionDataModule

    data_module = ECGFusionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        representation_type='cwt'
    )
    data_module.setup()
    test_loader = data_module.test_loader

    print(f"Test samples: {len(data_module.test_dataset)}")
    try:
        print(f"Test batches: {len(test_loader)}")
    except Exception:
        pass

    # Load encoders
    raw_enc_1d = torch.load(args.aligned_encoder_1d, map_location=device)
    raw_enc_2d = torch.load(args.aligned_encoder_2d, map_location=device)

    from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
    from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small

    encoder_1d = _load_model_or_state(raw_enc_1d, create_ecg_transformer_1d_small, '1D encoder')
    encoder_2d = _load_model_or_state(raw_enc_2d, create_ecg_transformer_2d_small, '2D encoder')

    # Import ECGFusionModel from train script
    from scripts.train_phase2_fusion import ECGFusionModel

    model = ECGFusionModel(
        encoder_1d=encoder_1d,
        encoder_2d=encoder_2d,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        freeze_encoders=True
    )

    # Load trained weights
    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and all(k.startswith('module.') or k in dict(model.named_parameters()) for k in checkpoint.keys()):
        model.load_state_dict(checkpoint)
    else:
        try:
            model.load_state_dict(checkpoint)
        except Exception:
            model = checkpoint

    model = model.to(device)

    # Test and save results
    metrics = test_fusion_model(model, test_loader, device, output_dir, class_names)

    print('\nTESTING COMPLETE')
    print(f'Results saved to {output_dir}')


if __name__ == '__main__':
    main()
