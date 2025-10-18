"""
Evaluation utilities for ECG Transformer models
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ECGEvaluator:
    """Comprehensive evaluation for ECG models"""
    
    def __init__(self, class_names: List[str] = None):
        if class_names is None:
            self.class_names = ['SB', 'AFIB', 'GSVT', 'SR']
        else:
            self.class_names = class_names
        
        self.num_classes = len(self.class_names)
    
    def evaluate_model(self, 
                      model: torch.nn.Module,
                      test_loader: torch.utils.data.DataLoader,
                      device: torch.device) -> Dict:
        """Comprehensive model evaluation"""
        
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                outputs = model(data)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        # Calculate comprehensive metrics
        results = self.calculate_metrics(all_targets, all_preds, all_probs)
        
        return results
    
    def calculate_metrics(self, 
                         targets: np.ndarray,
                         predictions: np.ndarray, 
                         probabilities: np.ndarray) -> Dict:
        """Calculate comprehensive classification metrics"""
        
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(targets, predictions)
        
        # Per-class and average metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            targets, predictions, average=None, zero_division=0
        )
        
        results['precision_per_class'] = precision.tolist()
        results['recall_per_class'] = recall.tolist()
        results['f1_per_class'] = f1.tolist()
        results['support_per_class'] = support.tolist()
        
        # Averaged metrics
        results['precision_macro'] = np.mean(precision)
        results['recall_macro'] = np.mean(recall)
        results['f1_macro'] = np.mean(f1)
        
        # Weighted metrics
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            targets, predictions, average='weighted', zero_division=0
        )
        results['precision_weighted'] = precision_w
        results['recall_weighted'] = recall_w
        results['f1_weighted'] = f1_w
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        results['confusion_matrix'] = cm.tolist()
        
        # AUC metrics
        try:
            # Multi-class AUC
            auc_macro = roc_auc_score(targets, probabilities, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(targets, probabilities, multi_class='ovr', average='weighted')
            
            results['auc_macro'] = auc_macro
            results['auc_weighted'] = auc_weighted
            
            # Per-class AUC
            targets_binarized = label_binarize(targets, classes=range(self.num_classes))
            auc_per_class = []
            
            for i in range(self.num_classes):
                if len(np.unique(targets_binarized[:, i])) > 1:
                    class_auc = roc_auc_score(targets_binarized[:, i], probabilities[:, i])
                    auc_per_class.append(class_auc)
                else:
                    auc_per_class.append(0.0)
            
            results['auc_per_class'] = auc_per_class
            
        except Exception as e:
            print(f"Warning: Could not calculate AUC metrics: {e}")
            results['auc_macro'] = 0.0
            results['auc_weighted'] = 0.0
            results['auc_per_class'] = [0.0] * self.num_classes
        
        # Classification report
        report = classification_report(
            targets, predictions, 
            target_names=self.class_names,
            output_dict=True,
            zero_division=0
        )
        results['classification_report'] = report
        
        return results
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            save_path: Optional[str] = None,
                            normalize: bool = False) -> plt.Figure:
        """Plot confusion matrix"""
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            cm = confusion_matrix
            title = 'Confusion Matrix'
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_roc_curves(self,
                       targets: np.ndarray,
                       probabilities: np.ndarray,
                       save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curves for multi-class classification"""
        
        # Binarize targets
        targets_bin = label_binarize(targets, classes=range(self.num_classes))
        
        # Compute ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(self.num_classes):
            if len(np.unique(targets_bin[:, i])) > 1:
                fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], probabilities[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            else:
                fpr[i] = np.array([0, 1])
                tpr[i] = np.array([0, 0])
                roc_auc[i] = 0.0
        
        # Compute micro-average ROC curve and AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(targets_bin.ravel(), probabilities.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple'])
        
        # Plot ROC curve for each class
        for i, color in zip(range(self.num_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        # Plot micro-average ROC curve
        ax.plot(fpr["micro"], tpr["micro"],
               label=f'Micro-average (AUC = {roc_auc["micro"]:.2f})',
               color='deeppink', linestyle=':', linewidth=4)
        
        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Multi-class ROC Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(self,
                              results: Dict,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot per-class metrics comparison"""
        
        metrics = ['Precision', 'Recall', 'F1-Score']
        precision = results['precision_per_class']
        recall = results['recall_per_class']
        f1 = results['f1_per_class']
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_evaluation_report(self,
                                 results: Dict,
                                 targets: np.ndarray,
                                 predictions: np.ndarray,
                                 probabilities: np.ndarray,
                                 output_dir: str,
                                 model_name: str = "ECG_Model"):
        """Generate comprehensive evaluation report"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / f'{model_name}_metrics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_serializable = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    results_serializable[key] = value.tolist()
                else:
                    results_serializable[key] = value
            json.dump(results_serializable, f, indent=2)
        
        # Save predictions
        np.save(output_dir / f'{model_name}_predictions.npy', predictions)
        np.save(output_dir / f'{model_name}_targets.npy', targets)
        np.save(output_dir / f'{model_name}_probabilities.npy', probabilities)
        
        # Generate plots
        cm = np.array(results['confusion_matrix'])
        
        # Confusion matrix
        self.plot_confusion_matrix(
            cm, 
            save_path=output_dir / f'{model_name}_confusion_matrix.png'
        )
        
        # Normalized confusion matrix
        self.plot_confusion_matrix(
            cm, 
            save_path=output_dir / f'{model_name}_confusion_matrix_normalized.png',
            normalize=True
        )
        
        # ROC curves
        self.plot_roc_curves(
            targets, probabilities,
            save_path=output_dir / f'{model_name}_roc_curves.png'
        )
        
        # Per-class metrics
        self.plot_per_class_metrics(
            results,
            save_path=output_dir / f'{model_name}_per_class_metrics.png'
        )
        
        # Create summary report
        self._create_summary_report(results, output_dir, model_name)
        
        print(f"Evaluation report saved to {output_dir}")
    
    def _create_summary_report(self, results: Dict, output_dir: Path, model_name: str):
        """Create a text summary report"""
        
        report_lines = []
        report_lines.append(f"EVALUATION REPORT: {model_name}")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        # Overall metrics
        report_lines.append("OVERALL PERFORMANCE:")
        report_lines.append(f"Accuracy: {results['accuracy']:.4f}")
        report_lines.append(f"Macro F1-Score: {results['f1_macro']:.4f}")
        report_lines.append(f"Weighted F1-Score: {results['f1_weighted']:.4f}")
        report_lines.append(f"Macro AUC: {results['auc_macro']:.4f}")
        report_lines.append(f"Weighted AUC: {results['auc_weighted']:.4f}")
        report_lines.append("")
        
        # Per-class performance
        report_lines.append("PER-CLASS PERFORMANCE:")
        report_lines.append(f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10} {'Support':<10}")
        report_lines.append("-" * 60)
        
        for i, class_name in enumerate(self.class_names):
            precision = results['precision_per_class'][i]
            recall = results['recall_per_class'][i]
            f1 = results['f1_per_class'][i]
            auc_score = results['auc_per_class'][i]
            support = results['support_per_class'][i]
            
            report_lines.append(
                f"{class_name:<8} {precision:<10.4f} {recall:<10.4f} "
                f"{f1:<10.4f} {auc_score:<10.4f} {support:<10}"
            )
        
        report_lines.append("")
        
        # Best and worst performing classes
        f1_scores = results['f1_per_class']
        best_class_idx = np.argmax(f1_scores)
        worst_class_idx = np.argmin(f1_scores)
        
        report_lines.append("CLASS PERFORMANCE SUMMARY:")
        report_lines.append(f"Best performing class: {self.class_names[best_class_idx]} (F1: {f1_scores[best_class_idx]:.4f})")
        report_lines.append(f"Worst performing class: {self.class_names[worst_class_idx]} (F1: {f1_scores[worst_class_idx]:.4f})")
        
        # Save report
        with open(output_dir / f'{model_name}_summary_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))


if __name__ == "__main__":
    # Test evaluator
    evaluator = ECGEvaluator()
    
    # Dummy data for testing
    n_samples = 100
    n_classes = 4
    
    targets = np.random.randint(0, n_classes, n_samples)
    predictions = np.random.randint(0, n_classes, n_samples)
    probabilities = np.random.rand(n_samples, n_classes)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)  # Normalize
    
    # Calculate metrics
    results = evaluator.calculate_metrics(targets, predictions, probabilities)
    
    print("Test evaluation:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['f1_macro']:.4f}")
    print(f"Weighted F1: {results['f1_weighted']:.4f}")
    
    # Generate test report
    evaluator.generate_evaluation_report(
        results, targets, predictions, probabilities,
        output_dir="test_evaluation",
        model_name="Test_Model"
    )