"""
Statistical Significance Testing for ECG Arrhythmia Classification Models
Comparing Fusion Model (Phase 2, Epoch 27) vs 1D and 2D Models
"""

import json
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd

def load_results(filepath):
    """Load test results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def compute_mcnemar_test(y_true, pred_model1, pred_model2):
    """
    McNemar's test for paired predictions
    Tests if two models have significantly different error rates
    """
    # Create contingency table
    n01 = np.sum((pred_model1 == y_true) & (pred_model2 != y_true))  # Model 1 correct, Model 2 wrong
    n10 = np.sum((pred_model1 != y_true) & (pred_model2 == y_true))  # Model 1 wrong, Model 2 correct
    
    # Create 2x2 contingency table for McNemar's test
    # Format: [[n_both_correct, n_model1_correct_model2_wrong],
    #          [n_model1_wrong_model2_correct, n_both_wrong]]
    n_both_correct = np.sum((pred_model1 == y_true) & (pred_model2 == y_true))
    n_both_wrong = np.sum((pred_model1 != y_true) & (pred_model2 != y_true))
    table = np.array([[n_both_correct, n01], [n10, n_both_wrong]])
    
    # Perform McNemar's test with continuity correction
    result = mcnemar(table, exact=False, correction=True)
    
    return result.statistic, result.pvalue, n01, n10

def bootstrap_ci(y_true, y_pred, metric_func, n_bootstrap=10000, ci=95):
    """
    Bootstrap confidence intervals for a given metric
    """
    scores = []
    n_samples = len(y_true)
    
    np.random.seed(42)  # For reproducibility
    for _ in range(n_bootstrap):
        indices = resample(range(n_samples), n_samples=n_samples)
        score = metric_func(y_true[indices], y_pred[indices])
        scores.append(score)
    
    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)
    mean = np.mean(scores)
    
    return mean, lower, upper

def reconstruct_predictions_from_confusion_matrix(confusion_matrix):
    """
    Reconstruct predictions and true labels from confusion matrix
    Note: This gives us the correct counts but loses the original order
    """
    n_classes = len(confusion_matrix)
    y_true = []
    y_pred = []
    
    for true_class in range(n_classes):
        for pred_class in range(n_classes):
            count = confusion_matrix[true_class][pred_class]
            y_true.extend([true_class] * count)
            y_pred.extend([pred_class] * count)
    
    return np.array(y_true), np.array(y_pred)

def compare_models(results1, results2, model1_name, model2_name):
    """
    Compare two models using statistical tests
    """
    print(f"\n{'='*80}")
    print(f"Comparing {model1_name} vs {model2_name}")
    print(f"{'='*80}\n")
    
    # Reconstruct predictions from confusion matrices
    cm1 = results1['confusion_matrix']
    cm2 = results2['confusion_matrix']
    
    y_true1, y_pred1 = reconstruct_predictions_from_confusion_matrix(cm1)
    y_true2, y_pred2 = reconstruct_predictions_from_confusion_matrix(cm2)
    
    # Verify they have the same ground truth (they should)
    assert len(y_true1) == len(y_true2), "Different number of samples!"
    assert np.array_equal(y_true1, y_true2), "Ground truth mismatch!"
    
    y_true = y_true1
    n_samples = len(y_true)
    
    print(f"Total test samples: {n_samples}")
    print(f"\nModel Performance:")
    print(f"  {model1_name}:")
    print(f"    Accuracy: {results1['accuracy']:.4f} ({results1['accuracy']*100:.2f}%)")
    print(f"    F1-Score: {results1['f1']:.4f}")
    print(f"    Precision: {results1['precision']:.4f}")
    print(f"    Recall: {results1['recall']:.4f}")
    print(f"    AUC: {results1['auc']:.4f}")
    
    print(f"\n  {model2_name}:")
    print(f"    Accuracy: {results2['accuracy']:.4f} ({results2['accuracy']*100:.2f}%)")
    print(f"    F1-Score: {results2['f1']:.4f}")
    print(f"    Precision: {results2['precision']:.4f}")
    print(f"    Recall: {results2['recall']:.4f}")
    print(f"    AUC: {results2['auc']:.4f}")
    
    print(f"\n  Difference ({model1_name} - {model2_name}):")
    print(f"    Accuracy: {(results1['accuracy'] - results2['accuracy'])*100:.2f} percentage points")
    print(f"    F1-Score: {(results1['f1'] - results2['f1']):.4f}")
    
    # McNemar's Test
    print(f"\n{'-'*80}")
    print("McNemar's Test (Testing for significant difference in error rates)")
    print(f"{'-'*80}")
    
    statistic, pvalue, n01, n10 = compute_mcnemar_test(y_true, y_pred1, y_pred2)
    
    print(f"  Contingency Table:")
    print(f"    {model1_name} correct, {model2_name} wrong: {n01}")
    print(f"    {model1_name} wrong, {model2_name} correct: {n10}")
    print(f"    Total disagreements: {n01 + n10}")
    print(f"\n  Test Results:")
    print(f"    Chi-square statistic: {statistic:.4f}")
    print(f"    p-value: {pvalue:.6f}")
    
    if pvalue < 0.001:
        significance = "*** (p < 0.001) - Highly significant"
    elif pvalue < 0.01:
        significance = "** (p < 0.01) - Very significant"
    elif pvalue < 0.05:
        significance = "* (p < 0.05) - Significant"
    else:
        significance = "ns (p >= 0.05) - Not significant"
    
    print(f"    Significance: {significance}")
    
    if pvalue < 0.05:
        if n01 > n10:
            print(f"    Interpretation: {model1_name} is significantly BETTER than {model2_name}")
        else:
            print(f"    Interpretation: {model2_name} is significantly BETTER than {model1_name}")
    else:
        print(f"    Interpretation: No significant difference between models")
    
    # Bootstrap Confidence Intervals
    print(f"\n{'-'*80}")
    print("Bootstrap Confidence Intervals (95% CI, 10,000 iterations)")
    print(f"{'-'*80}")
    
    metrics = {
        'Accuracy': accuracy_score,
        'F1-Score': lambda y_t, y_p: f1_score(y_t, y_p, average='weighted', zero_division=0),
        'Precision': lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
        'Recall': lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0)
    }
    
    print(f"\n  {model1_name}:")
    for metric_name, metric_func in metrics.items():
        mean, lower, upper = bootstrap_ci(y_true, y_pred1, metric_func)
        print(f"    {metric_name}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
    
    print(f"\n  {model2_name}:")
    for metric_name, metric_func in metrics.items():
        mean, lower, upper = bootstrap_ci(y_true, y_pred2, metric_func)
        print(f"    {metric_name}: {mean:.4f} (95% CI: [{lower:.4f}, {upper:.4f}])")
    
    return {
        'model1': model1_name,
        'model2': model2_name,
        'mcnemar_statistic': statistic,
        'mcnemar_pvalue': pvalue,
        'n01': n01,
        'n10': n10,
        'significant': pvalue < 0.05
    }

def print_per_class_comparison(results_fusion, results_1d, results_2d):
    """Print per-class metrics comparison"""
    print(f"\n{'='*80}")
    print("Per-Class Performance Comparison")
    print(f"{'='*80}\n")
    
    class_names = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)']
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Metric           Fusion      1D-CNN      2D-CNN")
        print(f"  {'-'*55}")
        print(f"  F1-Score       {results_fusion['f1_per_class'][i]:.4f}    {results_1d['f1_per_class'][i]:.4f}    {results_2d['f1_per_class'][i]:.4f}")
        print(f"  Precision      {results_fusion['precision_per_class'][i]:.4f}    {results_1d['precision_per_class'][i]:.4f}    {results_2d['precision_per_class'][i]:.4f}")
        print(f"  Recall         {results_fusion['recall_per_class'][i]:.4f}    {results_1d['recall_per_class'][i]:.4f}    {results_2d['recall_per_class'][i]:.4f}")
        print(f"  AUC            {results_fusion['auc_per_class'][i]:.4f}    {results_1d['auc_per_class'][i]:.4f}    {results_2d['auc_per_class'][i]:.4f}")

def main():
    """Main function to run all statistical tests"""
    
    # File paths
    fusion_path = '/ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/outputs/phase2_fusion_tuned_headonly_Lastresults/test_results.json'
    cnn1d_path = '/ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/outputs/1d_small_Lastresults/test_results.json'
    cnn2d_path = '/ocean/projects/cis250085p/shared/ecg-arrhythmia-classification/outputs/2d_stft_small_Lastresults/test_results.json'
    
    # Load results
    print("Loading test results...")
    results_fusion = load_results(fusion_path)
    results_1d = load_results(cnn1d_path)
    results_2d = load_results(cnn2d_path)
    
    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTING FOR ECG ARRHYTHMIA CLASSIFICATION")
    print("Fusion Model (Phase 2, Epoch 27) vs 1D-CNN and 2D-CNN Models")
    print(f"{'='*80}")
    
    # Compare Fusion vs 1D-CNN
    comparison1 = compare_models(results_fusion, results_1d, 
                                 "Fusion Model", "1D-CNN")
    
    # Compare Fusion vs 2D-CNN
    comparison2 = compare_models(results_fusion, results_2d, 
                                 "Fusion Model", "2D-CNN")
    
    # Compare 1D-CNN vs 2D-CNN (for completeness)
    comparison3 = compare_models(results_1d, results_2d, 
                                 "1D-CNN", "2D-CNN")
    
    # Per-class comparison
    print_per_class_comparison(results_fusion, results_1d, results_2d)
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY OF STATISTICAL TESTS")
    print(f"{'='*80}\n")
    
    comparisons = [comparison1, comparison2, comparison3]
    
    for comp in comparisons:
        print(f"\n{comp['model1']} vs {comp['model2']}:")
        print(f"  McNemar's test p-value: {comp['mcnemar_pvalue']:.6f}")
        print(f"  Statistically significant: {'Yes' if comp['significant'] else 'No'}")
        if comp['significant']:
            if comp['n01'] > comp['n10']:
                print(f"  Winner: {comp['model1']} (performs better on {comp['n01']} more samples)")
            else:
                print(f"  Winner: {comp['model2']} (performs better on {comp['n10']} more samples)")
    
    # Key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}\n")
    
    print(f"1. Overall Performance:")
    print(f"   - Fusion Model: {results_fusion['accuracy']*100:.2f}% accuracy, {results_fusion['f1']:.4f} F1")
    print(f"   - 1D-CNN Model: {results_1d['accuracy']*100:.2f}% accuracy, {results_1d['f1']:.4f} F1")
    print(f"   - 2D-CNN Model: {results_2d['accuracy']*100:.2f}% accuracy, {results_2d['f1']:.4f} F1")
    
    print(f"\n2. Statistical Significance:")
    if comparison1['significant']:
        print(f"   - Fusion vs 1D-CNN: Significant difference (p={comparison1['mcnemar_pvalue']:.6f})")
    else:
        print(f"   - Fusion vs 1D-CNN: No significant difference (p={comparison1['mcnemar_pvalue']:.6f})")
    
    if comparison2['significant']:
        print(f"   - Fusion vs 2D-CNN: Significant difference (p={comparison2['mcnemar_pvalue']:.6f})")
    else:
        print(f"   - Fusion vs 2D-CNN: No significant difference (p={comparison2['mcnemar_pvalue']:.6f})")
    
    print(f"\n3. Interpretation:")
    acc_diff_1d = (results_fusion['accuracy'] - results_1d['accuracy']) * 100
    acc_diff_2d = (results_fusion['accuracy'] - results_2d['accuracy']) * 100
    
    if abs(acc_diff_1d) < 0.5:
        print(f"   - Fusion and 1D-CNN have very similar performance (diff: {acc_diff_1d:.2f}%)")
    elif acc_diff_1d > 0:
        print(f"   - Fusion outperforms 1D-CNN by {acc_diff_1d:.2f} percentage points")
    else:
        print(f"   - 1D-CNN outperforms Fusion by {abs(acc_diff_1d):.2f} percentage points")
    
    if abs(acc_diff_2d) < 0.5:
        print(f"   - Fusion and 2D-CNN have very similar performance (diff: {acc_diff_2d:.2f}%)")
    elif acc_diff_2d > 0:
        print(f"   - Fusion outperforms 2D-CNN by {acc_diff_2d:.2f} percentage points")
    else:
        print(f"   - 2D-CNN outperforms Fusion by {abs(acc_diff_2d):.2f} percentage points")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
