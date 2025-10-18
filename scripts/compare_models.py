"""
Compare 1D vs 2D ECG Transformers
Run both architectures and compare their performance
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import ECGTrainer


def run_experiment(model_type: str, 
                  model_size: str,
                  data_dir: str,
                  output_dir: str,
                  config: dict) -> dict:
    """Run single experiment"""
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} ECG Transformer ({model_size})")
    print(f"{'='*60}")
    
    trainer = ECGTrainer(
        model_type=model_type,
        model_size=model_size,
        data_dir=data_dir,
        output_dir=output_dir,
        **config
    )
    
    try:
        # Train model
        training_history = trainer.train()
        
        # Test model
        test_results = trainer.test()
        
        # Compile results
        results = {
            'model_type': model_type,
            'model_size': model_size,
            'experiment_dir': str(trainer.exp_dir),
            'best_val_f1': trainer.best_val_score,
            'test_accuracy': test_results['accuracy'],
            'test_f1': test_results['f1'],
            'test_precision': test_results['precision'],
            'test_recall': test_results['recall'],
            'training_history': training_history,
            'total_epochs': trainer.current_epoch + 1
        }
        
        print(f"\n{model_type.upper()} Results:")
        print(f"Best Val F1: {trainer.best_val_score:.4f}")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"Test F1: {test_results['f1']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"Error training {model_type} model: {e}")
        return None


def compare_results(results_1d: dict, results_2d: dict, comparison_dir: Path):
    """Compare results between 1D and 2D models"""
    
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    # Create comparison table
    comparison_data = {
        'Metric': ['Validation F1', 'Test Accuracy', 'Test F1', 'Test Precision', 'Test Recall', 'Total Epochs'],
        '1D Transformer': [
            f"{results_1d['best_val_f1']:.4f}",
            f"{results_1d['test_accuracy']:.4f}",
            f"{results_1d['test_f1']:.4f}", 
            f"{results_1d['test_precision']:.4f}",
            f"{results_1d['test_recall']:.4f}",
            results_1d['total_epochs']
        ],
        '2D Transformer': [
            f"{results_2d['best_val_f1']:.4f}",
            f"{results_2d['test_accuracy']:.4f}",
            f"{results_2d['test_f1']:.4f}",
            f"{results_2d['test_precision']:.4f}",
            f"{results_2d['test_recall']:.4f}",
            results_2d['total_epochs']
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False))
    
    # Save comparison
    comparison_df.to_csv(comparison_dir / 'comparison_table.csv', index=False)
    
    # Determine winner
    print(f"\n{'='*30}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*30}")
    
    if results_1d['test_f1'] > results_2d['test_f1']:
        winner = '1D Transformer'
        margin = results_1d['test_f1'] - results_2d['test_f1']
    else:
        winner = '2D Transformer'
        margin = results_2d['test_f1'] - results_1d['test_f1']
    
    print(f"Winner: {winner}")
    print(f"F1 Score Margin: {margin:.4f}")
    
    # Plot comparison
    plot_comparison(results_1d, results_2d, comparison_dir)
    
    # Save detailed comparison
    detailed_comparison = {
        'winner': winner,
        'margin': margin,
        '1d_results': results_1d,
        '2d_results': results_2d,
        'comparison_table': comparison_data
    }
    
    with open(comparison_dir / 'detailed_comparison.json', 'w') as f:
        json.dump(detailed_comparison, f, indent=2)


def plot_comparison(results_1d: dict, results_2d: dict, comparison_dir: Path):
    """Create comparison plots"""
    
    # 1. Training curves comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get training histories
    hist_1d = results_1d['training_history']
    hist_2d = results_2d['training_history']
    
    epochs_1d = range(1, len(hist_1d['train_loss']) + 1)
    epochs_2d = range(1, len(hist_2d['train_loss']) + 1)
    
    # Training loss
    axes[0, 0].plot(epochs_1d, hist_1d['train_loss'], label='1D Transformer', color='blue')
    axes[0, 0].plot(epochs_2d, hist_2d['train_loss'], label='2D Transformer', color='red')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Validation loss
    axes[0, 1].plot(epochs_1d, hist_1d['val_loss'], label='1D Transformer', color='blue')
    axes[0, 1].plot(epochs_2d, hist_2d['val_loss'], label='2D Transformer', color='red')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Training accuracy
    axes[1, 0].plot(epochs_1d, hist_1d['train_acc'], label='1D Transformer', color='blue')
    axes[1, 0].plot(epochs_2d, hist_2d['train_acc'], label='2D Transformer', color='red')
    axes[1, 0].set_title('Training Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Validation F1
    axes[1, 1].plot(epochs_1d, hist_1d['val_f1'], label='1D Transformer', color='blue')
    axes[1, 1].plot(epochs_2d, hist_2d['val_f1'], label='2D Transformer', color='red')
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'training_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Final performance comparison
    metrics = ['Test Accuracy', 'Test F1', 'Test Precision', 'Test Recall']
    values_1d = [results_1d['test_accuracy'], results_1d['test_f1'], 
                 results_1d['test_precision'], results_1d['test_recall']]
    values_2d = [results_2d['test_accuracy'], results_2d['test_f1'],
                 results_2d['test_precision'], results_2d['test_recall']]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], values_1d, width, label='1D Transformer', color='blue', alpha=0.8)
    ax.bar([i + width/2 for i in x], values_2d, width, label='2D Transformer', color='red', alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Final Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (v1, v2) in enumerate(zip(values_1d, values_2d)):
        ax.text(i - width/2, v1 + 0.01, f'{v1:.3f}', ha='center', va='bottom')
        ax.text(i + width/2, v2 + 0.01, f'{v2:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(comparison_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plots saved to {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description='Compare 1D vs 2D ECG Transformers')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for experiments')
    parser.add_argument('--comparison-dir', type=str, default='outputs/comparison',
                       help='Directory to save comparison results')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, choices=['small', 'base', 'large'],
                       default='small', help='Model size for both architectures')
    
    # Training arguments (applied to both models)
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Execution options
    parser.add_argument('--run-1d-only', action='store_true', help='Run only 1D transformer')
    parser.add_argument('--run-2d-only', action='store_true', help='Run only 2D transformer')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    # Create comparison directory
    comparison_dir = Path(args.comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    config = {
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'device': args.device,
        'mixed_precision': True,
        'use_class_weights': True,
        'augment_train': True
    }
    
    print("ECG Transformer Comparison")
    print(f"Model size: {args.model_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')}")
    
    results = {}
    
    # Run 1D transformer
    if not args.run_2d_only:
        results['1d'] = run_experiment(
            model_type='1d',
            model_size=args.model_size,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config
        )
    
    # Run 2D transformer  
    if not args.run_1d_only:
        results['2d'] = run_experiment(
            model_type='2d',
            model_size=args.model_size,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config=config
        )
    
    # Compare results
    if len(results) == 2 and results['1d'] is not None and results['2d'] is not None:
        compare_results(results['1d'], results['2d'], comparison_dir)
    else:
        print("\nComparison skipped - need both models to complete successfully")
        
        # Save individual results
        for model_type, result in results.items():
            if result is not None:
                with open(comparison_dir / f'{model_type}_results.json', 'w') as f:
                    json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()