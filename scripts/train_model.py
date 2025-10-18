"""
Training Script for ECG Transformers
Command-line interface for training 1D and 2D ECG Transformer models
"""

import argparse
import sys
from pathlib import Path
import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import ECGTrainer


def main():
    parser = argparse.ArgumentParser(description='Train ECG Transformer models')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, choices=['1d', '2d'], required=True,
                       help='Type of transformer (1d or 2d)')
    parser.add_argument('--model-size', type=str, choices=['small', 'base', 'large'], 
                       default='small', help='Model size')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for experiments')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup epochs')
    
    # Training settings
    parser.add_argument('--no-class-weights', action='store_true', 
                       help='Disable class weights for balanced training')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    parser.add_argument('--gradient-clip-val', type=float, default=1.0,
                       help='Gradient clipping value')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--min-delta', type=float, default=0.001,
                       help='Minimum improvement for early stopping')
    
    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    
    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--save-all-checkpoints', action='store_true',
                       help='Save all checkpoints (not just best)')
    
    # Testing
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing (requires trained model)')
    parser.add_argument('--checkpoint-path', type=str,
                       help='Path to checkpoint for testing')
    
    # Configuration file
    parser.add_argument('--config', type=str,
                       help='YAML configuration file (overrides command line args)')
    
    parser.add_argument('--representation-type', type=str, default='mel',
                   choices=['mel', 'stft', 'cwt', 'stransform', 'mfcc', 'chroma', 'spectral_centroid'],
                   help='2D representation method for 2D model')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update args with config values
        for key, value in config.items():
            setattr(args, key.replace('-', '_'), value)
    
    # Setup trainer
    trainer = ECGTrainer(
        model_type=args.model_type,
        model_size=args.model_size,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        
        # Training hyperparameters
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        
        # Training settings
        use_class_weights=not args.no_class_weights,
        augment_train=not args.no_augmentation,
        mixed_precision=not args.no_mixed_precision,
        gradient_clip_val=args.gradient_clip_val,
        
        # Early stopping
        patience=args.patience,
        min_delta=args.min_delta,
        
        # Logging
        log_interval=args.log_interval,
        save_best_only=not args.save_all_checkpoints
    )
    
    if args.test_only:
        # Test only mode
        print("Running test evaluation...")
        if not trainer.data_module:
            trainer.setup()
        
        checkpoint_path = args.checkpoint_path
        if checkpoint_path is None:
            # Find best model in output directory
            model_dirs = list(Path(args.output_dir).glob(f"{args.model_type}_{args.model_size}_*"))
            if model_dirs:
                latest_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                checkpoint_path = latest_dir / 'best_model.pth'
                print(f"Using checkpoint: {checkpoint_path}")
            else:
                print("No trained model found. Please train first or specify --checkpoint-path")
                return
        
        test_results = trainer.test(checkpoint_path)
        
        print("\nTest Results:")
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"F1 Score: {test_results['f1']:.4f}")
        print(f"Precision: {test_results['precision']:.4f}")
        print(f"Recall: {test_results['recall']:.4f}")
        
    else:
        # Training mode
        print(f"Training {args.model_type.upper()} ECG Transformer ({args.model_size})")
        print(f"Device: {torch.device('cuda' if torch.cuda.is_available() and args.device != 'cpu' else 'cpu')}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Epochs: {args.num_epochs}")
        
        try:
            # Train model
            training_history = trainer.train()
            
            print("\nTraining completed!")
            print(f"Best validation F1: {trainer.best_val_score:.4f}")
            print(f"Experiment saved to: {trainer.exp_dir}")
            
            # Run test evaluation
            print("\nRunning test evaluation...")
            test_results = trainer.test()
            
            print("\nFinal Test Results:")
            print(f"Accuracy: {test_results['accuracy']:.4f}")
            print(f"F1 Score: {test_results['f1']:.4f}")
            print(f"Precision: {test_results['precision']:.4f}")
            print(f"Recall: {test_results['recall']:.4f}")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()