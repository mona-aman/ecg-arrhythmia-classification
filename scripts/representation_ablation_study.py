#!/usr/bin/env python3
"""
Comprehensive ablation study comparing 1D vs 2D ECG transformer architectures
with systematic evaluation of different 2D representations.

This script implements controlled experiments where everything is kept constant
except the representation method:
- Same ViT architecture (2D transformer)
- Same hyperparameters
- Same training procedure
- Same dataset splits
- Pure ablation on representation type

Author: ECG Arrhythmia Classification Research
"""

import os
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.ecg_transformer_1d import ECGTransformer1D, create_ecg_transformer_1d_base
from src.models.ecg_transformer_2d import ECGTransformer2D, create_ecg_transformer_2d_base
from src.data.dataset import ECGDataset
from src.training.trainer import ECGTrainer
from src.utils.evaluation import ECGEvaluator
from src.utils.visualization import create_comparison_plots


class RepresentationAblationStudy:
    """
    Systematic ablation study for ECG representation methods
    """
    
    def __init__(self, 
                 data_dir: str,
                 output_dir: str,
                 device: str = "auto",
                 seed: int = 42):
        """
        Initialize ablation study
        
        Args:
            data_dir: Path to preprocessed ECG data
            output_dir: Path to save results
            device: Device to use for training
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.seed = seed
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging
        self.setup_logging()
        
        # Set seeds for reproducibility
        self.set_seeds()
        
        # Define representation methods to test
        self.representation_methods = self.get_representation_methods()
        
        # Store results
        self.results = {}
        
        logging.info(f"Initialized ablation study with {len(self.representation_methods)} representation methods")
        logging.info(f"Device: {self.device}")
        logging.info(f"Output directory: {self.output_dir}")
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = self.output_dir / f"ablation_study_{self.timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def set_seeds(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    def get_representation_methods(self) -> Dict[str, Dict[str, Any]]:
        """
        Define all representation methods to test
        
        Returns:
            Dictionary mapping method names to their parameters
        """
        methods = {
            # 1D baseline
            "1d_raw": {
                "type": "1d",
                "description": "Raw 1D ECG with 1D patching"
            },
            
            # 2D representations
            "mel_spectrogram": {
                "type": "2d",
                "representation_type": "mel_spectrogram",
                "representation_params": {
                    "n_fft": 256,
                    "hop_length": 64,
                    "n_mels": 128,
                    "sample_rate": 250,
                    "f_min": 0.5,
                    "f_max": 40.0,
                },
                "description": "Mel-scale spectrograms"
            },
            
            "stft_spectrogram": {
                "type": "2d",
                "representation_type": "stft",
                "representation_params": {
                    "n_fft": 256,
                    "hop_length": 64,
                    "sample_rate": 250,
                },
                "description": "Short-Time Fourier Transform spectrograms"
            },
            
            "cwt_scalogram": {
                "type": "2d",
                "representation_type": "cwt",
                "representation_params": {
                    "scales": 64,
                    "sample_rate": 250,
                    "wavelet": "morl"
                },
                "description": "Continuous Wavelet Transform scalograms"
            },
            
            "s_transform": {
                "type": "2d",
                "representation_type": "s_transform",
                "representation_params": {
                    "n_fft": 256,
                    "sample_rate": 250,
                },
                "description": "S-Transform time-frequency representation"
            },
            
            "mfcc": {
                "type": "2d",
                "representation_type": "mfcc",
                "representation_params": {
                    "n_fft": 256,
                    "hop_length": 64,
                    "n_mels": 128,
                    "n_mfcc": 64,
                    "sample_rate": 250,
                },
                "description": "Mel-Frequency Cepstral Coefficients"
            },
            
            "chroma": {
                "type": "2d",
                "representation_type": "chroma",
                "representation_params": {
                    "n_fft": 256,
                    "hop_length": 64,
                    "sample_rate": 250,
                },
                "description": "Chroma feature representation"
            },
            
            "spectral_features": {
                "type": "2d",
                "representation_type": "spectral_features",
                "representation_params": {
                    "n_fft": 256,
                    "hop_length": 64,
                    "sample_rate": 250,
                },
                "description": "Spectral centroid, rolloff, and ZCR features"
            }
        }
        
        return methods
        
    def create_model(self, method_name: str, method_config: Dict[str, Any]) -> nn.Module:
        """
        Create model for specified representation method
        
        Args:
            method_name: Name of the representation method
            method_config: Configuration for the method
            
        Returns:
            Initialized model
        """
        if method_config["type"] == "1d":
            # 1D transformer
            model = create_ecg_transformer_1d_base()
            logging.info(f"Created 1D ECG Transformer for {method_name}")
            
        elif method_config["type"] == "2d":
            # 2D transformer with specified representation
            model = create_ecg_transformer_2d_base(
                representation_type=method_config["representation_type"],
                representation_params=method_config["representation_params"]
            )
            logging.info(f"Created 2D ECG Transformer for {method_name} ({method_config['description']})")
            
        else:
            raise ValueError(f"Unknown model type: {method_config['type']}")
            
        return model.to(self.device)
        
    def create_data_loaders(self, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create data loaders for train/val/test sets
        
        Args:
            batch_size: Batch size for data loaders
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create datasets
        train_dataset = ECGDataset(
            data_dir=self.data_dir / "train",
            split="train"
        )
        
        val_dataset = ECGDataset(
            data_dir=self.data_dir / "val", 
            split="val"
        )
        
        test_dataset = ECGDataset(
            data_dir=self.data_dir / "test",
            split="test"
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        logging.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
        
    def train_single_method(self, 
                           method_name: str, 
                           method_config: Dict[str, Any],
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           test_loader: DataLoader,
                           epochs: int = 50,
                           learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Train and evaluate a single representation method
        
        Args:
            method_name: Name of the representation method
            method_config: Configuration for the method
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Dictionary containing training and evaluation results
        """
        logging.info(f"\\n{'='*60}")
        logging.info(f"Training {method_name}: {method_config['description']}")
        logging.info(f"{'='*60}")
        
        # Create model
        model = self.create_model(method_name, method_config)
        
        # Create trainer
        trainer = ECGTrainer(
            model=model,
            device=self.device,
            output_dir=self.output_dir / method_name,
            experiment_name=f"{method_name}_{self.timestamp}"
        )
        
        # Train model
        training_history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            save_best=True,
            early_stopping_patience=10
        )
        
        # Test evaluation
        test_results = trainer.test(test_loader)
        
        # Create evaluator for detailed metrics
        evaluator = ECGEvaluator(num_classes=4)
        
        # Get predictions on test set
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = model(data)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate detailed metrics
        detailed_metrics = evaluator.evaluate_predictions(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets)
        )
        
        # Combine results
        results = {
            "method_name": method_name,
            "method_config": method_config,
            "training_history": training_history,
            "test_results": test_results,
            "detailed_metrics": detailed_metrics,
            "model_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Save individual results
        results_file = self.output_dir / method_name / "results.json"
        with open(results_file, 'w') as f:
            # Convert numpy types to native Python for JSON serialization
            json_results = self.convert_numpy_types(results)
            json.dump(json_results, f, indent=2)
        
        logging.info(f"Completed {method_name}")
        logging.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        logging.info(f"Test F1: {detailed_metrics['macro_f1']:.4f}")
        logging.info(f"Model Parameters: {results['model_params']:,}")
        
        return results
        
    def convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        else:
            return obj
            
    def run_ablation_study(self,
                          epochs: int = 50,
                          batch_size: int = 32,
                          learning_rate: float = 1e-4,
                          methods: List[str] = None) -> Dict[str, Any]:
        """
        Run complete ablation study
        
        Args:
            epochs: Number of training epochs per method
            batch_size: Batch size for training
            learning_rate: Learning rate
            methods: List of methods to test (None for all)
            
        Returns:
            Complete results dictionary
        """
        logging.info(f"\\n{'='*80}")
        logging.info(f"Starting ECG Representation Ablation Study")
        logging.info(f"Timestamp: {self.timestamp}")
        logging.info(f"{'='*80}")
        
        # Create data loaders (same for all methods)
        train_loader, val_loader, test_loader = self.create_data_loaders(batch_size)
        
        # Determine which methods to run
        if methods is None:
            methods_to_run = list(self.representation_methods.keys())
        else:
            methods_to_run = [m for m in methods if m in self.representation_methods]
            
        logging.info(f"Testing {len(methods_to_run)} representation methods:")
        for method in methods_to_run:
            logging.info(f"  - {method}: {self.representation_methods[method]['description']}")
        
        # Run experiments
        all_results = {}
        
        for method_name in methods_to_run:
            try:
                method_config = self.representation_methods[method_name]
                
                results = self.train_single_method(
                    method_name=method_name,
                    method_config=method_config,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    epochs=epochs,
                    learning_rate=learning_rate
                )
                
                all_results[method_name] = results
                self.results[method_name] = results
                
            except Exception as e:
                logging.error(f"Error training {method_name}: {str(e)}")
                continue
        
        # Create comprehensive comparison
        self.create_comparison_summary(all_results)
        
        # Save complete results
        complete_results = {
            "timestamp": self.timestamp,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": self.device,
                "seed": self.seed
            },
            "methods_tested": methods_to_run,
            "results": all_results
        }
        
        results_file = self.output_dir / f"complete_ablation_results_{self.timestamp}.json"
        with open(results_file, 'w') as f:
            json_results = self.convert_numpy_types(complete_results)
            json.dump(json_results, f, indent=2)
        
        logging.info(f"\\n{'='*80}")
        logging.info(f"Ablation Study Complete!")
        logging.info(f"Results saved to: {results_file}")
        logging.info(f"{'='*80}")
        
        return complete_results
        
    def create_comparison_summary(self, results: Dict[str, Any]):
        """
        Create comprehensive comparison summary
        
        Args:
            results: Results dictionary from all methods
        """
        logging.info("Creating comparison summary...")
        
        # Create summary table
        summary_data = []
        
        for method_name, result in results.items():
            summary_data.append({
                "Method": method_name,
                "Description": result["method_config"]["description"],
                "Type": result["method_config"]["type"],
                "Test_Accuracy": result["test_results"]["accuracy"],
                "Test_Loss": result["test_results"]["loss"],
                "Macro_F1": result["detailed_metrics"]["macro_f1"],
                "Weighted_F1": result["detailed_metrics"]["weighted_f1"],
                "Precision": result["detailed_metrics"]["macro_precision"],
                "Recall": result["detailed_metrics"]["macro_recall"],
                "Parameters": result["model_params"],
                "Best_Val_Acc": max(result["training_history"]["val_accuracy"]) if result["training_history"]["val_accuracy"] else 0,
                "Epochs_Trained": len(result["training_history"]["train_loss"]) if result["training_history"]["train_loss"] else 0
            })
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by test accuracy
        summary_df = summary_df.sort_values("Test_Accuracy", ascending=False)
        
        # Save summary
        summary_file = self.output_dir / f"summary_comparison_{self.timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Print summary
        logging.info("\\n" + "="*100)
        logging.info("ABLATION STUDY RESULTS SUMMARY")
        logging.info("="*100)
        
        for idx, row in summary_df.iterrows():
            logging.info(f"\\n{idx+1}. {row['Method']} ({row['Type'].upper()})")
            logging.info(f"   Description: {row['Description']}")
            logging.info(f"   Test Accuracy: {row['Test_Accuracy']:.4f}")
            logging.info(f"   Macro F1: {row['Macro_F1']:.4f}")
            logging.info(f"   Parameters: {row['Parameters']:,}")
        
        # Find best performing methods
        best_1d = summary_df[summary_df["Type"] == "1d"].iloc[0] if len(summary_df[summary_df["Type"] == "1d"]) > 0 else None
        best_2d = summary_df[summary_df["Type"] == "2d"].iloc[0] if len(summary_df[summary_df["Type"] == "2d"]) > 0 else None
        
        logging.info("\\n" + "="*50)
        logging.info("BEST PERFORMING METHODS")
        logging.info("="*50)
        
        if best_1d is not None:
            logging.info(f"Best 1D Method: {best_1d['Method']} (Accuracy: {best_1d['Test_Accuracy']:.4f})")
            
        if best_2d is not None:
            logging.info(f"Best 2D Method: {best_2d['Method']} (Accuracy: {best_2d['Test_Accuracy']:.4f})")
            
        if best_1d is not None and best_2d is not None:
            improvement = best_2d['Test_Accuracy'] - best_1d['Test_Accuracy']
            logging.info(f"2D vs 1D Improvement: {improvement:+.4f} ({improvement/best_1d['Test_Accuracy']*100:+.2f}%)")


def main():
    """Main function to run ablation study"""
    parser = argparse.ArgumentParser(description="ECG Representation Ablation Study")
    
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to preprocessed ECG data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Path to save results")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs per method")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--methods", type=str, nargs="+", default=None,
                       help="Specific methods to test (default: all)")
    
    args = parser.parse_args()
    
    # Create ablation study
    study = RepresentationAblationStudy(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed
    )
    
    # Run ablation study
    results = study.run_ablation_study(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        methods=args.methods
    )
    
    return results


if __name__ == "__main__":
    main()