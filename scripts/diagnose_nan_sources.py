#!/usr/bin/env python3
"""
Diagnose preprocessed data to find sources of NaN during training
"""

import numpy as np
from pathlib import Path
import sys

def diagnose_preprocessed_data(data_dir: str):
    """Check preprocessed data for extreme values"""
    
    print("="*70)
    print("DIAGNOSING PREPROCESSED DATA FOR NaN SOURCES")
    print("="*70)
    
    data_path = Path(data_dir)
    
    # Aggregated counters across splits
    total_sampled = 0
    total_extreme = 0
    total_nan = 0
    total_inf = 0

    # Check each split
    for split in ['train', 'val', 'test']:
        split_dir = data_path / split
        if not split_dir.exists():
            continue
            
        print(f"\n{split.upper()} Split:")
        print("-"*70)
        
        npz_files = list(split_dir.glob("*.npz"))
        print(f"Total files: {len(npz_files)}")
        
        # Sample files to check
        sample_size = min(100, len(npz_files))
        sample_files = npz_files[:sample_size]
        
        all_mins = []
        all_maxs = []
        all_means = []
        all_stds = []
        extreme_count = 0
        nan_count = 0
        inf_count = 0
        
        for i, npz_file in enumerate(sample_files):
            try:
                data = np.load(npz_file)
                ecg_data = data['ecg_data']
                
                # Check for NaN/Inf
                if np.any(np.isnan(ecg_data)):
                    nan_count += 1
                if np.any(np.isinf(ecg_data)):
                    inf_count += 1
                
                # Get statistics
                data_min = np.min(ecg_data)
                data_max = np.max(ecg_data)
                data_mean = np.mean(ecg_data)
                data_std = np.std(ecg_data)
                
                all_mins.append(data_min)
                all_maxs.append(data_max)
                all_means.append(data_mean)
                all_stds.append(data_std)
                
                # Check for extreme values and show examples
                if abs(data_min) > 10 or abs(data_max) > 10:
                    extreme_count += 1
                    if extreme_count <= 5:  # Show first 5 extreme examples
                        print(f"  Extreme file {extreme_count}: {npz_file.name}")
                        print(f"    Range: [{data_min:.2f}, {data_max:.2f}]")
                    
            except Exception as e:
                print(f"  Error reading {npz_file.name}: {e}")
        
        # Print statistics
        print(f"\nSample statistics (n={sample_size}):")
        print(f"  Min value: {np.min(all_mins):.2f} (should be around -3 after z-score)")
        print(f"  Max value: {np.max(all_maxs):.2f} (should be around +3 after z-score)")
        print(f"  Mean range: [{np.min(all_means):.4f}, {np.max(all_means):.4f}] (should be ~0)")
        print(f"  Std range: [{np.min(all_stds):.4f}, {np.max(all_stds):.4f}] (should be ~1)")
        
        print(f"\nIssue detection:")
        print(f"  Files with NaN: {nan_count}/{sample_size}")
        print(f"  Files with Inf: {inf_count}/{sample_size}")
        print(f"  Files with extreme values (|x| > 10): {extreme_count}/{sample_size}")
        
        if extreme_count > 0:
            print(f"\n  ⚠️  WARNING: {extreme_count} files have extreme values!")
            print(f"     This can cause NaN during training")
            print(f"     Recommendation: Add input clipping or re-normalize data")
        
        if nan_count > 0 or inf_count > 0:
            print(f"\n  ❌ ERROR: Data contains NaN or Inf values!")
            print(f"     These must be removed before training")
        # Aggregate counters
        total_sampled += sample_size
        total_extreme += extreme_count
        total_nan += nan_count
        total_inf += inf_count
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    # Only make recommendations if we sampled any files
    if total_sampled > 0 and total_extreme > total_sampled * 0.1:
        print("\n1. ADD INPUT CLIPPING in your trainer:")
        print("   In train_epoch(), before forward pass:")
        print("   data = torch.clamp(data, min=-10, max=10)")
        
        print("\n2. OR LOWER LEARNING RATE:")
        print("   --learning-rate 1e-5  # Instead of 1e-4")
        
        print("\n3. OR RE-NORMALIZE DATA:")
        print("   Re-run preprocessing with quality checks")
        print("   Or add additional normalization layer in model")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        data_dir = "data/processed_batched"
        print(f"No data directory specified, using: {data_dir}")
    else:
        data_dir = sys.argv[1]
    
    diagnose_preprocessed_data(data_dir)