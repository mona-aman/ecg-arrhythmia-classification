#!/usr/bin/env python3
"""
Verify which files in metadata.csv actually exist and create filtered version
"""

import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

def verify_files(metadata_path: str, base_dir: str, output_path: str = None):
    """
    Check which files exist and create filtered metadata
    
    Args:
        metadata_path: Path to metadata.csv
        base_dir: Base directory containing the CSV files (e.g., processed_signals/WFDBRecords)
        output_path: Output path for filtered metadata (default: metadata_verified.csv)
    """
    print(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    base_path = Path(base_dir)
    
    print(f"\nTotal records in metadata: {len(df)}")
    print(f"Checking files in: {base_path}")
    print("This may take a few minutes...\n")
    
    # Check which files exist
    exists_list = []
    missing_files = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Verifying files"):
        file_path = base_path / row['file_path']
        exists = file_path.exists()
        exists_list.append(exists)
        
        if not exists:
            missing_files.append(row['file_path'])
    
    df['file_exists'] = exists_list
    
    # Statistics
    total = len(df)
    existing = df['file_exists'].sum()
    missing = total - existing
    
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Total records:     {total:,}")
    print(f"Files found:       {existing:,} ({existing/total*100:.1f}%)")
    print(f"Files missing:     {missing:,} ({missing/total*100:.1f}%)")
    print(f"{'='*60}\n")
    
    if missing > 0:
        print(f"First 20 missing files:")
        for i, mf in enumerate(missing_files[:20], 1):
            print(f"  {i:2d}. {mf}")
        if missing > 20:
            print(f"  ... and {missing-20} more")
        print()
    
    # Create filtered dataframe (only existing files)
    df_verified = df[df['file_exists']].copy()
    df_verified = df_verified.drop('file_exists', axis=1)
    
    # Save both versions
    if output_path is None:
        output_path = metadata_path.replace('.csv', '_verified.csv')
    
    # Save verified (filtered) version
    df_verified.to_csv(output_path, index=False)
    print(f"✅ Saved VERIFIED metadata (only existing files): {output_path}")
    
    # Save full version with exists flag
    full_path = metadata_path.replace('.csv', '_with_exists_flag.csv')
    df.to_csv(full_path, index=False)
    print(f"✅ Saved FULL metadata with exists flag: {full_path}")
    
    # Save list of missing files
    missing_list_path = metadata_path.replace('.csv', '_missing_files.txt')
    with open(missing_list_path, 'w') as f:
        for mf in missing_files:
            f.write(f"{mf}\n")
    print(f"✅ Saved list of missing files: {missing_list_path}")
    
    print(f"\n{'='*60}")
    print("RECOMMENDATION:")
    print(f"{'='*60}")
    print(f"Use the verified metadata for training:")
    print(f"  {output_path}")
    print(f"This contains only the {existing:,} records with existing files.")
    print(f"{'='*60}\n")
    
    return df_verified, df


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_and_filter_metadata.py <metadata.csv> <base_dir>")
        print("\nExample:")
        print("  python verify_and_filter_metadata.py processed_signals/metadata.csv processed_signals/WFDBRecords")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    base_dir = sys.argv[2]
    
    df_verified, df_full = verify_files(metadata_path, base_dir)