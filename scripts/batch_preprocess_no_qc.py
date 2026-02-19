#!/usr/bin/env python3
"""
Batch Preprocessing Pipeline - NO QUALITY CHECKS (accept all signals)
"""

import sys
import os
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from collections import defaultdict
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_loader import ECGDataLoader
from src.data.label_mapper import FourClassLabelMapper
# Use the no-filter preprocessor as requested
from src.data.preprocessor_no_filter import ECGPreprocessorNoFilter as ECGPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def extract_patient_id(filename: str) -> str:
    """Extract patient ID from filename"""
    return Path(filename).stem


def patient_level_split(reference_df: pd.DataFrame, 
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       random_seed: int = 42) -> tuple:
    """Perform patient-level train/val/test split to prevent data leakage"""
    
    # Extract patient IDs
    reference_df['patient_id'] = reference_df['Patient ID'].apply(extract_patient_id)
    
    # Group by patient and get their class distribution
    patient_groups = defaultdict(list)
    patient_labels = {}
    
    for idx, row in reference_df.iterrows():
        patient_id = row['patient_id']
        label = row['label']
        
        if not pd.isna(label):
            patient_groups[patient_id].append(idx)
            if patient_id not in patient_labels:
                patient_labels[patient_id] = label
    
    logger.info(f"Found {len(patient_groups)} unique patients")
    logger.info(f"Patients with valid labels: {len(patient_labels)}")
    
    # Get unique patients and their primary labels
    unique_patients = list(patient_labels.keys())
    patient_primary_labels = [patient_labels[pid] for pid in unique_patients]
    
    # Split patients (not samples) into train/val/test
    patients_train, patients_temp, labels_train, labels_temp = train_test_split(
        unique_patients, patient_primary_labels,
        train_size=train_ratio,
        stratify=patient_primary_labels,
        random_state=random_seed
    )
    
    # Split temp into val and test
    val_ratio_adjusted = val_ratio / (1 - train_ratio)
    patients_val, patients_test, labels_val, labels_test = train_test_split(
        patients_temp, labels_temp,
        train_size=val_ratio_adjusted,
        stratify=labels_temp,
        random_state=random_seed
    )
    
    # Convert patient lists to sample indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    for patient_id in patients_train:
        train_indices.extend(patient_groups[patient_id])
    
    for patient_id in patients_val:
        val_indices.extend(patient_groups[patient_id])
        
    for patient_id in patients_test:
        test_indices.extend(patient_groups[patient_id])
    
    logger.info(f"Patient-level split:")
    logger.info(f"  Train patients: {len(patients_train)} -> {len(train_indices)} samples")
    logger.info(f"  Val patients: {len(patients_val)} -> {len(val_indices)} samples") 
    logger.info(f"  Test patients: {len(patients_test)} -> {len(test_indices)} samples")
    
    # Verify no patient overlap
    train_patients_set = set(patients_train)
    val_patients_set = set(patients_val)
    test_patients_set = set(patients_test)
    
    assert len(train_patients_set & val_patients_set) == 0, "Train-Val patient overlap!"
    assert len(train_patients_set & test_patients_set) == 0, "Train-Test patient overlap!"
    assert len(val_patients_set & test_patients_set) == 0, "Val-Test patient overlap!"
    
    logger.info("✅ Verified no patient overlap between splits")
    
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def process_batch(ecg_data_batch: dict, reference_batch: pd.DataFrame, 
                  preprocessor: ECGPreprocessor, batch_num: int) -> tuple:
    """Process a batch of ECG data - NO QUALITY CHECKS"""
    
    logger.info(f"Processing batch {batch_num}: {len(reference_batch)} signals")
    
    X_batch = []
    y_batch = []
    valid_patient_info = []  # Store patient info instead of indices
    failed_count = 0
    
    for idx, row in reference_batch.iterrows():
        patient_id = row['Patient ID']
        if patient_id in ecg_data_batch:
            ecg_signal = ecg_data_batch[patient_id]
            
            # ✅ QUALITY CHECKS DISABLED - accept all signals
            processed_signal = preprocessor.preprocess_single(ecg_signal, check_quality=True)
            
            if processed_signal is not None:
                X_batch.append(processed_signal)
                y_batch.append(row['label'])
                # Store patient info directly instead of index
                valid_patient_info.append({
                    'patient_id': patient_id,
                    'label': row['label'],
                    'class_name': row['class_name']
                })
            else:
                failed_count += 1
    
    logger.info(f"Batch {batch_num}: {len(X_batch)} successful, {failed_count} failed")
    
    if len(X_batch) > 0:
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
    
    return X_batch, y_batch, valid_patient_info


def batch_preprocess_pipeline(config_path: str = "config/config.yaml", batch_size: int = 5000):
    """Complete preprocessing pipeline using batch processing"""
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract paths
    data_dir = config['data']['raw_dir']
    reference_csv = config['data']['reference_file']
    output_dir = config['data']['processed_dir'] + "_batched"
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and map labels first (lightweight)
    logger.info("\\n" + "="*60)
    logger.info("STEP 1: Loading Reference and Mapping Labels")
    logger.info("="*60)
    
    # Load just the reference file
    loader = ECGDataLoader(data_dir, reference_csv)
    reference_df = loader.reference_df.copy()
    
    # Map labels
    mapper = FourClassLabelMapper()
    reference_df = mapper.map_dataframe(
        reference_df,
        rhythm_column='Dx',
        label_column='label',
        class_name_column='class_name'
    )
    
    # Filter valid records
    reference_df = mapper.filter_valid_records(reference_df, label_column='label')
    logger.info(f"Total valid records: {len(reference_df)}")
    
    # Step 2: Create patient-level splits
    logger.info("\\n" + "="*60)
    logger.info("STEP 2: Creating Patient-Level Splits")
    logger.info("="*60)
    
    idx_train, idx_val, idx_test = patient_level_split(
        reference_df,
        train_ratio=config['preprocessing']['train_ratio'],
        val_ratio=config['preprocessing']['val_ratio'],
        random_seed=config['preprocessing']['random_seed']
    )
    
    # Step 3: Process in batches
    logger.info("\\n" + "="*60)
    logger.info("STEP 3: Batch Processing ECG Signals (NO QUALITY CHECKS)")
    logger.info("="*60)
    logger.info("⚠️  Quality checks DISABLED - all signals will be accepted")
    
    preprocessor = ECGPreprocessor(config['preprocessing'])
    
    # Create split directories
    train_dir = Path(output_dir) / "train"
    val_dir = Path(output_dir) / "val" 
    test_dir = Path(output_dir) / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split separately
    splits_data = [
        ("train", idx_train, train_dir),
        ("val", idx_val, val_dir),
        ("test", idx_test, test_dir)
    ]
    
    total_processed = 0
    processing_stats = {}
    
    for split_name, indices, split_dir in splits_data:
        logger.info(f"\\nProcessing {split_name} split: {len(indices)} samples")
        
        split_df = reference_df.iloc[indices].reset_index(drop=True)
        split_processed = 0
        split_failed = 0
        
        # Process in batches
        for start_idx in range(0, len(split_df), batch_size):
            end_idx = min(start_idx + batch_size, len(split_df))
            batch_df = split_df.iloc[start_idx:end_idx]
            
            # Load ECG data for this batch only
            batch_patients = batch_df['Patient ID'].tolist()
            ecg_data_batch = {}
            
            logger.info(f"Loading batch {start_idx//batch_size + 1}: patients {start_idx}-{end_idx}")
            
            for idx, row in batch_df.iterrows():
                patient_id = row['Patient ID']
                recording_path = row['Recording']  # This contains the nested path like "01/010/JS00001.csv"
                ecg_signal = loader.load_single_ecg(recording_path)
                if ecg_signal is not None:
                    ecg_data_batch[patient_id] = ecg_signal
            
            # Process this batch
            X_batch, y_batch, valid_patient_info = process_batch(
                ecg_data_batch, batch_df, preprocessor, start_idx//batch_size + 1
            )
            
            # Save processed signals individually using patient info
            for i, (x, label, patient_info) in enumerate(zip(X_batch, y_batch, valid_patient_info)):
                patient_id = patient_info['patient_id']
                
                np.savez_compressed(
                    split_dir / f"{patient_id}.npz",
                    ecg_data=x,
                    label=label,
                    patient_id=patient_id,
                    diagnosis_labels=label,
                    class_name=patient_info['class_name']
                )
                
                split_processed += 1
                total_processed += 1
            
            split_failed += len(batch_df) - len(X_batch)
            
            # Clear memory
            del ecg_data_batch, X_batch, y_batch
            
            logger.info(f"Batch complete. Total processed so far: {total_processed}")
        
        processing_stats[split_name] = {
            'total': len(indices),
            'processed': split_processed,
            'failed': split_failed,
            'success_rate': split_processed / len(indices)
        }
        
        logger.info(f"{split_name} complete: {split_processed}/{len(indices)} processed")
    
    # Step 4: Save metadata and summary
    logger.info("\\n" + "="*60)
    logger.info("STEP 4: Saving Metadata and Summary")
    logger.info("="*60)
    
    # Save split metadata
    for split_name, indices in [('train', idx_train), ('val', idx_val), ('test', idx_test)]:
        split_df = reference_df.iloc[indices].reset_index(drop=True)
        split_df.to_csv(f"{output_dir}/{split_name}_metadata.csv", index=False)
    
    # Save class info
    class_info = {
        'class_names': mapper.class_names,
        'class_to_idx': mapper.class_to_idx,
        'num_classes': len(mapper.class_names)
    }
    
    with open(f"{output_dir}/class_info.yaml", 'w') as f:
        yaml.dump(class_info, f)
    
    # Save summary
    summary = {
        "total_records": len(reference_df),
        "total_processed": total_processed,
        "batch_size": batch_size,
        "quality_checks_enabled": False,  # ✅ Documented
        "splits": processing_stats,
        "target_frequency": config['preprocessing']['target_fs'],
        "target_length": config['preprocessing']['target_length'],
        "output_directory": str(output_dir),
        "patient_level_split": True
    }
    
    with open(f"{output_dir}/processing_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final summary
    logger.info("\\n" + "="*60)
    logger.info("BATCH PREPROCESSING SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Total processed: {total_processed}/{len(reference_df)} signals")
    
    for split_name, stats in processing_stats.items():
        logger.info(f"{split_name}: {stats['processed']}/{stats['total']} ({stats['success_rate']:.1%})")
    
    logger.info(f"Output saved to: {output_dir}")
    logger.info("✅ Batch preprocessing completed successfully!")
    
    return output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch ECG Preprocessing - NO QUALITY CHECKS")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--batch_size", type=int, default=5000,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("⚠️  QUALITY CHECKS DISABLED")
    logger.info("   All signals will be accepted (including high amplitudes)")
    logger.info("="*60)
    
    output_dir = batch_preprocess_pipeline(args.config, args.batch_size)
    print(f"\\n🚀 Batch preprocessing completed! Check results in: {output_dir}")