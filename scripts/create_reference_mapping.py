#!/usr/bin/env python3
"""
Create Reference Mapping
Generate a CSV file that maps Patient IDs to ECG file paths
"""

import pandas as pd
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_ecg_files(base_dir):
    """
    Find all ECG CSV files in the WFDBRecords directory structure
    
    Args:
        base_dir: Base directory containing WFDBRecords
    
    Returns:
        Dictionary mapping patient IDs to file paths
    """
    base_path = Path(base_dir)
    ecg_files = {}
    
    # Search for all CSV files
    for csv_file in base_path.rglob("*.csv"):
        # Extract patient ID from filename (e.g., JS00001.csv -> JS00001)
        patient_id = csv_file.stem
        
        # Get relative path from base directory
        relative_path = csv_file.relative_to(base_path)
        ecg_files[patient_id] = str(relative_path)
    
    return ecg_files

def create_reference_mapping():
    """Create comprehensive reference mapping"""
    
    # Paths
    metadata_file = "/ocean/projects/cis250019p/maman/ecg-arrhythmia-classification/data/raw/patient_records_new.csv"
    wfdb_dir = "/ocean/projects/cis250019p/maman/ecg-arrhythmia-classification/data/raw/WFDBRecords"
    output_file = "/ocean/projects/cis250019p/maman/ecg-arrhythmia-classification/data/raw/reference_mapping.csv"
    
    # Load patient metadata
    logger.info(f"Loading patient metadata from {metadata_file}")
    metadata_df = pd.read_csv(metadata_file)
    logger.info(f"Found {len(metadata_df)} patient records")
    
    # Find ECG files
    logger.info(f"Searching for ECG files in {wfdb_dir}")
    ecg_files = find_ecg_files(wfdb_dir)
    logger.info(f"Found {len(ecg_files)} ECG files")
    
    # Create mapping
    mapping_records = []
    matched_count = 0
    
    for _, row in metadata_df.iterrows():
        patient_id = row['Patient ID']
        
        # Check if ECG file exists
        if patient_id in ecg_files:
            mapping_record = {
                'Patient ID': patient_id,
                'Recording': ecg_files[patient_id],
                'Age': row['Age'],
                'Sex': row['Sex'],
                'Dx': row['Dx'],
                'Rx': row['Rx'],
                'Hx': row['Hx'],
                'Sx': row['Sx']
            }
            mapping_records.append(mapping_record)
            matched_count += 1
        else:
            logger.warning(f"No ECG file found for patient {patient_id}")
    
    # Create DataFrame
    mapping_df = pd.DataFrame(mapping_records)
    
    # Save to CSV
    mapping_df.to_csv(output_file, index=False)
    
    logger.info(f"Created reference mapping with {len(mapping_df)} records")
    logger.info(f"Matched {matched_count}/{len(metadata_df)} patients with ECG files")
    logger.info(f"Saved mapping to {output_file}")
    
    # Show sample
    print("\nSample of reference mapping:")
    print(mapping_df.head())
    
    return output_file

if __name__ == "__main__":
    create_reference_mapping()