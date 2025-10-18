# Quick Start Guide

## 📋 Prerequisites

You need:
1. **ECG CSV files** (12-lead ECG data in CSV format)
2. **REFERENCE.csv** (metadata file with rhythm labels)

## 🗂️ Step 1: Organize Your Data

### Expected Folder Structure

```
ecg-arrhythmia-classification/
└── data/
    └── raw/
        ├── JS00001.csv          # ECG recording 1
        ├── JS00002.csv          # ECG recording 2
        ├── ...                  # More ECG files
        └── REFERENCE.csv        # Metadata with labels
```

### CSV File Format

#### ECG CSV Files
Your ECG CSV files should be structured as either:

**Option A: Leads as columns (PREFERRED)**
```csv
Lead_I,Lead_II,Lead_III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
0.05,0.12,0.08,-0.03,0.04,0.10,0.02,0.06,0.08,0.10,0.11,0.09
0.06,0.13,0.09,-0.02,0.05,0.11,0.03,0.07,0.09,0.11,0.12,0.10
...
```

**Option B: Leads as rows**
```csv
Time,Lead_I,Time,Lead_II,...
0.000,0.05,0.000,0.12,...
0.002,0.06,0.002,0.13,...
...
```

#### REFERENCE.csv Format

**Required columns:**
- `Recording`: ECG filename (with or without .csv extension)
- `Rhythm`: SNOMED-CT rhythm code

**Optional columns:**
- `Age`: Patient age
- `Sex`: Patient sex (M/F)
- Additional diagnosis codes

**Example REFERENCE.csv:**
```csv
Recording,Rhythm,Age,Sex
JS00001,426783006,45,M
JS00002,164889003,67,F
JS00003,426177001,52,M
```

### SNOMED-CT Codes for 4 Classes

| Class | SNOMED Codes | Description |
|-------|-------------|-------------|
| **SB** | 426177001 | Sinus Bradycardia |
| **AFIB** | 164889003, 164890007 | Atrial Fibrillation, Atrial Flutter |
| **GSVT** | 426783006, 713427006, 233896004, 233897008, 427393009 | Various supraventricular tachycardias |
| **SR** | 426783006, 427084000 | Sinus Rhythm, Sinus Irregularity |

## 🚀 Step 2: Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## ⚙️ Step 3: Configure

Edit `config/config.yaml`:

```yaml
data:
  raw_dir: "data/raw"
  reference_file: "data/raw/REFERENCE.csv"

preprocessing:
  original_fs: 500        # Your CSV sampling frequency
  target_fs: 250          # Downsample to this
  target_length: 2500     # 10 seconds at 250 Hz
  notch_freq: 50          # 50 Hz for Europe/Asia, 60 Hz for Americas
```

## 🔧 Step 4: Check Your CSV Structure

Before preprocessing, check your CSV structure:

```bash
python -c "
from src.data.data_loader import ECGDataLoader
import pandas as pd

# Load reference
ref = pd.read_csv('data/raw/REFERENCE.csv')
print('Reference file columns:', ref.columns.tolist())
print('First record:', ref.iloc[0]['Recording'])

# Check CSV structure
loader = ECGDataLoader('data/raw', 'data/raw/REFERENCE.csv')
first_file = ref.iloc[0]['Recording']
structure = loader.check_csv_structure(first_file)
print(f'CSV shape: {structure[\"shape\"]}')
print(f'Columns: {structure[\"columns\"][:5]}...')
"
```

## 🏃 Step 5: Preprocess Data

```bash
python scripts/preprocess_data.py
```

**This creates:**
```
data/processed/
├── X_train.npy              # Shape: (N_train, 12, 2500)
├── X_val.npy                # Shape: (N_val, 12, 2500)
├── X_test.npy               # Shape: (N_test, 12, 2500)
├── y_train.npy              # Shape: (N_train,)
├── y_val.npy                # Shape: (N_val,)
├── y_test.npy               # Shape: (N_test,)
├── train_metadata.csv
├── val_metadata.csv
├── test_metadata.csv
├── class_info.yaml
└── class_weights.npy
```

## 🎯 Step 6: Train Model

```bash
python scripts/train_model.py
```

Monitor training in real-time with TensorBoard:
```bash
tensorboard --logdir outputs/logs
```

## 📊 Step 7: Evaluate

```bash
python scripts/evaluate_model.py
```

## 🔍 Troubleshooting

### Issue: "CSV file not found"
**Solution:** Ensure your CSV files match the filenames in REFERENCE.csv

### Issue: "Wrong ECG shape"
**Solution:** Check if your CSV has 12 columns (or rows) for 12 leads. Modify `data_loader.py` if needed.

### Issue: "No valid labels"
**Solution:** Check that your REFERENCE.csv has correct SNOMED codes for the 4 classes

### Issue: "All signals filtered out"
**Solution:** Your signals may be too short or poor quality. Adjust quality thresholds in config:
```yaml
preprocessing:
  min_std: 0.001    # Lower threshold
  max_amplitude: 20  # Higher threshold
```

## 📝 Custom CSV Format

If your CSV format is different, modify `src/data/data_loader.py`:

```python
def load_single_ecg(self, filename: str):
    df = pd.read_csv(filepath)
    
    # YOUR CUSTOM LOADING CODE HERE
    # Example: if leads are in specific columns
    ecg_data = df[['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 
                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6']].values.T
    
    return ecg_data.astype(np.float32)
```

## 🎓 Next Steps

1. **Explore data**: Use `notebooks/01_data_exploration.ipynb`
2. **Tune hyperparameters**: Edit `config/config.yaml`
3. **Try different models**: Modify `src/models/transformer.py`
4. **Add augmentation**: Configure in `config.yaml` under `augmentation`

## 💡 Tips

- Start with a small subset (100-1000 samples) to test the pipeline
- Check class distribution - if very imbalanced, use `use_class_weights: true`
- Use GPU for training: set `device: cuda` in config
- Save your trained models in `outputs/models/`

## 📧 Need Help?

Open an issue on GitHub with:
- Your CSV structure (first few rows)
- REFERENCE.csv format
- Error messages
- Config file

---

**Ready to start? Run the preprocessing!** 🚀
```bash
python scripts/preprocess_data.py
```