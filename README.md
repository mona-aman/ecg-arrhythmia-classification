# ECG 4-Class Arrhythmia Classification with Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Deep learning-based classification of ECG arrhythmias using Transformer models on the Chapman-Shaoxing 12-lead ECG Database.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This project implements a Transformer-based model for classifying 12-lead ECG signals into **4 major arrhythmia classes**:

| Class | Name | Description |
|-------|------|-------------|
| **SB** | Sinus Bradycardia | Heart rate < 60 bpm |
| **AFIB** | Atrial Fibrillation | Includes AF and atrial flutter |
| **GSVT** | General Supraventricular Tachycardia | Fast rhythm from above ventricles |
| **SR** | Sinus Rhythm | Normal rhythm + sinus irregularity |

### Key Features
- ✅ Handles CSV format ECG data (12-lead)
- ✅ Signal preprocessing pipeline (filtering, resampling, normalization)
- ✅ 4-class rhythm classification based on merged SNOMED-CT codes
- ✅ Transformer architecture for temporal pattern learning
- ✅ Class imbalance handling with weighted loss
- ✅ Data augmentation techniques
- ✅ Comprehensive evaluation metrics

## 📊 Dataset

**Chapman-Shaoxing 12-lead ECG Database**
- 45,152 ECG recordings from 34,905 patients
- 500 Hz sampling frequency (resampled to 250 Hz)
- 10-second recordings
- Expert-labeled SNOMED-CT codes

Based on: *Zheng et al. (2020). A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients. Scientific Reports.*

### Expected Data Structure
```
data/raw/
├── *.csv                    # ECG CSV files (12 leads × time points)
└── REFERENCE.csv           # Metadata with rhythm labels
```

**REFERENCE.csv should contain:**
- `Recording`: Filename of ECG
- `Rhythm`: SNOMED-CT rhythm code(s)
- `Age`: Patient age (optional)
- `Sex`: Patient sex (optional)

## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
ecg-arrhythmia-classification/
│
├── config/
│   └── config.yaml                 # Configuration file
│
├── data/
│   ├── raw/                        # Your CSV files here
│   └── processed/                  # Generated preprocessed data
│
├── src/
│   ├── data/
│   │   ├── data_loader.py         # Load CSV ECG files
│   │   ├── label_mapper.py        # Map SNOMED codes to 4 classes
│   │   └── preprocessor.py        # Signal preprocessing
│   ├── models/
│   │   └── transformer.py         # Transformer model
│   ├── training/
│   │   ├── trainer.py             # Training loop
│   │   └── evaluator.py           # Evaluation
│   └── utils/
│       └── visualization.py       # Plotting utilities
│
├── scripts/
│   ├── preprocess_data.py         # Preprocessing pipeline
│   ├── train_model.py             # Training script
│   └── evaluate_model.py          # Evaluation script
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_training.ipynb
│
└── outputs/
    ├── models/                    # Saved checkpoints
    ├── logs/                      # Training logs
    └── figures/                   # Plots
```

## 🔧 Usage

### Step 1: Configure Paths

Edit `config/config.yaml`:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  reference_file: "data/raw/REFERENCE.csv"

preprocessing:
  target_fs: 250
  target_length: 2500
  filter_lowcut: 0.5
  filter_highcut: 40
  notch_freq: 50  # Use 60 for Americas
```

### Step 2: Preprocess Data

```bash
python scripts/preprocess_data.py
```

**This will:**
1. Load ECG CSV files
2. Map SNOMED codes to 4 classes
3. Apply signal preprocessing:
   - Bandpass filter (0.5-40 Hz)
   - Notch filter (50/60 Hz)
   - Resampling (500 → 250 Hz)
   - Z-score normalization per lead
   - Fixed-length segmentation (2500 samples)
4. Create train/val/test splits (70/15/15%)
5. Save preprocessed data to `data/processed/`

### Step 3: Train Model

```bash
python scripts/train_model.py
```

**Training features:**
- Transformer architecture
- Class-weighted loss for imbalance
- Data augmentation
- Early stopping
- Model checkpointing

### Step 4: Evaluate Model

```bash
python scripts/evaluate_model.py
```

**Evaluation includes:**
- Accuracy, Precision, Recall, F1-score
- ROC-AUC for each class
- Confusion matrix
- Per-class performance analysis

## 🏗️ Model Architecture

### Transformer Model
```
Input (12, 2500) → Positional Encoding → Transformer Encoder
  ↓ (6 layers, 8 heads, d_model=256)
Global Average Pooling → FC Layer → 4 Classes
```

**Hyperparameters:**
- Model dimension: 256
- Attention heads: 8
- Transformer layers: 6
- Feedforward dim: 1024
- Dropout: 0.1

## 📈 Results

### Expected Performance

| Metric | SB | AFIB | GSVT | SR | Overall |
|--------|-----|------|------|-----|---------|
| Accuracy | - | - | - | - | ~94% |
| Precision | - | - | - | - | - |
| Recall | - | - | - | - | - |
| F1-Score | - | - | - | - | - |

*Fill in after training your model*

### Class Distribution

Typical distribution in the dataset:
- **SR**: ~70-75% (majority class)
- **AFIB**: ~10-15%
- **SB**: ~5-10%
- **GSVT**: ~5-10%

## 🔬 Preprocessing Pipeline

1. **Filtering**
   - Bandpass: 0.5-40 Hz (removes baseline wander & noise)
   - Notch: 50/60 Hz (removes powerline interference)

2. **Resampling**
   - 500 Hz → 250 Hz (reduces computational cost)

3. **Normalization**
   - Z-score per lead (preserves lead-specific patterns)

4. **Quality Control**
   - Removes flat signals (std < 0.01)
   - Filters extreme values

5. **Segmentation**
   - Fixed length: 2500 samples (10 seconds at 250 Hz)
   - Zero-padding for shorter signals

## 📚 Citation

If you use this code, please cite:

```bibtex
@article{zheng2020ecg,
  title={A 12-lead electrocardiogram database for arrhythmia research covering more than 10,000 patients},
  author={Zheng, Jianwei and others},
  journal={Scientific Reports},
  volume={10},
  number={1},
  pages={1--11},
  year={2020},
  publisher={Nature Publishing Group}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Chapman University & Shaoxing People's Hospital for the dataset
- PhysioNet for hosting medical datasets
- PyTorch team for the deep learning framework

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Training! 🚀**