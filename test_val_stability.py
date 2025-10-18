import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
from models import create_model
from data.torch_dataset import ECGDataModule

print("Testing validation stability...")

# Setup
data_module = ECGDataModule(
    data_dir='data/processed',
    batch_size=32,
    augment_train=False,
    num_workers=0
)
data_module.setup()

model = create_model('1d', 'small', num_classes=4)
model.eval()  # CRITICAL: set to eval mode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Run validation 5 times
print("\nRunning validation 5 times on same untrained model:")

for run in range(5):
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in data_module.val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    print(f"Run {run+1}: Acc={acc:.4f}, F1={f1:.4f}")

print("\n" + "="*60)
print("If all 5 runs are IDENTICAL → training dynamics issue")
print("If runs are DIFFERENT → model.eval() or data loading bug")
print("="*60)
