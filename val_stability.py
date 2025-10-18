# cat > test_val_stability.py << 'EOF'
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
from training.trainer import ECGTrainer

print("Testing validation stability...")
trainer = ECGTrainer(
    model_type='1d',
    model_size='small',
    data_dir='data/processed',
    num_epochs=1,
    batch_size=32,
    augment_train=False  # Disable augmentation for test
)
trainer.setup()

# Run validation 5 times WITHOUT any training
print("\nRunning validation 5 times on untrained model:")
for i in range(5):
    metrics = trainer.validate()
    print(f"Run {i+1}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Loss={metrics['loss']:.4f}")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("If results are IDENTICAL → Problem is in training dynamics")
print("If results are DIFFERENT → Bug in model.eval() or dropout")
print("="*60)
# EOF