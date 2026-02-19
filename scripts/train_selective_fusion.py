"""
Training Script for Selective Fusion Model
Includes comprehensive diagnostics at every step
"""

import sys
from pathlib import Path

# Add paths FIRST before any torch imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'models'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'data'))

# Now import everything else
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import json

# Try to import our modules with error handling
try:
    from selective_fusion_model import SelectiveFusionConfidence, load_pretrained_models
except ImportError:
    print("Error: Cannot import selective_fusion_model")
    print("Make sure selective_fusion_model.py is in src/models/")
    sys.exit(1)

try:
    from fusion_dataset import ECGFusionDataModule
except ImportError:
    print("Error: Cannot import fusion_dataset")
    print("Make sure fusion_dataset.py is in src/data/")
    sys.exit(1)


def run_diagnostic_on_batch(model, x_1d, x_2d, labels, device):
    """Run diagnostic on a single batch"""
    model.eval()
    with torch.no_grad():
        x_1d = x_1d.to(device)
        x_2d = x_2d.to(device)
        labels = labels.to(device)
        
        # Test 1D branch
        feat_1d = model.encoder_1d(x_1d)
        if feat_1d.dim() == 3:
            feat_1d = feat_1d.mean(dim=1)
        logits_1d = model.head_1d(feat_1d)
        pred_1d = logits_1d.argmax(dim=1)
        acc_1d = (pred_1d == labels).float().mean().item()
        
        # Test 2D branch
        feat_2d = model.encoder_2d(x_2d)
        if feat_2d.dim() == 3:
            feat_2d = feat_2d.mean(dim=1)
        logits_2d = model.head_2d(feat_2d)
        pred_2d = logits_2d.argmax(dim=1)
        acc_2d = (pred_2d == labels).float().mean().item()
        
        # Test full model
        output = model(x_1d, x_2d)
        pred_full = output.argmax(dim=1)
        acc_full = (pred_full == labels).float().mean().item()
        
        return {
            '1d_acc': acc_1d,
            '2d_acc': acc_2d,
            'full_acc': acc_full,
            'pred_1d': pred_1d.cpu().numpy(),
            'pred_2d': pred_2d.cpu().numpy(),
            'pred_full': pred_full.cpu().numpy(),
            'labels': labels.cpu().numpy()
        }


def comprehensive_diagnostic(model, val_loader, device):
    """Comprehensive diagnostic before training"""
    print("\n" + "="*70)
    print("COMPREHENSIVE PRE-TRAINING DIAGNOSTIC")
    print("="*70)
    
    # 1. Parameter check
    print("\n[1] PARAMETER CHECK")
    print("-"*70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    if trainable_params / total_params > 0.5:
        print("❌ ERROR: Too many trainable params! Expected <20%")
        return False
    else:
        print("✓ Trainable % looks good")
    
    # 2. Test on multiple batches
    print("\n[2] BATCH TESTING")
    print("-"*70)
    
    results_1d = []
    results_2d = []
    results_full = []
    
    num_test_batches = min(5, len(val_loader))
    
    for i, batch in enumerate(val_loader):
        if i >= num_test_batches:
            break
        
        x_1d, x_2d, labels = batch
        
        diagnostics = run_diagnostic_on_batch(model, x_1d, x_2d, labels, device)
        
        results_1d.append(diagnostics['1d_acc'])
        results_2d.append(diagnostics['2d_acc'])
        results_full.append(diagnostics['full_acc'])
        
        if i == 0:
            print(f"\nFirst batch details:")
            print(f"  1D shape: {x_1d.shape}")
            print(f"  2D shape: {x_2d.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  1D preds:   {diagnostics['pred_1d'][:10]}")
            print(f"  2D preds:   {diagnostics['pred_2d'][:10]}")
            print(f"  Full preds: {diagnostics['pred_full'][:10]}")
            print(f"  Labels:     {diagnostics['labels'][:10]}")
    
    avg_1d = np.mean(results_1d)
    avg_2d = np.mean(results_2d)
    avg_full = np.mean(results_full)
    
    print(f"\nAverage accuracy over {num_test_batches} batches:")
    print(f"  1D branch:  {avg_1d*100:.2f}% (expected ~92%)")
    print(f"  2D branch:  {avg_2d*100:.2f}% (expected ~89%)")
    print(f"  Full model: {avg_full*100:.2f}% (expected ~92%)")
    
    # Check results
    issues = []
    if avg_1d < 0.7:
        issues.append(f"1D branch too low: {avg_1d*100:.1f}% (expected ~92%)")
    if avg_2d < 0.7:
        issues.append(f"2D branch too low: {avg_2d*100:.1f}% (expected ~89%)")
    if avg_full < 0.7:
        issues.append(f"Full model too low: {avg_full*100:.1f}% (expected ~92%)")
    if avg_full < avg_1d * 0.9:
        issues.append(f"Full model worse than 1D: {avg_full*100:.1f}% < {avg_1d*100:.1f}%")
    
    if issues:
        print(f"\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        print(f"\n⚠️  CRITICAL: Model is broken! Do not train yet!")
        return False
    else:
        print(f"\n✓ All diagnostic checks passed!")
        print(f"✓ Safe to start training")
        return True


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    train_loss = 0.0
    train_preds = []
    train_targets = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        x_1d, x_2d, labels = batch
        
        x_1d = x_1d.to(device)
        x_2d = x_2d.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(x_1d, x_2d)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_targets.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_targets, train_preds)
    train_f1 = f1_score(train_targets, train_preds, average='macro')
    
    return train_loss, train_acc, train_f1


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    
    val_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            x_1d, x_2d, labels = batch
            
            x_1d = x_1d.to(device)
            x_2d = x_2d.to(device)
            labels = labels.to(device)
            
            outputs = model(x_1d, x_2d)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_targets, val_preds)
    val_f1 = f1_score(val_targets, val_preds, average='macro')
    
    return val_loss, val_acc, val_f1, val_preds, val_targets


def main():
    parser = argparse.ArgumentParser(description='Train Selective Fusion Model')
    
    # Model paths
    parser.add_argument('--model-1d-path', type=str, required=True)
    parser.add_argument('--model-2d-path', type=str, required=True)
    
    # Data
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Model config
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--fusion-mode', type=str, default='confidence',
                       choices=['confidence', 'always_class2', 'learned'])
    parser.add_argument('--class2-threshold', type=float, default=0.5)
    
    # Training
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=15)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='outputs/fusion')
    parser.add_argument('--exp-name', type=str, default='selective_fusion')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda')
    
    # Diagnostics
    parser.add_argument('--skip-diagnostics', action='store_true',
                       help='Skip pre-training diagnostics (not recommended)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("SELECTIVE FUSION MODEL TRAINING")
    print("="*70)
    print(f"Experiment: {args.exp_name}")
    print(f"Fusion mode: {args.fusion_mode}")
    print(f"Class 2 threshold: {args.class2_threshold}")
    print(f"Device: {device}")
    print("="*70)
    
    # Load data
    print("\n[Loading Data]")
    data_module = ECGFusionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        representation_type='cwt'  # You mentioned using CWT
    )
    data_module.setup()
    
    train_loader = data_module.train_loader
    val_loader = data_module.val_loader
    test_loader = data_module.test_loader
    
    print(f"✓ Data loaded: {len(train_loader)} train batches")
    
    # Load pretrained models
    print("\n[Loading Pretrained Models]")
    try:
        model_1d, model_2d = load_pretrained_models(
            path_1d=args.model_1d_path,
            path_2d=args.model_2d_path,
            device=device
        )
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("\nTrying alternative loading...")
        
        # Alternative: Load as state dicts
        checkpoint_1d = torch.load(args.model_1d_path, map_location=device)
        checkpoint_2d = torch.load(args.model_2d_path, map_location=device)
        
        print(f"1D checkpoint type: {type(checkpoint_1d)}")
        print(f"2D checkpoint type: {type(checkpoint_2d)}")
        
        if 'model_state_dict' in checkpoint_1d:
            print("\n⚠️  Need to reconstruct models from state_dict")
            print("   Please provide model classes or modify loading code")
            return
        
        raise
    
    # Create fusion model
    print("\n[Creating Fusion Model]")
    model = SelectiveFusionConfidence(
        full_model_1d=model_1d,
        full_model_2d=model_2d,
        num_classes=args.num_classes,
        class2_threshold=args.class2_threshold,
        fusion_mode=args.fusion_mode
    )
    model = model.to(device)
    
    # Run diagnostics
    if not args.skip_diagnostics:
        print("\n[Running Pre-Training Diagnostics]")
        diagnostic_passed = comprehensive_diagnostic(model, val_loader, device)
        
        if not diagnostic_passed:
            print("\n❌ DIAGNOSTICS FAILED - STOPPING")
            print("   Fix the issues above before training")
            return
    else:
        print("\n⚠️  Skipping diagnostics (not recommended)")
    
    # Setup training
    print("\n[Setting Up Training]")
    
    # Loss function with class weights
    class_counts = np.array([2408, 1327, 235, 2126])  # From your data
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    
    print(f"✓ Training setup complete")
    print(f"  Optimizer: AdamW (lr={args.learning_rate})")
    print(f"  Scheduler: CosineAnnealing")
    print(f"  Loss: CrossEntropy with class weights")
    
    # Training loop
    print("\n[Starting Training]")
    print("="*70)
    
    best_val_f1 = 0.0
    patience_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_targets = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print results
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc
            }, output_dir / 'best_model.pth')
            
            print(f"  💾 Saved best model (Val F1: {val_f1:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⚠️  Early stopping triggered (patience={args.patience})")
            break
        
        print("="*70)
    
    # Load best model for testing
    print("\n[Testing Best Model]")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x_1d, x_2d, labels = batch
            
            x_1d = x_1d.to(device)
            x_2d = x_2d.to(device)
            labels = labels.to(device)
            
            outputs = model(x_1d, x_2d)
            preds = outputs.argmax(dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    
    print("\n" + "="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"\nPer-class results:")
    print(classification_report(test_targets, test_preds, 
                               target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3']))
    print("="*70)
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'best_val_f1': float(best_val_f1),
        'config': vars(args),
        'history': history
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()