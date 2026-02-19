#!/usr/bin/env python3
"""
Minimal Standalone Training Script for Selective Fusion
All code in one file to avoid import issues
"""

def main():
    # Import everything INSIDE main() to avoid circular imports
    import argparse
    import sys
    from pathlib import Path
    
    # Parse args first
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-1d-path', type=str, required=True)
    parser.add_argument('--model-2d-path', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='outputs/fusion')
    parser.add_argument('--exp-name', type=str, default='selective_v1')
    args = parser.parse_args()
    
    # NOW import torch (after argparse)
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import json
    
    # Import audio transforms
    try:
        import torchaudio.transforms as T
    except:
        print("Warning: torchaudio not available, using dummy transforms")
        T = None
    
    print("="*70)
    print("SELECTIVE FUSION TRAINING - STANDALONE VERSION")
    print("="*70)
    print(f"\nImports successful!")
    print(f"  Torch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    # ========================================================================
    # DATASET
    # ========================================================================
    
    class ECGFusionDataset(Dataset):
        """Simple fusion dataset"""
        
        def __init__(self, data_dir, split='train'):
            self.data_dir = Path(data_dir) / split
            self.files = sorted(list(self.data_dir.glob("*.npz")))
            
            # Setup mel transform
            if T is not None:
                self.mel_transform = T.MelSpectrogram(
                    sample_rate=250,
                    n_fft=256,
                    hop_length=64,
                    n_mels=128,
                    f_min=0.5,
                    f_max=40.0
                )
                self.amp_to_db = T.AmplitudeToDB()
            else:
                self.mel_transform = None
            
            print(f"  {split}: {len(self.files)} samples")
        
        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            data = np.load(self.files[idx])
            ecg_1d = torch.from_numpy(data['ecg_data']).float()  # (12, 5000)
            label = int(data['label'])
            
            # Generate 2D representation
            if self.mel_transform is not None:
                ecg_2d = []
                for lead_idx in range(ecg_1d.shape[0]):
                    mel = self.mel_transform(ecg_1d[lead_idx])
                    mel_db = self.amp_to_db(mel)
                    ecg_2d.append(mel_db)
                ecg_2d = torch.stack(ecg_2d)  # (12, freq, time)
            else:
                # Dummy 2D data
                ecg_2d = torch.randn(12, 128, 40)
            
            return ecg_1d, ecg_2d, label
    
    # ========================================================================
    # MODEL
    # ========================================================================
    
    class SelectiveFusion(nn.Module):
        """Selective fusion model"""
        
        def __init__(self, model_1d, model_2d, class2_threshold=0.5):
            super().__init__()
            
            # Extract encoder and head
            if hasattr(model_1d, 'encoder') and hasattr(model_1d, 'head'):
                self.encoder_1d = model_1d.encoder
                self.head_1d = model_1d.head
            else:
                # Try splitting at last layer
                layers = list(model_1d.children())
                self.encoder_1d = nn.Sequential(*layers[:-1])
                self.head_1d = layers[-1]
            
            if hasattr(model_2d, 'encoder') and hasattr(model_2d, 'head'):
                self.encoder_2d = model_2d.encoder
                self.head_2d = model_2d.head
            else:
                layers = list(model_2d.children())
                self.encoder_2d = nn.Sequential(*layers[:-1])
                self.head_2d = layers[-1]
            
            self.class2_threshold = class2_threshold
            
            # Freeze encoders
            for p in self.encoder_1d.parameters():
                p.requires_grad = False
            for p in self.encoder_2d.parameters():
                p.requires_grad = False
            
            print(f"\n[SelectiveFusion Created]")
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  Total params: {total:,}")
            print(f"  Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
        
        def forward(self, x_1d, x_2d):
            # Encode
            feat_1d = self.encoder_1d(x_1d)
            feat_2d = self.encoder_2d(x_2d)
            
            # Pool if needed
            if feat_1d.dim() == 3:
                feat_1d = feat_1d.mean(1)
            if feat_2d.dim() == 3:
                feat_2d = feat_2d.mean(1)
            
            # Classify
            logits_1d = self.head_1d(feat_1d)
            logits_2d = self.head_2d(feat_2d)
            
            # Selective fusion
            probs_2d = F.softmax(logits_2d, dim=-1)
            class2_conf = probs_2d[:, 2]
            use_2d = (class2_conf > self.class2_threshold).float().unsqueeze(1)
            
            output = use_2d * logits_2d + (1 - use_2d) * logits_1d
            
            return output
    
    # ========================================================================
    # SETUP
    # ========================================================================
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDevice: {device}")
    print(f"Output: {output_dir}")
    
    # Load data
    print(f"\n[Loading Data]")
    train_dataset = ECGFusionDataset(args.data_dir, 'train')
    val_dataset = ECGFusionDataset(args.data_dir, 'val')
    test_dataset = ECGFusionDataset(args.data_dir, 'test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Load models
    print(f"\n[Loading Pretrained Models]")
    print(f"  1D: {args.model_1d_path}")
    print(f"  2D: {args.model_2d_path}")
    
    checkpoint_1d = torch.load(args.model_1d_path, map_location=device)
    checkpoint_2d = torch.load(args.model_2d_path, map_location=device)
    
    # Extract models
    if isinstance(checkpoint_1d, nn.Module):
        model_1d = checkpoint_1d
    elif 'model' in checkpoint_1d:
        model_1d = checkpoint_1d['model']
    else:
        print(f"  Unknown 1D checkpoint structure: {checkpoint_1d.keys()}")
        return
    
    if isinstance(checkpoint_2d, nn.Module):
        model_2d = checkpoint_2d
    elif 'model' in checkpoint_2d:
        model_2d = checkpoint_2d['model']
    else:
        print(f"  Unknown 2D checkpoint structure: {checkpoint_2d.keys()}")
        return
    
    model_1d = model_1d.to(device)
    model_2d = model_2d.to(device)
    
    # Create fusion model
    print(f"\n[Creating Fusion Model]")
    model = SelectiveFusion(model_1d, model_2d, class2_threshold=0.5)
    model = model.to(device)
    
    # Diagnostic
    print(f"\n[Running Diagnostic]")
    model.eval()
    batch = next(iter(val_loader))
    x_1d, x_2d, labels = batch
    x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)
    
    with torch.no_grad():
        # Test 1D
        f1 = model.encoder_1d(x_1d)
        if f1.dim() == 3: f1 = f1.mean(1)
        l1 = model.head_1d(f1)
        a1 = (l1.argmax(1) == labels).float().mean().item()
        
        # Test 2D
        f2 = model.encoder_2d(x_2d)
        if f2.dim() == 3: f2 = f2.mean(1)
        l2 = model.head_2d(f2)
        a2 = (l2.argmax(1) == labels).float().mean().item()
        
        # Test full
        out = model(x_1d, x_2d)
        af = (out.argmax(1) == labels).float().mean().item()
    
    print(f"  1D branch:  {a1*100:.1f}% (expected ~92%)")
    print(f"  2D branch:  {a2*100:.1f}% (expected ~89%)")
    print(f"  Full model: {af*100:.1f}% (expected ~92%)")
    
    if a1 < 0.7 or a2 < 0.7 or af < 0.7:
        print(f"\n❌ Diagnostic FAILED - model is broken!")
        return
    
    print(f"  ✓ Diagnostic passed!")
    
    # Setup training
    print(f"\n[Setup Training]")
    class_weights = torch.FloatTensor([0.279, 0.509, 2.894, 0.319]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Training loop
    print(f"\n[Training]")
    print("="*70)
    
    best_val_f1 = 0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        
        for x_1d, x_2d, labels in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_1d, x_2d)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='macro')
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for x_1d, x_2d, labels in val_loader:
                x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)
                outputs = model(x_1d, x_2d)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        scheduler.step()
        
        # Print
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print(f"  Train: Loss={train_loss:.4f} Acc={train_acc*100:.2f}% F1={train_f1:.4f}")
        print(f"  Val:   Loss={val_loss:.4f} Acc={val_acc*100:.2f}% F1={val_f1:.4f}")
        
        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1
            }, output_dir / 'best_model.pth')
            print(f"  💾 Saved (F1={val_f1:.4f})")
        
        print("="*70)
    
    # Test
    print(f"\n[Testing]")
    checkpoint = torch.load(output_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for x_1d, x_2d, labels in test_loader:
            x_1d, x_2d, labels = x_1d.to(device), x_2d.to(device), labels.to(device)
            outputs = model(x_1d, x_2d)
            test_preds.extend(outputs.argmax(1).cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='macro')
    
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    print(f"Test F1: {test_f1:.4f}")
    print(f"\n{classification_report(test_targets, test_preds, target_names=['Class 0', 'Class 1', 'Class 2', 'Class 3'])}")
    
    # Save results
    results = {'test_acc': float(test_acc), 'test_f1': float(test_f1)}
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Done! Results saved to {output_dir}")


if __name__ == "__main__":
    main()