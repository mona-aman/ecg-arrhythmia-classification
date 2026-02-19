#!/usr/bin/env python3
"""
Diagnostic script to check encoder outputs
Run this BEFORE training to verify encoders are working correctly
"""

import torch
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / 'src'
for path in [project_root, src_dir, src_dir / 'data', src_dir / 'models']:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Load data
try:
    from src.data.fusion_dataset import ECGFusionDataModule
except ImportError:
    from fusion_dataset import ECGFusionDataModule

print("="*70)
print("ENCODER DIAGNOSTIC TOOL")
print("="*70)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Load data
print("\n[Loading Data]")
data_module = ECGFusionDataModule(
    data_dir="data/processed_batched",
    batch_size=2,
    num_workers=0,
    representation_type='stft'
)
data_module.setup()

# Get one batch
batch = next(iter(data_module.train_loader))
x_1d, x_2d, labels = batch
x_1d = x_1d.to(device)
x_2d = x_2d.to(device)

print(f"  Input shapes:")
print(f"    1D: {x_1d.shape}")
print(f"    2D: {x_2d.shape}")

# Load encoders
print("\n[Loading Aligned Encoders]")
encoder_1d_path = "outputs/contrastive_alignment_stft/aligned_encoder_1d.pth"
encoder_2d_path = "outputs/contrastive_alignment_stft/aligned_encoder_2d.pth"

encoder_1d_state = torch.load(encoder_1d_path, map_location=device)
encoder_2d_state = torch.load(encoder_2d_path, map_location=device)

print(f"\n  1D state dict keys:")
for key in list(encoder_1d_state.keys())[:10]:
    print(f"    {key}: {encoder_1d_state[key].shape}")

print(f"\n  2D state dict keys:")
for key in list(encoder_2d_state.keys())[:10]:
    print(f"    {key}: {encoder_2d_state[key].shape}")

# Check if head exists
has_1d_head = any('head' in k for k in encoder_1d_state.keys())
has_2d_head = any('head' in k for k in encoder_2d_state.keys())
print(f"\n  1D has 'head' in keys: {has_1d_head}")
print(f"  2D has 'head' in keys: {has_2d_head}")

# Create encoders
try:
    from src.models.ecg_transformer_1d import create_ecg_transformer_1d_small
    from src.models.ecg_transformer_2d import create_ecg_transformer_2d_small
except ImportError:
    from ecg_transformer_1d import create_ecg_transformer_1d_small
    from models.ecg_transformer_2d import create_ecg_transformer_2d_small

encoder_1d = create_ecg_transformer_1d_small()
encoder_2d = create_ecg_transformer_2d_small(representation_type='mel_spectrogram')

print(f"\n  Encoder architecture:")
print(f"    1D has .head: {hasattr(encoder_1d, 'head')}")
print(f"    2D has .head: {hasattr(encoder_2d, 'head')}")
if hasattr(encoder_1d, 'head'):
    print(f"    1D head type: {type(encoder_1d.head)}")
if hasattr(encoder_2d, 'head'):
    print(f"    2D head type: {type(encoder_2d.head)}")

# Load weights
encoder_1d.load_state_dict(encoder_1d_state, strict=False)
encoder_2d.load_state_dict(encoder_2d_state, strict=False)
encoder_1d = encoder_1d.to(device)
encoder_2d = encoder_2d.to(device)
encoder_1d.eval()
encoder_2d.eval()

print("\n" + "="*70)
print("TESTING ENCODER OUTPUTS")
print("="*70)

# Test 1: Direct call
print("\n[Test 1: Direct encoder call]")
with torch.no_grad():
    out_1d = encoder_1d(x_1d)
    out_2d = encoder_2d(x_2d)
    print(f"  1D output shape: {out_1d.shape}")
    print(f"  2D output shape: {out_2d.shape}")
    
    if out_1d.shape[1] == 4:
        print(f"  ⚠️  WARNING: 1D encoder returns LOGITS (4D), not embeddings!")
    else:
        print(f"  ✓ 1D encoder returns embeddings ({out_1d.shape[1]}D)")
    
    if out_2d.shape[1] == 4:
        print(f"  ⚠️  WARNING: 2D encoder returns LOGITS (4D), not embeddings!")
    else:
        print(f"  ✓ 2D encoder returns embeddings ({out_2d.shape[1]}D)")

# Test 2: Remove head and try again
print("\n[Test 2: After removing classification head]")
if hasattr(encoder_1d, 'head'):
    encoder_1d.head = torch.nn.Identity()
if hasattr(encoder_2d, 'head'):
    encoder_2d.head = torch.nn.Identity()

with torch.no_grad():
    out_1d = encoder_1d(x_1d)
    out_2d = encoder_2d(x_2d)
    print(f"  1D output shape: {out_1d.shape}")
    print(f"  2D output shape: {out_2d.shape}")
    
    if out_1d.shape[1] == 4:
        print(f"  ⚠️  WARNING: Still returning logits after head removal!")
    else:
        print(f"  ✓ Now returns embeddings ({out_1d.shape[1]}D)")

# Test 3: Direct transformer access
print("\n[Test 3: Direct transformer internals]")
if hasattr(encoder_1d, 'patch_embed'):
    with torch.no_grad():
        x_patches = encoder_1d.patch_embed(x_1d)
        print(f"  1D after patch_embed: {x_patches.shape}")
        
        for blk in encoder_1d.blocks:
            x_patches = blk(x_patches)
        print(f"  1D after blocks: {x_patches.shape}")
        
        x_patches = encoder_1d.norm(x_patches)
        print(f"  1D after norm: {x_patches.shape}")
        
        cls_token = x_patches[:, 0]
        print(f"  1D CLS token: {cls_token.shape}")
        
        if cls_token.shape[1] == 4:
            print(f"  ⚠️  ERROR: CLS token is 4D - something is very wrong!")
        else:
            print(f"  ✓ CLS token is {cls_token.shape[1]}D - this is correct!")

if hasattr(encoder_2d, 'patch_embed'):
    with torch.no_grad():
        x_patches = encoder_2d.patch_embed(x_2d)
        print(f"  2D after patch_embed: {x_patches.shape}")
        
        for blk in encoder_2d.blocks:
            x_patches = blk(x_patches)
        print(f"  2D after blocks: {x_patches.shape}")
        
        x_patches = encoder_2d.norm(x_patches)
        print(f"  2D after norm: {x_patches.shape}")
        
        cls_token = x_patches[:, 0]
        print(f"  2D CLS token: {cls_token.shape}")
        
        if cls_token.shape[1] == 4:
            print(f"  ⚠️  ERROR: CLS token is 4D - something is very wrong!")
        else:
            print(f"  ✓ CLS token is {cls_token.shape[1]}D - this is correct!")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)
print("\nIf you see warnings above, the encoders are returning logits.")
print("This will cause 85% accuracy in fusion model.")
print("You need to fix how encoders are saved in Phase 1.")
print("="*70)
