"""
Helper to load models from state_dict checkpoints
Reconstructs the model architecture based on your project structure
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add your model paths
sys.path.insert(0, 'src')
sys.path.insert(0, 'src/models')


def load_model_from_checkpoint(checkpoint_path: str, device='cpu'):
    """
    Load a model from checkpoint, handling both full models and state_dicts
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        model: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Case 1: Checkpoint IS the model
    if isinstance(checkpoint, nn.Module):
        print(f"  Loaded: Direct model")
        return checkpoint
    
    # Case 2: Checkpoint contains 'model' key
    if 'model' in checkpoint and isinstance(checkpoint['model'], nn.Module):
        print(f"  Loaded: From 'model' key")
        return checkpoint['model']
    
    # Case 3: Checkpoint contains 'model_state_dict'
    if 'model_state_dict' in checkpoint:
        print(f"  Found state_dict, reconstructing model...")
        
        # Try to load model architecture from config
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"  Config: {config}")
            
            # Try to reconstruct based on config
            model = reconstruct_model_from_config(config, checkpoint['model_state_dict'])
            if model is not None:
                return model
        
        # Otherwise, try to infer from state_dict keys
        state_dict = checkpoint['model_state_dict']
        model = reconstruct_model_from_state_dict(state_dict)
        if model is not None:
            return model
        
        raise ValueError("Cannot reconstruct model from state_dict. Please provide model architecture.")
    
    raise ValueError(f"Unknown checkpoint format: {checkpoint.keys()}")


def reconstruct_model_from_config(config, state_dict):
    """Reconstruct model from saved config"""
    
    # Try importing your model classes
    try:
        # For 1D models
        if 'ECG1DTransformer' in str(config) or 'transformer' in str(config).lower():
            from models.ecg_transformer import ECG1DTransformer
            
            model = ECG1DTransformer(
                input_channels=config.get('input_channels', 12),
                seq_length=config.get('seq_length', 5000),
                d_model=config.get('d_model', 384),
                nhead=config.get('nhead', 6),
                num_layers=config.get('num_layers', 4),
                num_classes=config.get('num_classes', 4)
            )
            model.load_state_dict(state_dict)
            print(f"  ✓ Reconstructed ECG1DTransformer")
            return model
            
        # For 2D models
        elif 'ECG2DModel' in str(config) or '2d' in str(config).lower():
            from models.ecg_2d_model import ECG2DModel
            
            model = ECG2DModel(
                num_classes=config.get('num_classes', 4),
                # Add other params as needed
            )
            model.load_state_dict(state_dict)
            print(f"  ✓ Reconstructed ECG2DModel")
            return model
            
    except Exception as e:
        print(f"  Failed to reconstruct from config: {e}")
    
    return None


def reconstruct_model_from_state_dict(state_dict):
    """Try to infer model structure from state_dict keys"""
    
    keys = list(state_dict.keys())
    print(f"  State dict has {len(keys)} parameters")
    print(f"  First few keys: {keys[:5]}")
    
    # Check if it's a transformer
    if any('transformer' in k for k in keys):
        print(f"  Detected: Transformer-based model")
        
        # Try to import and create
        try:
            from models.ecg_transformer import ECG1DTransformer
            
            # Infer dimensions from state_dict
            # Look for embedding or first layer
            for k, v in state_dict.items():
                if 'embed' in k or 'patch' in k:
                    print(f"    {k}: {v.shape}")
            
            # Create model with default params
            model = ECG1DTransformer(
                input_channels=12,
                seq_length=5000,
                d_model=384,
                nhead=6,
                num_layers=4,
                num_classes=4
            )
            
            # Try to load
            try:
                model.load_state_dict(state_dict)
                print(f"  ✓ Successfully loaded into ECG1DTransformer")
                return model
            except Exception as e:
                print(f"  Failed to load state_dict: {e}")
                
        except ImportError:
            print(f"  Cannot import ECG1DTransformer")
    
    # Check if it has 'encoder' structure
    elif any('encoder' in k for k in keys):
        print(f"  Detected: Model with encoder structure")
        
        # Get encoder keys
        encoder_keys = [k for k in keys if 'encoder' in k]
        print(f"  Encoder keys: {encoder_keys[:3]}...")
    
    return None


def extract_encoder_from_model(model):
    """Extract encoder from a full model (removing classification head)"""
    
    # Check if model has explicit encoder attribute
    if hasattr(model, 'encoder'):
        print(f"  ✓ Found 'encoder' attribute")
        return model.encoder
    
    # Otherwise, try to separate encoder from head
    if hasattr(model, 'head') or hasattr(model, 'classifier'):
        # Get all modules except head
        encoder_modules = []
        for name, module in model.named_children():
            if name not in ['head', 'classifier', 'fc']:
                encoder_modules.append(module)
        
        if encoder_modules:
            encoder = nn.Sequential(*encoder_modules)
            print(f"  ✓ Created encoder from {len(encoder_modules)} modules")
            return encoder
    
    # Last resort: assume last layer is classifier
    modules = list(model.children())
    if len(modules) > 1:
        encoder = nn.Sequential(*modules[:-1])
        print(f"  ✓ Created encoder from first {len(modules)-1} layers")
        return encoder
    
    print(f"  ⚠️  Cannot extract encoder, returning full model")
    return model


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_model_helper.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    print(f"\nTesting model loading from: {checkpoint_path}")
    print("="*60)
    
    try:
        model = load_model_from_checkpoint(checkpoint_path, device='cpu')
        print(f"\n✓ Model loaded successfully!")
        print(f"  Type: {type(model)}")
        
        # Extract encoder
        encoder = extract_encoder_from_model(model)
        print(f"\n✓ Encoder extracted!")
        print(f"  Type: {type(encoder)}")
        
        # Test forward pass
        if '1d' in checkpoint_path.lower():
            x = torch.randn(2, 12, 5000)  # 1D ECG
        else:
            x = torch.randn(2, 12, 128, 40)  # 2D representation
        
        with torch.no_grad():
            out = encoder(x)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()