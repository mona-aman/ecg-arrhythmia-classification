"""
Test script to verify 1D and 2D ECG Transformer models
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import create_model, get_model_info


def test_models():
    """Test both 1D and 2D models"""
    
    print("Testing ECG Transformer Models")
    print("=" * 50)
    
    # Test input: (batch_size, n_leads, seq_length)
    batch_size = 2
    n_leads = 12
    seq_length = 2500
    
    x = torch.randn(batch_size, n_leads, seq_length)
    print(f"Input shape: {x.shape}")
    print()
    
    # Test 1D model
    print("1D ECG Transformer:")
    print("-" * 30)
    
    try:
        model_1d = create_model('1d', 'small')
        output_1d = model_1d(x)
        
        info_1d = get_model_info(model_1d)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {info_1d['total_params']:,}")
        print(f"  Output shape: {output_1d.shape}")
        print(f"  Embed dim: {info_1d['embed_dim']}")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
    
    # Test 2D model
    print("2D ECG Transformer:")
    print("-" * 30)
    
    try:
        model_2d = create_model('2d', 'small')
        output_2d = model_2d(x)
        
        info_2d = get_model_info(model_2d)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {info_2d['total_params']:,}")
        print(f"  Output shape: {output_2d.shape}")
        print(f"  Embed dim: {info_2d['embed_dim']}")
        
        # Test spectrogram generation
        spectrograms = model_2d.get_spectrograms(x)
        print(f"  Spectrogram shape: {spectrograms.shape}")
        print()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print()
    
    # Compare model sizes
    print("Model Comparison:")
    print("-" * 30)
    
    models = {
        '1D Small': create_model('1d', 'small'),
        '1D Base': create_model('1d', 'base'),
        '2D Small': create_model('2d', 'small'),
        '2D Base': create_model('2d', 'base'),
    }
    
    print(f"{'Model':<12} {'Parameters':<12} {'Memory (MB)':<12}")
    print("-" * 40)
    
    for name, model in models.items():
        try:
            info = get_model_info(model)
            # Rough memory estimate
            memory_mb = info['total_params'] * 4 / (1024 * 1024)
            print(f"{name:<12} {info['total_params']:>10,} {memory_mb:>10.1f}")
        except Exception as e:
            print(f"{name:<12} {'Error':>10} {'Error':>10}")
    
    print("\n✓ Model testing completed!")


if __name__ == "__main__":
    test_models()