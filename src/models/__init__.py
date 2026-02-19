"""
Model Factory for ECG Transformers
Unified interface for creating 1D and 2D ECG Transformer models
"""

from .ecg_transformer_1d import (
    ECGTransformer1D,
    create_ecg_transformer_1d_small,
    create_ecg_transformer_1d_base,
    create_ecg_transformer_1d_large
)

from .ecg_transformer_2d import (
    ECGTransformer2D,
    create_ecg_transformer_2d_small,
    create_ecg_transformer_2d_base,
    create_ecg_transformer_2d_large
)


def create_model(model_type: str, size: str = "small", num_classes: int = 4, 
                 representation_type: str = "mel_spectrogram", **kwargs):
    """
    Create ECG Transformer model
    
    Args:
        model_type: '1d' or '2d'
        size: 'small', 'base', or 'large'
        num_classes: Number of output classes
        **kwargs: Additional model parameters
    
    Returns:
        PyTorch model
    """
    
    if model_type.lower() == '1d':
        if size == 'small':
            model = create_ecg_transformer_1d_small()
        elif size == 'base':
            model = create_ecg_transformer_1d_base()
        elif size == 'large':
            model = create_ecg_transformer_1d_large()
        else:
            raise ValueError(f"Unsupported size: {size}")
            
    elif model_type.lower() == '2d':
        if size == 'small':
            model = create_ecg_transformer_2d_small(representation_type=representation_type)
        elif size == 'base':
            model = create_ecg_transformer_2d_base(representation_type=representation_type)
        elif size == 'large':
            model = create_ecg_transformer_2d_large(representation_type=representation_type)
        else:
            raise ValueError(f"Unsupported size: {size}")
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Update number of classes if different
    if hasattr(model, 'head') and model.head.out_features != num_classes:
        import torch.nn as nn
        model.head = nn.Linear(model.head.in_features, num_classes)
        model.num_classes = num_classes
    
    return model


def get_model_info(model):
    """Get model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_type': '1D' if isinstance(model, ECGTransformer1D) else '2D',
        'embed_dim': model.embed_dim,
        'num_classes': model.num_classes
    }
    
    return info


def compare_models():
    """Compare different model configurations"""
    import torch
    
    models = {
        '1D Small': create_model('1d', 'small'),
        '1D Base': create_model('1d', 'base'),
        '2D Small': create_model('2d', 'small'),
        '2D Base': create_model('2d', 'base'),
    }
    
    print("Model Comparison:")
    print("=" * 60)
    print(f"{'Model':<15} {'Params':<12} {'Memory (MB)':<12} {'Type':<6}")
    print("-" * 60)
    
    # Test input
    x = torch.randn(1, 12, 2500)
    
    for name, model in models.items():
        info = get_model_info(model)
        
        # Estimate memory usage
        model.eval()
        with torch.no_grad():
            try:
                _ = model(x)
                # Rough memory estimate (parameters + activations)
                memory_mb = info['total_params'] * 4 / (1024 * 1024)  # 4 bytes per float32
                memory_str = f"{memory_mb:.1f}"
            except Exception as e:
                memory_str = "Error"
        
        print(f"{name:<15} {info['total_params']:>10,} {memory_str:>10} {info['model_type']:>6}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Create 1D model
    model_1d = create_model('1d', 'small')
    print(f"1D Model: {get_model_info(model_1d)}")
    
    # Create 2D model
    model_2d = create_model('2d', 'small')
    print(f"2D Model: {get_model_info(model_2d)}")
    
    # Test with dummy input
    x = torch.randn(2, 12, 2500)
    
    print(f"\nInput shape: {x.shape}")
    
    output_1d = model_1d(x)
    output_2d = model_2d(x)
    
    print(f"1D Output shape: {output_1d.shape}")
    print(f"2D Output shape: {output_2d.shape}")
    
    # Compare models
    print("\n")
    compare_models()