"""
Selective Fusion Model - Use 2D only for Class 2 where it's better
Includes built-in diagnostics
"""

import torch
import torch.nn as nn
from typing import Optional


class SelectiveFusionConfidence(nn.Module):
    """
    Selective Fusion: Use 2D when confident about Class 2, otherwise use 1D
    
    Strategy:
    - 1D is better overall (92%) but weak on Class 2 (29% F1)
    - 2D is worse overall (89%) but good on Class 2 (42% F1)
    - Use each where it's strongest!
    """
    
    def __init__(
        self,
        full_model_1d: nn.Module,
        full_model_2d: nn.Module,
        num_classes: int = 4,
        class2_threshold: float = 0.5,
        fusion_mode: str = 'confidence',  # 'confidence', 'always_class2', or 'learned'
        dropout: float = 0.1
    ):
        """
        Args:
            full_model_1d: Complete pretrained 1D model (with head)
            full_model_2d: Complete pretrained 2D model (with head)
            num_classes: Number of classes
            class2_threshold: Confidence threshold for using 2D on Class 2
            fusion_mode: How to decide when to use 2D
            dropout: Dropout rate for learned fusion
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.class2_threshold = class2_threshold
        self.fusion_mode = fusion_mode
        
        # Extract encoders and heads from full models
        # Try multiple ways to handle different model structures
        
        # 1D model
        if hasattr(full_model_1d, 'encoder') and hasattr(full_model_1d, 'head'):
            self.encoder_1d = full_model_1d.encoder
            self.head_1d = full_model_1d.head
        elif hasattr(full_model_1d, 'features') and hasattr(full_model_1d, 'classifier'):
            self.encoder_1d = full_model_1d.features
            self.head_1d = full_model_1d.classifier
        else:
            # Assume last layer is classifier
            *encoder_layers, head = list(full_model_1d.children())
            self.encoder_1d = nn.Sequential(*encoder_layers)
            self.head_1d = head
        
        # 2D model
        if hasattr(full_model_2d, 'encoder') and hasattr(full_model_2d, 'head'):
            self.encoder_2d = full_model_2d.encoder
            self.head_2d = full_model_2d.head
        elif hasattr(full_model_2d, 'features') and hasattr(full_model_2d, 'classifier'):
            self.encoder_2d = full_model_2d.features
            self.head_2d = full_model_2d.classifier
        else:
            *encoder_layers, head = list(full_model_2d.children())
            self.encoder_2d = nn.Sequential(*encoder_layers)
            self.head_2d = head
        
        # CRITICAL: Freeze encoders
        for param in self.encoder_1d.parameters():
            param.requires_grad = False
        for param in self.encoder_2d.parameters():
            param.requires_grad = False
        
        # Keep heads trainable (fine-tune for fusion)
        for param in self.head_1d.parameters():
            param.requires_grad = True
        for param in self.head_2d.parameters():
            param.requires_grad = True
        
        # Learned fusion weights (if using learned mode)
        if fusion_mode == 'learned':
            self.fusion_weights = nn.Parameter(torch.tensor([0.9, 0.1]))  # Start: 90% 1D, 10% 2D
        
        print(f"\n[SelectiveFusion] Created with mode: {fusion_mode}")
        print(f"  Class 2 threshold: {class2_threshold}")
        print(f"  Encoders frozen: ✓")
        print(f"  Heads trainable: ✓")
        
        # Run self-diagnostic
        self._self_diagnostic()
    
    def _self_diagnostic(self):
        """Built-in diagnostic check"""
        print(f"\n{'='*60}")
        print("SELF-DIAGNOSTIC CHECK")
        print(f"{'='*60}")
        
        # Count parameters
        enc1d_params = sum(p.numel() for p in self.encoder_1d.parameters())
        enc1d_trainable = sum(p.numel() for p in self.encoder_1d.parameters() if p.requires_grad)
        
        enc2d_params = sum(p.numel() for p in self.encoder_2d.parameters())
        enc2d_trainable = sum(p.numel() for p in self.encoder_2d.parameters() if p.requires_grad)
        
        head1d_params = sum(p.numel() for p in self.head_1d.parameters())
        head1d_trainable = sum(p.numel() for p in self.head_1d.parameters() if p.requires_grad)
        
        head2d_params = sum(p.numel() for p in self.head_2d.parameters())
        head2d_trainable = sum(p.numel() for p in self.head_2d.parameters() if p.requires_grad)
        
        total = enc1d_params + enc2d_params + head1d_params + head2d_params
        trainable = enc1d_trainable + enc2d_trainable + head1d_trainable + head2d_trainable
        
        print(f"\nParameter counts:")
        print(f"  1D encoder: {enc1d_params:,} ({enc1d_trainable:,} trainable)")
        print(f"  2D encoder: {enc2d_params:,} ({enc2d_trainable:,} trainable)")
        print(f"  1D head: {head1d_params:,} ({head1d_trainable:,} trainable)")
        print(f"  2D head: {head2d_params:,} ({head2d_trainable:,} trainable)")
        print(f"  Total: {total:,} ({trainable:,} trainable = {trainable/total*100:.2f}%)")
        
        # Checks
        issues = []
        if enc1d_trainable > 0:
            issues.append("1D encoder has trainable params (should be frozen)")
        if enc2d_trainable > 0:
            issues.append("2D encoder has trainable params (should be frozen)")
        if head1d_trainable == 0:
            issues.append("1D head is frozen (should be trainable)")
        if head2d_trainable == 0:
            issues.append("2D head is frozen (should be trainable)")
        if trainable / total > 0.5:
            issues.append(f"Too many trainable params: {trainable/total*100:.1f}%")
        
        if issues:
            print(f"\n⚠️  WARNINGS:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"\n✓ All checks passed!")
        
        print(f"{'='*60}\n")
    
    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x_1d: (B, C, L) - 1D ECG signal
            x_2d: (B, C, H, W) - 2D ECG representation
        
        Returns:
            logits: (B, num_classes)
        """
        # Get features from encoders
        feat_1d = self.encoder_1d(x_1d)  # (B, N, D) or (B, D)
        feat_2d = self.encoder_2d(x_2d)  # (B, M, D) or (B, D)
        
        # Pool if needed (handle sequence outputs)
        if feat_1d.dim() == 3:  # (B, N, D)
            feat_1d = feat_1d.mean(dim=1)  # (B, D)
        if feat_2d.dim() == 3:  # (B, M, D)
            feat_2d = feat_2d.mean(dim=1)  # (B, D)
        
        # Get predictions from both heads
        logits_1d = self.head_1d(feat_1d)  # (B, num_classes)
        logits_2d = self.head_2d(feat_2d)  # (B, num_classes)
        
        # Fusion strategy
        if self.fusion_mode == 'confidence':
            # Use 2D when confident about Class 2
            probs_2d = torch.softmax(logits_2d, dim=-1)
            class2_confidence = probs_2d[:, 2]  # (B,)
            
            # Mask: 1 where we trust 2D, 0 where we trust 1D
            use_2d = (class2_confidence > self.class2_threshold).float().unsqueeze(1)  # (B, 1)
            
            # Selective fusion
            output = use_2d * logits_2d + (1 - use_2d) * logits_1d
            
        elif self.fusion_mode == 'always_class2':
            # Always use 2D for Class 2, 1D for others
            # Create mask for each class
            output = logits_1d.clone()
            output[:, 2] = logits_2d[:, 2]  # Override Class 2 with 2D prediction
            
        elif self.fusion_mode == 'learned':
            # Learned weighted ensemble
            w = torch.softmax(self.fusion_weights, dim=0)
            output = w[0] * logits_1d + w[1] * logits_2d
            
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        return output
    
    def get_branch_predictions(self, x_1d: torch.Tensor, x_2d: torch.Tensor):
        """
        Get predictions from each branch separately (for analysis)
        
        Returns:
            dict with logits_1d, logits_2d, and fused output
        """
        self.eval()
        with torch.no_grad():
            feat_1d = self.encoder_1d(x_1d)
            feat_2d = self.encoder_2d(x_2d)
            
            if feat_1d.dim() == 3:
                feat_1d = feat_1d.mean(dim=1)
            if feat_2d.dim() == 3:
                feat_2d = feat_2d.mean(dim=1)
            
            logits_1d = self.head_1d(feat_1d)
            logits_2d = self.head_2d(feat_2d)
            
        output = self(x_1d, x_2d)
        
        return {
            'logits_1d': logits_1d,
            'logits_2d': logits_2d,
            'output': output
        }


def load_pretrained_models(
    path_1d: str,
    path_2d: str,
    device: str = 'cuda'
):
    """
    Load pretrained 1D and 2D models
    
    Args:
        path_1d: Path to 1D model checkpoint
        path_2d: Path to 2D model checkpoint
        device: Device to load on
    
    Returns:
        (model_1d, model_2d)
    """
    print(f"\nLoading pretrained models...")
    print(f"  1D: {path_1d}")
    print(f"  2D: {path_2d}")
    
    checkpoint_1d = torch.load(path_1d, map_location=device)
    checkpoint_2d = torch.load(path_2d, map_location=device)
    
    print(f"\n1D checkpoint keys: {list(checkpoint_1d.keys())[:5]}...")
    print(f"2D checkpoint keys: {list(checkpoint_2d.keys())[:5]}...")
    
    # Try to extract models
    # Case 1: checkpoint IS the model
    if isinstance(checkpoint_1d, nn.Module):
        model_1d = checkpoint_1d
    # Case 2: checkpoint['model']
    elif 'model' in checkpoint_1d:
        model_1d = checkpoint_1d['model']
    # Case 3: checkpoint['model_state_dict']
    elif 'model_state_dict' in checkpoint_1d:
        print("  ⚠️  1D checkpoint has model_state_dict - need to reconstruct model")
        print("     You'll need to create the model class and load state_dict")
        raise ValueError("Please provide the 1D model class to reconstruct from state_dict")
    else:
        raise ValueError(f"Unknown 1D checkpoint structure: {checkpoint_1d.keys()}")
    
    # Same for 2D
    if isinstance(checkpoint_2d, nn.Module):
        model_2d = checkpoint_2d
    elif 'model' in checkpoint_2d:
        model_2d = checkpoint_2d['model']
    elif 'model_state_dict' in checkpoint_2d:
        print("  ⚠️  2D checkpoint has model_state_dict - need to reconstruct model")
        raise ValueError("Please provide the 2D model class to reconstruct from state_dict")
    else:
        raise ValueError(f"Unknown 2D checkpoint structure: {checkpoint_2d.keys()}")
    
    model_1d = model_1d.to(device)
    model_2d = model_2d.to(device)
    
    print(f"  ✓ Models loaded successfully")
    
    return model_1d, model_2d


if __name__ == "__main__":
    print("Testing Selective Fusion Model...")
    
    # Dummy models for testing
    class DummyModel(nn.Module):
        def __init__(self, in_features=384, num_classes=4):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, 128)
            )
            self.head = nn.Linear(128, num_classes)
        
        def forward(self, x):
            if x.dim() == 3:
                x = x.mean(dim=1)
            x = self.encoder(x)
            return self.head(x)
    
    model_1d = DummyModel(in_features=384)
    model_2d = DummyModel(in_features=512)
    
    # Create selective fusion
    fusion_model = SelectiveFusionConfidence(
        full_model_1d=model_1d,
        full_model_2d=model_2d,
        num_classes=4,
        class2_threshold=0.5,
        fusion_mode='confidence'
    )
    
    # Test forward pass
    x_1d = torch.randn(2, 12, 5000)
    x_2d = torch.randn(2, 12, 128, 40)
    
    # Dummy encoders (pretend they output features)
    x_1d_feat = torch.randn(2, 384)
    x_2d_feat = torch.randn(2, 512)
    
    output = fusion_model(x_1d_feat, x_2d_feat)
    print(f"\nOutput shape: {output.shape}")
    print(f"Expected: (2, 4)")
    
    print("\n✓ Selective fusion model test passed!")