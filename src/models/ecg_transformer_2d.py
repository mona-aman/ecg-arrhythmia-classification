"""
2D ECG Transformer with Multiple Representation Support
Converts ECG signals to various 2D representations and processes with Vision Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from src.models.ecg_representations import ECGTo2DConverter


class PatchEmbedding2D(nn.Module):
    """Convert 2D spectrogram patches to embeddings"""
    
    def __init__(self,
                 img_size: Tuple[int, int] = (128, 39),  # (n_mels, time_frames)
                 patch_size: Tuple[int, int] = (16, 4),
                 in_channels: int = 12,
                 embed_dim: int = 768,
                 dropout: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.n_patches_h = img_size[0] // patch_size[0]
        self.n_patches_w = img_size[1] // patch_size[1]
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        # Patch embedding using 2D convolution
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, height, width)
        Returns:
            (batch_size, n_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Extract patches: (batch_size, embed_dim, n_patches_h, n_patches_w)
        x = self.patch_embed(x)
        
        # Flatten patches: (batch_size, embed_dim, n_patches)
        x = x.flatten(2)
        
        # Transpose: (batch_size, n_patches, embed_dim)
        x = x.transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        # If positional embedding length doesn't match sequence length, resize/interpolate
        if self.pos_embed.size(1) != x.size(1):
            # pos_embed shape: (1, n_patches+1, embed_dim)
            # x shape: (batch_size, n_patches+1, embed_dim)
            try:
                cls_pos = self.pos_embed[:, :1, :]
                pos_tokens = self.pos_embed[:, 1:, :].permute(0, 2, 1)  # (1, embed_dim, old_len)

                # Interpolate to new length (exclude cls token)
                new_len = x.size(1) - 1
                pos_tokens_interp = F.interpolate(pos_tokens, size=new_len, mode='linear', align_corners=False)
                pos_tokens_interp = pos_tokens_interp.permute(0, 2, 1)  # (1, new_len, embed_dim)

                pos_embed_resized = torch.cat([cls_pos, pos_tokens_interp], dim=1)
                x = x + pos_embed_resized
            except Exception:
                # Fallback to broadcasting (may error) so we raise a clearer message
                raise RuntimeError(f"Positional embedding size {self.pos_embed.size(1)} does not match input sequence length {x.size(1)} and interpolation failed")
        else:
            x = x + self.pos_embed
        
        return self.dropout(x)


class MultiHeadSelfAttention2D(nn.Module):
    """Multi-head self attention for 2D transformer"""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        return self.proj(out)


class MLP2D(nn.Module):
    """MLP block for 2D transformer"""
    
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlock2D(nn.Module):
    """Transformer encoder block for 2D"""
    
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention2D(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP2D(embed_dim, mlp_ratio, dropout)
        
        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x):
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) regularization"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
            
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class ECGTransformer2D(nn.Module):
    """
    2D ECG Transformer for arrhythmia classification
    Converts ECG to various 2D representations and processes with Vision Transformer
    """
    
    def __init__(self,
                 # ECG parameters
                 seq_length: int = 2500,
                 n_leads: int = 12,
                 sample_rate: int = 250,
                 
        # 2D representation parameters
        representation_type: str = "mel_spectrogram",
        representation_params: dict = None,                 # Transformer parameters
                 patch_size: Tuple[int, int] = (16, 4),
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: int = 4,
                 num_classes: int = 4,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.representation_type = representation_type
        
        # Set default parameters if none provided
        if representation_params is None:
            if representation_type == "mel_spectrogram":
                representation_params = {
                    "n_fft": 256,
                    "hop_length": 64,
                    "n_mels": 128,
                    "sample_rate": sample_rate,
                    "f_min": 0.5,
                    "f_max": 40.0,
                }
            elif representation_type == "stft":
                representation_params = {
                    "n_fft": 256,
                    "hop_length": 64,
                    "sample_rate": sample_rate,
                }
            elif representation_type == "cwt":
                representation_params = {
                    "scales": 64,
                    "sample_rate": sample_rate,
                }
            else:
                representation_params = {}
        
        # Map representation type names
        REPR_NAME_MAP = {
            'mel_spectrogram': 'mel',
            'mel': 'mel',
            's_transform': 'stransform',
            'stransform': 'stransform',
            'stft': 'stft',
            'cwt': 'cwt',
            'mfcc': 'mfcc'
        }

        # Convert ECG to 2D representations
        converter_method = REPR_NAME_MAP.get(representation_type, representation_type)
        self.ecg_to_2d = ECGTo2DConverter(
            method=converter_method,
            **representation_params
        )
        
        # Calculate expected 2D output dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, n_leads, seq_length)
            dummy_output = self.ecg_to_2d(dummy_input)
            _, _, n_freq_bins, time_frames = dummy_output.shape

        img_size = (n_freq_bins, time_frames)
        print(f"2D representation shape: {n_freq_bins} x {time_frames}")
        
        # Patch embedding
        self.patch_embed = PatchEmbedding2D(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=n_leads,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Transformer encoder blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock2D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_leads, seq_length)
        Returns:
            (batch_size, num_classes)
        """
        # Convert ECG to 2D representation
        x_2d = self.ecg_to_2d(x)  # (batch_size, n_leads, freq_bins, time_frames)
        # print(f"DEBUG: Spectrogram shape: {x_2d.shape}")
        # print(f"DEBUG: Spectrogram range: [{x_2d.min():.3f}, {x_2d.max():.3f}]")
        # print(f"DEBUG: Has NaN: {torch.isnan(x_2d).any()}")
        # print(f"DEBUG: Has Inf: {torch.isinf(x_2d).any()}")
        # Patch embedding
        x = self.patch_embed(x_2d)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification: use CLS token (first token)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        return logits
    
    def get_representations(self, x):
        """Get 2D representations for visualization"""
        with torch.no_grad():
            return self.ecg_to_2d(x)
    
    def get_attention_weights(self, x, layer_idx: int = -1):
        """Get attention weights from specified layer"""
        # Convert to 2D representations
        x_2d = self.ecg_to_2d(x)
        x = self.patch_embed(x_2d)
        
        for i, block in enumerate(self.blocks):
            if i == layer_idx or (layer_idx == -1 and i == len(self.blocks) - 1):
                # Get attention weights from this layer
                normed_x = block.norm1(x)
                qkv = block.attn.qkv(normed_x)
                batch_size, seq_len, _ = normed_x.shape
                num_heads = block.attn.num_heads
                head_dim = block.attn.head_dim
                
                qkv = qkv.reshape(batch_size, seq_len, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
                attn_weights = F.softmax(scores, dim=-1)
                
                return attn_weights
            x = block(x)
        
        return None


def create_ecg_transformer_2d_small(representation_type="mel_spectrogram", representation_params=None):
    """Create a smaller 2D ECG Transformer for faster training"""
    return ECGTransformer2D(
        seq_length=2500,
        n_leads=12,
        sample_rate=250,
        representation_type=representation_type,
        representation_params=representation_params,
        patch_size=(16, 4),
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        num_classes=4,
        dropout=0.1,
        drop_path_rate=0.1
    )


def create_ecg_transformer_2d_base(representation_type="mel_spectrogram", representation_params=None):
    """Create a base-sized 2D ECG Transformer"""
    return ECGTransformer2D(
        seq_length=2500,
        n_leads=12,
        sample_rate=250,
        representation_type=representation_type,
        representation_params=representation_params,
        patch_size=(16, 4),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=4,
        dropout=0.1,
        drop_path_rate=0.1
    )


def create_ecg_transformer_2d_large(representation_type="mel_spectrogram", representation_params=None):
    """Create a large 2D ECG Transformer"""
    return ECGTransformer2D(
        seq_length=2500,
        n_leads=12,
        sample_rate=250,
        representation_type=representation_type,
        representation_params=representation_params,
        patch_size=(16, 4),
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        num_classes=4,
        dropout=0.1,
        drop_path_rate=0.2
    )


if __name__ == "__main__":
    # Test the model
    model = create_ecg_transformer_2d_small()
    
    # Create dummy input: (batch_size, n_leads, seq_length)
    x = torch.randn(2, 12, 2500)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Test 2D representation generation
    representations = model.get_representations(x)
    print(f"2D representation shape: {representations.shape}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")