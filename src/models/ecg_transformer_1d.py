"""
1D ECG Transformer
Based on Vision Transformer architecture adapted for 1D ECG signals
Processes raw ECG time series data directly
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PatchEmbedding1D(nn.Module):
    """Convert 1D ECG signal to patches and embed them"""
    
    def __init__(self, 
                 seq_length: int = 2500,
                 n_leads: int = 12,
                 patch_size: int = 25,
                 embed_dim: int = 768,
                 dropout: float = 0.15):
        super().__init__()
        
        self.seq_length = seq_length
        self.n_leads = n_leads
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches per lead
        self.n_patches_per_lead = seq_length // patch_size
        self.total_patches = self.n_patches_per_lead * n_leads
        
        # Patch embedding: Conv1D to extract patches
        self.patch_embed = nn.Conv1d(
            in_channels=n_leads,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional embeddings
        # self.pos_embed = nn.Parameter(torch.randn(1, self.total_patches + 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches_per_lead + 1, embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, n_leads, seq_length)
        Returns:
            (batch_size, n_patches + 1, embed_dim)
        """
        batch_size = x.shape[0]
        
        # Extract patches: (batch_size, embed_dim, n_patches_per_lead)
        x = self.patch_embed(x)
        
        # Reshape: (batch_size, embed_dim, n_patches_per_lead) -> (batch_size, n_patches_per_lead, embed_dim)
        x = x.transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention mechanism"""
    
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
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        return self.proj(out)


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
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


class TransformerBlock(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)
        
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


class ECGTransformer1D(nn.Module):
    """
    1D ECG Transformer for arrhythmia classification
    Processes raw ECG time series data
    """
    
    def __init__(self,
                 seq_length: int = 2500,
                 n_leads: int = 12,
                 patch_size: int = 25,
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
        
        # Patch embedding
        self.patch_embed = PatchEmbedding1D(
            seq_length=seq_length,
            n_leads=n_leads,
            patch_size=patch_size,
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Transformer encoder blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
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
        elif isinstance(m, nn.Conv1d):
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
        # Patch embedding
        x = self.patch_embed(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification: use CLS token (first token)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        
        return logits
    
    def get_attention_weights(self, x, layer_idx: int = -1):
        """Get attention weights from specified layer"""
        x = self.patch_embed(x)
        
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


def create_ecg_transformer_1d_small():
    """Create a smaller 1D ECG Transformer for faster training"""
    return ECGTransformer1D(
        seq_length=2500,
        n_leads=12,
        patch_size=25,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4,
        num_classes=4,
        dropout=0.1,
        drop_path_rate=0.1
    )


def create_ecg_transformer_1d_base():
    """Create a base-sized 1D ECG Transformer"""
    return ECGTransformer1D(
        seq_length=2500,
        n_leads=12,
        patch_size=25,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_classes=4,
        dropout=0.1,
        drop_path_rate=0.1
    )


def create_ecg_transformer_1d_large():
    """Create a large 1D ECG Transformer"""
    return ECGTransformer1D(
        seq_length=2500,
        n_leads=12,
        patch_size=25,
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
    model = create_ecg_transformer_1d_small()
    
    # Create dummy input: (batch_size, n_leads, seq_length)
    x = torch.randn(2, 12, 2500)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")