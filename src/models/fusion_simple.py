import torch
import torch.nn as nn
import torch.nn.functional as F

class ECGFusionModelSimple(nn.Module):
    """
    Minimal 1D+2D fusion classifier.
    - Expects encoders that map:
        1D:  (B, C1, T) -> pooled embedding (B, D1)
        2D:  (B, C2, H, W) -> pooled embedding (B, D2)
    - Fusion: gated weighting + concat -> MLP -> logits
    - Exposes modality attention weights for visualization.
    """

    def __init__(self, encoder_1d: nn.Module, encoder_2d: nn.Module,
                 dim_1d: int, dim_2d: int, num_classes: int = 4, hidden: int = 512):
        super().__init__()
        self.enc1d = encoder_1d
        self.enc2d = encoder_2d

        # modality gates (produce a scalar gate per modality from its embedding)
        self.gate_1d = nn.Linear(dim_1d, 1)
        self.gate_2d = nn.Linear(dim_2d, 1)

        self.fuse = nn.Sequential(
            nn.Linear(dim_1d + dim_2d, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )

        # simple global pooling if your encoders output feature maps
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        self.pool2d = nn.AdaptiveAvgPool2d((1,1))

    def _embed_1d(self, x1):
        """
        Accepts either (B, C, T) or already-embedded (B, D1).
        If tensor has 3 dims, we allow enc1d to output either:
          - (B, D1) directly, or
          - a feature map (B, D1, T') which we pool to (B, D1).
        """
        h = self.enc1d(x1)
        if h.ndim == 3:  # (B, D1, T')
            h = self.pool1d(h).squeeze(-1)  # (B, D1)
        return h  # (B, D1)

    def _embed_2d(self, x2):
        """
        Accepts either (B, C, H, W) or already-embedded (B, D2).
        If tensor has 4 dims, we allow enc2d to output either:
          - (B, D2) directly, or
          - a feature map (B, D2, H', W') which we pool to (B, D2).
        """
        h = self.enc2d(x2)
        if h.ndim == 4:  # (B, D2, H', W')
            h = self.pool2d(h).flatten(1)  # (B, D2)
        return h  # (B, D2)

    def forward(self, x1, x2, return_gates=False):
        z1 = self._embed_1d(x1)   # (B, D1)
        z2 = self._embed_2d(x2)   # (B, D2)

        g1 = torch.sigmoid(self.gate_1d(z1))  # (B,1)
        g2 = torch.sigmoid(self.gate_2d(z2))  # (B,1)

        # normalized modality weights (soft-attention over modalities)
        gates = torch.cat([g1, g2], dim=1)           # (B,2)
        alpha = F.softmax(gates, dim=1)              # (B,2)
        a1 = alpha[:, :1]
        a2 = alpha[:, 1:]

        z = torch.cat([a1 * z1, a2 * z2], dim=1)     # (B, D1+D2)
        logits = self.fuse(z)

        if return_gates:
            return logits, alpha
        return logits
