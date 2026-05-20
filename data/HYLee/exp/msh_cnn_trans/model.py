"""
Reference: https://arxiv.org/abs/2507.20408

1. Dataset Preprocessing
2. Signal Transformation to 2D Image (wavelet -> scalogram)
3. Model
1) CNN Feature Extraction
- backbone: MobileNetV2
2) Feature Emphasizing Block
- transformer 기반 self-attention block
- MobileNetV2 feature -> flatten/project -> Transformer
3) Classifier
- Transformer output -> MLP -> softmax
4) Loss
- class-weighted sparse categorical focal loss
- L(y, p) = - w_y * (1 - p_y)^gamma * log(p_y)


"""

from __future__ import annotations

import torch
from torch import nn

class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        pretrained_backbone: bool = False,
        transformer_layers: int = 4,
        attention_heads: int = 8,
        ffn_hidden_size: int = 2048,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.embedding_dim = 1280

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=attention_heads,
            dim_feedforward=ffn_hidden_size,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        self.norm = nn.LayerNorm(self.embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)


if __name__ == "__main__":
    import torch

    model = HybridCNNTransformer(num_classes=5)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("input", tuple(x.shape))
    print("output", tuple(y.shape))
    print("finite", bool(torch.isfinite(y).all()))


