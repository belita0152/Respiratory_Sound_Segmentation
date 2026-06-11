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
from torch.nn import functional as F

class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int = 3,
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
        if input_channels != 3:
            first_conv = backbone.features[0][0]
            new_conv = nn.Conv2d(
                input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None,
            )
            if pretrained_backbone:
                with torch.no_grad():
                    new_conv.weight.copy_(
                        first_conv.weight.mean(dim=1, keepdim=True).repeat(1, input_channels, 1, 1)
                    )
                    if first_conv.bias is not None:
                        new_conv.bias.copy_(first_conv.bias)
            backbone.features[0][0] = new_conv
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
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(self.embedding_dim, 512, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_width = x.shape[-1]
        x = self.features(x)
        batch_size, channels, feature_height, feature_width = x.shape

        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(batch_size, channels, feature_height, feature_width)

        x = self.segmentation_head(x)
        x = x.mean(dim=2)
        return F.interpolate(x, size=input_width, mode="linear", align_corners=False)


if __name__ == "__main__":

    model = HybridCNNTransformer(num_classes=5)

    x = torch.randn(2, 3, 224, 224)
    y = model(x)

    print("input", tuple(x.shape))
    print("output", tuple(y.shape))
    print("finite", bool(torch.isfinite(y).all()))


