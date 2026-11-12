"""
Lightweight Depthwise-Separable 1D Temporal Convolutional Network with Multi-Head Self-Attention.

Architecture:
- Input projection (320-dim -> 128-dim)
- 4x Dilated Residual Depthwise-Separable 1D TCN blocks (exponential receptive field: dilations 1, 2, 4, 8)
- 2-Head Temporal Self-Attention Block (identifies apex inflection frames)
- Adaptive Global Temporal Pooling
- Linear Classification Head (< 1 MB INT8 footprint, sub-1ms inference)
"""
import os
from typing import Tuple
import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise-Separable 1D Convolution:
    Separates channel filtering into depthwise Conv1d (groups=channels) and pointwise 1x1 Conv1d.
    Reduces parameter count by ~85% while maintaining receptive field.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class ResidualTCNBlock(nn.Module):
    """
    Residual block with two dilated depthwise-separable conv layers.
    """
    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.conv2 = DepthwiseSeparableConv1d(channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.drop(out)
        out = self.conv2(out)
        return out + residual


class TemporalAttentionBlock(nn.Module):
    """
    Lightweight 2-Head Temporal Self-Attention layer.
    Allows the model to dynamically weight the key inflection frames of a sign.
    """
    def __init__(self, dim: int = 128, num_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (Batch, Time, Dim)
        attn_out, _ = self.mha(x, x, x)
        return self.norm(x + self.drop(attn_out))


class EchoGestureTCN(nn.Module):
    """
    Full Sign Language Classifier Network.

    Input shape: (Batch, Features=320, Time=32)
    Output shape: (Batch, num_classes)
    """
    def __init__(
        self,
        input_dim: int = 320,
        hidden_dim: int = 128,
        num_classes: int = 100,
        dropout: float = 0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 1. Input Linear Projection Layer
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6()
        )

        # 2. Dilated Temporal Convolutional Blocks
        self.tcn_blocks = nn.Sequential(
            ResidualTCNBlock(hidden_dim, dilation=1, dropout=dropout),
            ResidualTCNBlock(hidden_dim, dilation=2, dropout=dropout),
            ResidualTCNBlock(hidden_dim, dilation=4, dropout=dropout),
            ResidualTCNBlock(hidden_dim, dilation=8, dropout=dropout)
        )

        # 3. Temporal Self-Attention
        self.attention = TemporalAttentionBlock(dim=hidden_dim, num_heads=2, dropout=dropout)

        # 4. Global Temporal Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (Batch, Features=320, Time=32)
        Returns:
            Logits of shape (Batch, num_classes)
        """
        feat = self.input_proj(x)
        feat = self.tcn_blocks(feat)

        # Transpose to (Batch, Time, Dim) for MultiheadAttention
        feat_t = feat.transpose(1, 2)
        attn = self.attention(feat_t)
        # Transpose back to (Batch, Dim, Time)
        feat = attn.transpose(1, 2)

        pooled = self.global_pool(feat)
        flat = torch.flatten(pooled, start_dim=1)
        logits = self.classifier(flat)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def export_onnx(self, output_path: str, batch_size: int = 1, seq_len: int = 32):
        self.eval().cpu()
        dummy_input = torch.randn(batch_size, self.input_dim, seq_len)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.onnx.export(
            self,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["landmarks_seq"],
            output_names=["logits"],
            dynamic_axes={"landmarks_seq": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False
        )
        return output_path
