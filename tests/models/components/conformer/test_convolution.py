import torch
from pytest import mark

from src.models.components.conformer.convolution import ConvolutionModule


@mark.parametrize("batch_size, channels, feats_T", [(1, 80, 5), (3, 80, 10), (5, 513, 15)])
def test_convolution(batch_size: int, channels: int, feats_T: int):

    conv = ConvolutionModule(channels=channels, kernel_size=3)

    x = torch.rand(batch_size, channels, feats_T)  # (B, C, T)

    x = x.transpose(1, 2)  # (B, T, C)
    y = conv(x).transpose(1, 2)  # (B, C, T)

    assert y.size() == torch.Size([batch_size, channels, feats_T])
