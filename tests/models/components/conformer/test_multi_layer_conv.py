from pytest import mark
import torch

from src.models.components.conformer.multi_layer_conv import MultiLayeredConv1d

@mark.parametrize(
    """
    batch_size, 
    in_chans, 
    hidden_chans, 
    kernel_size, 
    feats_T
    """,
    [
        (1,80,192,3,5),
        (3,513,192,5,15),
    ]
)
def test_multi_layered_conv1d(
    batch_size,
    in_chans, 
    hidden_chans,
    kernel_size,
    feats_T,
):

    # multi_layer_convインスタンス
    multi_layer_conv = MultiLayeredConv1d(
        in_chans=in_chans,
        hidden_chans=hidden_chans,
        kernel_size=kernel_size,
        dropout_rate=0.1
    )

    # ランダムtensor作成
    x = torch.rand(batch_size,in_chans,feats_T)
    x = x.transpose(1,2)            # (B, T, C)

    y = multi_layer_conv(x)

    assert y.size() == torch.Size([batch_size,feats_T,in_chans])

