from pytest import mark
import torch

from src.models.components.conformer_decoder_fastspeech2 import ConformerDecoder

@mark.parametrize(
    """
    batch_size,
    attention_dim,
    n_channels,
    feats_T
    """,
    [
        (1,192,80,5),
        (3,384,513,15),

    ]
)
def test_decoder(
    batch_size,
    attention_dim,
    n_channels,
    feats_T
):
    # conformer_decoderインスタンス
    decoder = ConformerDecoder(
        idim=feats_T,
        odim=n_channels,
        adim=attention_dim
        )

    # ランダムtensor作成
    z = torch.rand(batch_size,attention_dim,feats_T)

    y = decoder(z)

    assert y.size() == torch.Size([batch_size,n_channels,feats_T])