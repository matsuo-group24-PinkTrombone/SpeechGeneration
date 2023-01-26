import torch
from pytest import mark

from src.models.components.conformer.postnet import Postnet


@mark.parametrize(
    """
    batch_size,
    n_layers,
    n_chans,
    n_filts,
    feats_T,
    use_batch_norm
    """,
    [
        (1, 1, 80, 3, 5, True),
        (3, 2, 513, 5, 15, False),
    ],
)
def test_postnet(
    batch_size: int, n_layers: int, n_chans: int, n_filts: int, feats_T: int, use_batch_norm: bool
):
    # Postnetインスタンス
    idim = 1  # idimは内部で参照されない引数なので適当に1を代入
    odim = feats_T  # odimが実質idimの役割

    postnet = Postnet(
        idim=idim,
        odim=odim,
        n_layers=n_layers,
        n_chans=n_chans,
        n_filts=n_filts,
        dropout_rate=0.5,
        use_batch_norm=use_batch_norm,
    )

    # ランダムテンソル作成
    x = torch.rand(batch_size, n_chans, feats_T)
    x = x.transpose(1, 2)  # (B, T, C)
    y = postnet(x)

    assert y.size() == torch.Size([batch_size, feats_T, n_chans])
