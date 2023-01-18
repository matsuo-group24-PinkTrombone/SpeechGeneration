from pytest import mark
import torch

from src.models.components.conformer.encoder import Encoder

@mark.parametrize(
    """
    batch_size, 
    attention_dim, 
    feats_T,
    legacy
    """,
    [
        (1,192,5,True),
        (3,384,15,False),

    ]
)
def test_encoder(
    batch_size,
    attention_dim,
    feats_T,
    legacy,
):
    # encoderインスタンス
    if legacy:
        pos_enc_layer_type="legacy_rel_pos"
        selfattention_layer_type="legacy_rel_selfattn"
    else:
        pos_enc_layer_type="rel_pos"
        selfattention_layer_type="rel_selfattn"
    encoder = Encoder(
        idim=feats_T,
        attention_dim=attention_dim,
        pos_enc_layer_type=pos_enc_layer_type,
        selfattention_layer_type=selfattention_layer_type
    )

    # ランダムtensor作成
    z = torch.rand(batch_size,feats_T,attention_dim)

    y = encoder(z,None)