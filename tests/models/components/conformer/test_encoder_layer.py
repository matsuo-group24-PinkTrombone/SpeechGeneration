import torch
from pytest import mark

from src.models.components.conformer.attention import RelPositionMultiHeadedAttention
from src.models.components.conformer.convolution import ConvolutionModule
from src.models.components.conformer.embedding import RelPositionalEncoding
from src.models.components.conformer.encoder_layer import EncoderLayer
from src.models.components.conformer.multi_layer_conv import MultiLayeredConv1d


@mark.parametrize(
    """
    batch_size,
    attention_dim,
    macaron_style,
    use_cnn_module,
    normalize_before,
    concat_after,
    stochastic_depth_rate,
    """,
    [
        (1, 192, True, True, True, True, 0.0),
        (3, 384, False, False, False, False, 0.1),
    ],
)
def test_multi_layered_conv1d(
    batch_size,
    attention_dim,
    macaron_style,
    use_cnn_module,
    normalize_before,
    concat_after,
    stochastic_depth_rate,
):
    feats_T = 5

    # positional_encodingインスタンス
    pos_enc = RelPositionalEncoding(attention_dim, 0.1)

    # self_attention_layerインスタンス
    selfattn_layer = RelPositionMultiHeadedAttention(
        n_head=4, n_feat=attention_dim, dropout_rate=0.1
    )

    # positionwise_layerインスタンス
    positionwise_layer = MultiLayeredConv1d(
        in_chans=attention_dim, hidden_chans=attention_dim, kernel_size=3, dropout_rate=0.1
    )

    # convolution_layerインスタンス
    convolution_layer = ConvolutionModule(channels=attention_dim, kernel_size=3)

    # encoder_layerインスタンス
    encoder_layer = EncoderLayer(
        attention_dim,
        selfattn_layer,
        positionwise_layer,
        positionwise_layer if macaron_style else None,
        convolution_layer if use_cnn_module else None,
        dropout_rate=0.1,
        normalize_before=normalize_before,
        concat_after=concat_after,
        stochastic_depth_rate=stochastic_depth_rate,
    )

    # ランダムtensor作成
    mean = torch.rand(batch_size, attention_dim, feats_T)
    logs = torch.rand(batch_size, attention_dim, feats_T)
    x_mask = torch.tensor([[[1] * feats_T] for i in range(batch_size)])
    z = (mean + torch.randn_like(mean) * torch.exp(logs)) * x_mask  # reparametarize
    z = z.transpose(1, 2)

    zs = pos_enc(z)  # (z, pos_emb)

    z_and_pos_emb, masks = encoder_layer(zs, x_mask)  # (z, pos_emb), masks
    z_hat = z_and_pos_emb[0]

    assert z_hat.size() == torch.Size([batch_size, feats_T, attention_dim])
