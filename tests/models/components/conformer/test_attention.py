import torch
from pytest import mark

from src.models.components.conformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    RelPositionMultiHeadedAttention
)
from src.models.components.conformer.embedding import (
    LegacyRelPositionalEncoding,
    RelPositionalEncoding
)

@mark.parametrize(
    "batch_size, n_head, n_feat, dropout_rate, feats_T",
    [
        (1,4,80,0.1,5),
        (3,4,80,0.1,5),
        (3,8,80,0.1,5),
        (3,8,512,0.1,5),
        (3,8,512,0,5),
        (3,8,512,0,10)
    ]
)
def test_legacy_rel_position_multi_headed_attention(
    batch_size:int,
    n_head:int,
    n_feat:int,
    dropout_rate:int,
    feats_T:int
):
    assert n_feat % n_head == 0
    # attentionインスタンス
    legacy_rel_pos_attn = LegacyRelPositionMultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout_rate
    )

    # positional_encodingインスタンス
    pos_enc = LegacyRelPositionalEncoding(
        d_model=n_feat,
        dropout_rate=dropout_rate
    )

    # ランダムtensor作成
    x = torch.rand(batch_size,n_feat,feats_T)   #(B, C, T)
    x = x.transpose(1,2)                        #(B, T, C)

    # positional encoding
    dropout_x, pos_emb = pos_enc(x)

    assert dropout_x.size() == torch.Size([batch_size,feats_T,n_feat])
    assert dropout_x.all() == (False if dropout_rate > 0 else True)
    assert pos_emb.size() == torch.Size([1,feats_T,n_feat])

    # self attentioin
    q = k = v = dropout_x                       #(B, T, C)
    mask = torch.tensor([[[1 for j in range(feats_T)]] for i in range(batch_size)])
    y = legacy_rel_pos_attn(q,k,v,pos_emb,mask)

    assert y.size() == torch.Size([batch_size,feats_T,n_feat])

@mark.parametrize(
    "batch_size, n_head, n_feat, dropout_rate, feats_T",
    [
        (1,4,80,0.1,5),
        (3,4,80,0.1,5),
        (3,8,80,0.1,5),
        (3,8,512,0.1,5),
        (3,8,512,0,5),
        (3,8,512,0,10)
    ]
)
def test_rel_position_multi_headed_attention(
    batch_size:int,
    n_head:int,
    n_feat:int,
    dropout_rate:int,
    feats_T:int
):
    assert n_feat % n_head == 0
    # attentionインスタンス
    rel_pos_attn = RelPositionMultiHeadedAttention(
        n_head=n_head,
        n_feat=n_feat,
        dropout_rate=dropout_rate
    )

    # positional_encodingインスタンス
    pos_enc = RelPositionalEncoding(
        d_model=n_feat,
        dropout_rate=dropout_rate
    )

    # ランダムtensor作成
    x = torch.rand(batch_size,n_feat,feats_T)   #(B, C, T)
    x = x.transpose(1,2)                        #(B, T, C)

    # positional encoding
    dropout_x, pos_emb = pos_enc(x)

    assert dropout_x.size() == torch.Size([batch_size,feats_T,n_feat])
    assert dropout_x.all() == (False if dropout_rate > 0 else True)
    assert pos_emb.size() == torch.Size([1,feats_T*2-1,n_feat])

    # self attentioin
    q = k = v = dropout_x                       #(B, T, C)
    mask = torch.tensor([[[1 for j in range(feats_T)]] for i in range(batch_size)])
    y = rel_pos_attn(q,k,v,pos_emb,mask)

    assert y.size() == torch.Size([batch_size,feats_T,n_feat])