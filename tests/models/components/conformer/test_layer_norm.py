import torch

from src.models.components.conformer.layer_norm import LayerNorm

def test_layer_norm():
    """
    「分散が0＝すべての値が等しい」なので軸方向で値が等しいかを判定できる
     誤差で完全に0にならないっぽいので1e-12未満で判定
    """
    x = torch.tensor(
        [[[4*i+j*2+k for k in range(2)] for j in range(4)] for i in range(4)],
        dtype=torch.float32)
    # >>> x
    # tensor([[[ 0.,  1.],
    #          [ 2.,  3.],
    #          [ 4.,  5.],
    #          [ 6.,  7.]],

    #         [[ 4.,  5.],
    #          [ 6.,  7.],
    #          [ 8.,  9.],
    #          [10., 11.]],

    #         [[ 8.,  9.],
    #          [10., 11.],
    #          [12., 13.],
    #          [14., 15.]],

    #         [[12., 13.],
    #          [14., 15.],
    #          [16., 17.],
    #          [18., 19.]]])
    
    # default normalize (dim=-1)
    layer_norm = LayerNorm(nout=2)
    assert (layer_norm(x).var(dim=0) < 1e-12).all()
    assert (layer_norm(x).var(dim=1) < 1e-12).all()

    # default normalize (dim=0)
    layer_norm = LayerNorm(nout=4,dim=1)
    assert (layer_norm(x).var(dim=0) < 1e-12).all()
    assert (layer_norm(x).var(dim=2) < 1e-12).all()
    
    # default normalize (dim=1)
    layer_norm = LayerNorm(nout=4,dim=0)
    assert (layer_norm(x).var(dim=2) < 1e-12).all()
    assert (layer_norm(x).var(dim=1) < 1e-12).all()
    