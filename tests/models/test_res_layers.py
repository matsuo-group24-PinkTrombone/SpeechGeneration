import pytest
import torch
from src.models.components.res_layers import ResBlock, ResLayers
in_size = 4
hid_size = 3
res_h_size = 2
out_size = 4

def test_res_block():
    mod = ResBlock(hid_size, res_h_size, hid_size)
    assert mod.fc1 is not None
    assert mod.relu is not None
    assert mod.fc2 is not None

    x = torch.rand(1, 1, hid_size) # 3 dim tensor
    out = mod(x)
    assert out.size(-1) == hid_size

@pytest.mark.parametrize("layers", [1, 3])
def test_res_layers(layers):
    mod = ResLayers(in_size, hid_size, res_h_size, layers, out_size)    
    x = torch.rand(1, 1, in_size) # 3 dim tensor
    out = mod(x)
    assert out.size(-1) == out_size
