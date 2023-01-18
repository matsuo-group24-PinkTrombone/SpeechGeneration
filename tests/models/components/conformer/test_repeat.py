import torch
from pytest import mark

from src.models.components.conformer.repeat import repeat

@mark.parametrize("N_repeat", [1,3,5])
def test_repeat(N_repeat:int):
    """
    Example:
    N_repeat=3でLinearを繰り返す場合

    >>> repeat(3,lambda n:torch.nn.Linear(n+1,n+2))
    MultiSequential(
      (0): Linear(in_features=1, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=3, bias=True)
      (2): Linear(in_features=3, out_features=4, bias=True)
    )
    """
    repeated_layer = repeat(N_repeat,lambda n:torch.nn.Linear(n+1,n+2))
    assert len(repeated_layer) == N_repeat