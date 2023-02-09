import pytest
import torch

from src.models.components.linear_layers import LinearLayers


@pytest.mark.parametrize(
    """
    batch_size,
    input_size,
    hidden_size,
    layers,
    output_size,
    bias
    """,
    [
        (1, 16, 32, 1, 4, False),
        (3, 128, 256, 8, 64, True),
    ],
)
def test_linear_layers(
    batch_size: int,
    input_size: int,
    hidden_size: int,
    layers: int,
    output_size: int,
    bias: bool,
):
    """
    Args:
        batch_size (int):
        input_size (int):
        hidden_size (int):
        layers (int):
        output_size (int):
        bias (bool):
    """

    # instance
    linear_layers = LinearLayers(
        input_size=input_size,
        hidden_size=hidden_size,
        layers=layers,
        output_size=output_size,
        bias=bias,
    )

    # create input tesnor
    x = torch.rand((batch_size, input_size), dtype=torch.float32)

    output = linear_layers(x)

    assert output.size() == torch.Size([batch_size, output_size])
