import pytest
import torch

from src.models.components.controller import Controller
from src.models.components.posterior_encoder_vits import PosteriorEncoderVITS


@pytest.mark.parametrize(
    """
    batch_size,
    hidden_size,
    state_size,
    target_mel_channels,
    feats_T,
    c_hidden_size,
    action_size,
    prob
    """,
    [
        (1, 192, 192, 80, 5, 192, 7, False),
        (5, 12, 12, 80, 5, 12, 44, True),
    ],
)
def test_controller(
    batch_size: int,
    hidden_size: int,
    state_size: int,
    target_mel_channels: int,
    feats_T: int,
    c_hidden_size: int,
    action_size: int,
    prob: bool,
):
    # define encoder_modules
    pos_enc_vits = PosteriorEncoderVITS(
        in_channels=target_mel_channels, hidden_channels=state_size, out_channels=state_size
    )
    encoder_modules = pos_enc_vits.get_encoder_modules()

    # controller instanse
    input_size=100
    controller = Controller(
        hidden_size=hidden_size,
        state_size=state_size,
        feats_T=feats_T,
        c_hidden_size=c_hidden_size,
        action_size=action_size,
        input_size=input_size,
        bias=True,
    )

    # create random tensor
    hidden = torch.randn(batch_size, hidden_size)
    state = torch.randn(batch_size, state_size)
    target = torch.randn(batch_size, target_mel_channels, feats_T)
    assert target.transpose(1, 2).size() == torch.Size([batch_size, feats_T, target_mel_channels])
    c_hidden = torch.randn(batch_size, c_hidden_size)

    action, next_controller_hidden = controller(hidden, state, target, c_hidden, prob)

    assert next_controller_hidden.size() == torch.Size([batch_size, c_hidden_size])
    assert action.size() == torch.Size([batch_size, action_size])
