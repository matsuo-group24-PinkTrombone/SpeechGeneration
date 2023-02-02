import pytest
import torch

from src.models.components.observation_auto_encoder import ObservationEncoder
from src.models.components.posterior_encoder_vits import PosteriorEncoderVITS


@pytest.mark.parametrize(
    """
    batch_size,
    hidden_size,
    state_size,
    v_channels,
    mel_channels,
    feats_T
    """,
    [(1, 192, 192, 44, 80, 5)],
)
def test_observation_encoder(
    batch_size: int, hidden_size: int, state_size: int, v_channels: int, mel_channels: int, feats_T
):
    # instance posterior encoder
    mel_encoder = PosteriorEncoderVITS(
        in_channels=mel_channels,
        out_channels=state_size,
        hidden_channels=state_size,
        global_channels=v_channels,
    )

    # instance observation encoder
    obs_encoder = ObservationEncoder(
        mel_encoder=mel_encoder, state_size=state_size, hiddden_size=hidden_size, feats_T=feats_T
    )

    # create input
    v_t = torch.rand((batch_size, v_channels))
    mel = torch.rand((batch_size, mel_channels, feats_T))
    hidden = torch.rand((batch_size, hidden_size))

    obs = (v_t, mel)

    state = obs_encoder(hidden, obs)

    assert state.sample().size() == torch.Size([batch_size, state_size])
