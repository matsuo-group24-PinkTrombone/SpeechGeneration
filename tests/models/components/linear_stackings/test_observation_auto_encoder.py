import pytest
import torch

from src.models.components.conformer_decoder_fastspeech2 import ConformerDecoder
from src.models.components.linear_stackings.observation_auto_encoder import (
    ObservationDecoder,
    ObservationEncoder,
)
from src.models.components.posterior_encoder_vits import PosteriorEncoderVITS


@pytest.mark.parametrize(
    """
    batch_size,
    hidden_size,
    state_size,
    v_channels,
    mel_channels,
    feats_T,
    stacked_linear_hidden_size,
    num_layers,
    """,
    [(1, 192, 192, 75, 80, 5, 128, 2)],
)
def test_observation_encoder(
    batch_size: int,
    hidden_size: int,
    state_size: int,
    v_channels: int,
    mel_channels: int,
    feats_T: int,
    stacked_linear_hidden_size: int,
    num_layers: int,
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
        mel_encoder=mel_encoder,
        state_size=state_size,
        hidden_size=hidden_size,
        feats_T=feats_T,
        stacked_linear_hidden_size=stacked_linear_hidden_size,
        num_layers=num_layers,
    )

    # create input
    v_t = torch.rand((batch_size, v_channels))
    mel = torch.rand((batch_size, mel_channels, feats_T))
    hidden = torch.rand((batch_size, hidden_size))

    obs = (v_t, mel)

    state = obs_encoder(hidden, obs)

    assert state.sample().size() == torch.Size([batch_size, state_size])


@pytest.mark.parametrize(
    """
    batch_size,
    hidden_size,
    state_size,
    v_channels,
    mel_channels,
    feats_T,
    stacked_linear_hidden_size,
    """,
    [(1, 192, 192, 75, 80, 5, 128)],
)
def test_observation_decoder(
    batch_size: int,
    hidden_size: int,
    state_size: int,
    v_channels: int,
    mel_channels: int,
    feats_T: int,
    stacked_linear_hidden_size: int,
):
    # instance posterior encoder
    idim = hidden_size + state_size
    conformer_decoder = ConformerDecoder(idim=idim, odim=mel_channels, adim=idim)

    # instance observation encoder
    obs_decoder = ObservationDecoder(
        decoder=conformer_decoder,
        feats_T=feats_T,
        voc_state_size=v_channels,
        stacked_linear_hidden_size=stacked_linear_hidden_size,
        num_layers=3,
    )

    # create input
    hidden = torch.rand((batch_size, hidden_size))
    state = torch.rand((batch_size, state_size))

    reconst_obs = obs_decoder(
        hidden,
        state,
    )

    _voc, _mel = reconst_obs

    assert _mel.size() == torch.Size([batch_size, mel_channels, feats_T])
    assert _voc.size() == torch.Size([batch_size, v_channels])
