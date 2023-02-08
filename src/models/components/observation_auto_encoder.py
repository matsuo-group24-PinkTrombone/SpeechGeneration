from typing import Callable, Optional

import torch

from ..abc.observation_auto_encoder import ObservationDecoder as AbsObservationDecoder
from ..abc.observation_auto_encoder import ObservationEncoder as AbsObservationEncoder
from .conformer_decoder_fastspeech2 import ConformerDecoder
from .linear_layers import LinearLayers
from .posterior_encoder_vits import PosteriorEncoderVITS


class ObservationEncoder(AbsObservationEncoder):
    def __init__(
        self,
        mel_encoder: PosteriorEncoderVITS,
        state_size: int,
        hidden_size: int,
        feats_T: int,
        min_logs: float = 0.1,
    ):
        super().__init__()

        self.feats_T = feats_T
        self.state_size = state_size
        self.min_logs = min_logs

        self.obs_embedding_layer = mel_encoder
        self.time_reduction_layer = torch.nn.Linear(feats_T, 1)

        self.fc_mean = LinearLayers(
            input_size=state_size + hidden_size,
            hidden_size=(state_size + hidden_size) * 2,
            layers=3,
            output_size=state_size,
        )
        self.fc_logs = LinearLayers(
            input_size=state_size + hidden_size,
            hidden_size=(state_size + hidden_size) * 2,
            layers=3,
            output_size=state_size,
        )

    def embed_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Embed observation to latent space.

        Args:
            obs (_tensor_or_any): observation data.
                For instance, obs=(voc state v_t, generated sound g_t) <- Tuple of Tensor!
        Returns:
            embedded_obs (_tensor_or_any): Embedded observation.
        """
        # vocal tract state embedding
        vt, mel = obs
        batch_size = mel.size(0)
        z, m, logs, x_mask = self.obs_embedding_layer(
            x=mel,
            x_lengths=torch.tensor([self.feats_T] * batch_size, dtype=torch.float32),
            g=vt.unsqueeze(2),
        )

        embedded_obs = self.time_reduction_layer(m).squeeze(2)

        return embedded_obs

    def encode(
        self, hidden: torch.Tensor, embedded_obs: torch.Tensor
    ) -> torch.distributions.Distribution:
        """Encode hidden state and embedded_obs to world 'state'.
        Args:
            hidden (_tensor_or_any): hidden state `h_t`
            embedded_obs (_tensor_or_any): Embedded obsevation data.

        Returns:
            state (_t_or_any[Distribution]): world state `s_t`.
                Type is `Distiribution` class for computing kl divergence.
        """
        x = torch.concat((hidden, embedded_obs), dim=1)
        mean = self.fc_mean(x)
        logs = torch.nn.functional.softplus(self.fc_logs(x)) + self.min_logs

        return torch.distributions.normal.Normal(mean, logs)

    @property
    def state_shape(self) -> tuple[int]:
        """Returns world 'state' shape.

        Do not contain batch dim.
        """
        return (self.state_size,)


class ObservationDecoder(AbsObservationDecoder):
    def __init__(
        self,
        decoder: ConformerDecoder,
        voc_state_size: int,
        feats_T: int,
        conv_kernel_size: int = 3,
        conv_padding_size: int = 1,
        conv_bias: bool = True,
    ) -> None:
        """
        Args:
            decoder: Decoder to reconstruct the mel spectrogram
        """
        super().__init__()

        # decoder input (hidden_size + stat_size)
        self.decoder = decoder

        self.time_extend_conv = torch.nn.Conv1d(
            in_channels=1,
            out_channels=feats_T,
            kernel_size=conv_kernel_size,
            padding=conv_padding_size,
            bias=conv_bias,
        )

        self.voc_decoder = LinearLayers(
            input_size=self.decoder.idim,
            hidden_size=self.decoder.idim * 2,
            layers=3,
            output_size=voc_state_size,
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode to observation.

        Args:
            hidden (Tensor): hidden state `h_t`.
            state (Tensor): world state `s_t`.
        Returns:
            obs (tuple[Tensor, Tensor]): reconstructed observation (o^_t).
                obs=(voc state v_t, generated sound g_t) is returned.
        """
        concat_input = torch.concat((hidden, state), dim=1)

        time_extend_input = self.time_extend_conv(
            concat_input.unsqueeze(1)  # (batch, 1, channels)
        )

        decoder_input = time_extend_input.transpose(1, 2)  # (batch, channels, feats_T)

        reconst_mel = self.decoder(torch.tanh(decoder_input))

        reconst_voc = self.voc_decoder(concat_input)

        reconst_obs = (reconst_voc, reconst_mel)

        return reconst_obs
