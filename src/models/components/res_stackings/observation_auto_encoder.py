import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Normal

from ...abc.observation_auto_encoder import ObservationDecoder as AbsObservationDecoder
from ...abc.observation_auto_encoder import ObservationEncoder as AbsObservationEncoder
from ..conformer_decoder_fastspeech2 import ConformerDecoder
from ..posterior_encoder_vits import PosteriorEncoderVITS
from ..res_layers import ResLayers


class ObservationEncoder(AbsObservationEncoder):
    def __init__(
        self,
        mel_encoder: PosteriorEncoderVITS,
        state_size: int,
        hidden_size: int,
        feats_T: int,
        stacked_res_hidden_size: int,
        num_layers: int = 1,
        min_logs: float = 0.1,
        bias: bool = True,
    ) -> None:
        """
        Args:
            mel_encoder (PosteriorEncoderVITS): Mel spectrogram encoder.
            state_size (int): The size of state(s_t)
            hidden_size (int): The size of hidden state(h_t)
            feats_T (int): Time step length of mel spectrogram.
            stacked_linear_hidden_size (int): The hidden size of `fc_input_layer`.
            num_layers (int): The number of internal layers (For stacked linear).
            min_logs (float, optional): inimum value of standard deviation. Defaults to 0.1.
            bias (bool): Bias of `nn.Linear`.
        """
        super().__init__()

        self.feats_T = feats_T
        self.state_size = state_size
        self.min_logs = min_logs

        self.obs_embedding_layer = mel_encoder
        self.time_reduction_layer = torch.nn.Linear(feats_T, 1)

        self.fc_input_layer = ResLayers(
            hidden_size + state_size,
            stacked_res_hidden_size,
            num_layers,
            hidden_size,
            bias,
        )

        self.fc_mean = nn.Linear(hidden_size, state_size, bias)
        self.fc_logs = nn.Linear(hidden_size, state_size, bias)

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
        _, m, _, _ = self.obs_embedding_layer(
            x=mel,
            x_lengths=torch.tensor([self.feats_T] * batch_size, dtype=torch.float32),
            g=vt.unsqueeze(2),
        )

        embedded_obs = self.time_reduction_layer(m).squeeze(2)

        return embedded_obs

    def encode(self, hidden: torch.Tensor, embedded_obs: torch.Tensor) -> Distribution:
        """Encode hidden state and embedded_obs to world 'state'.
        Args:
            hidden (_tensor_or_any): hidden state `h_t`
            embedded_obs (_tensor_or_any): Embedded obsevation data.

        Returns:
            state (_t_or_any[Distribution]): world state `s_t`.
                Type is `Distiribution` class for computing kl divergence.
        """
        x = torch.concat((hidden, embedded_obs), dim=1)
        x = self.fc_input_layer(x)
        mean = self.fc_mean(x)
        logs = F.softplus(self.fc_logs(x)) + self.min_logs

        return Normal(mean, logs)

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
        stacked_linear_hidden_size: int,
        num_layers: int = 1,
        conv_kernel_size: int = 3,
        conv_padding_size: int = 1,
        conv_bias: bool = True,
    ) -> None:
        """
        Args:
            decoder (ConformerDecoder): Decoder to reconstruct the mel spectrogram
            voc_state_size (int): The size of vocal state array.
            feats_T (int): Time step length of mel spectrogram.
            stacked_linear_hidden_size (int): The hidden size of `fc_input_layer`.
            num_layers (int): The number of internal layers (For stacked linear).
            conv_kernel_size (int): Input convolution kernel size.
            conv_padding_size (int): Input convolution padding size.
            conv_bias (bool): Whether the input convolution has bias or not.
            bias (bool): Bias of `nn.Linear`.
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

        self.voc_decoder = ResLayers(
            input_size=self.decoder.idim,
            hidden_size=stacked_linear_hidden_size,
            layers=num_layers,
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
                obs=(voc state v_t, generated sound g_t) will be returned.
        """
        concat_input = torch.concat((hidden, state), dim=1)

        time_extend_input = self.time_extend_conv(
            concat_input.unsqueeze(1)  # (batch, 1, channels)
        )

        decoder_input = time_extend_input.transpose(1, 2)  # (batch, channels, feats_T)

        reconst_mel = self.decoder(torch.relu(decoder_input))

        reconst_voc = self.voc_decoder(concat_input)

        reconst_obs = (reconst_voc, reconst_mel)

        return reconst_obs
