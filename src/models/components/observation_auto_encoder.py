import torch

from ..abc.observation_auto_encoder import ObservationEncoder as AbsObservationEncoder
from ..components.posterior_encoder_vits import PosteriorEncoderVITS


class ObservationEncoder(AbsObservationEncoder):
    def __init__(
        self,
        mel_encoder: PosteriorEncoderVITS,
        state_size: int,
        hiddden_size: int,
        feats_T: int,
        min_logs: float = 0.1,
    ):
        super().__init__()

        self.feats_T = feats_T
        self.state_size = state_size
        self.min_logs = min_logs

        self.obs_embedding_layer = mel_encoder
        self.time_reduction_layer = torch.nn.Linear(feats_T, 1)

        self.fc_mean = torch.nn.Linear(state_size + hiddden_size, state_size)
        self.fc_logs = torch.nn.Linear(state_size + hiddden_size, state_size)

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
