from collections import OrderedDict
from functools import partial
from typing import Any, Optional

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as audioF
from gym.spaces import Box
from torch import Tensor
from torch.distributions import kl_divergence
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ..datamodules import buffer_names
from ..datamodules.replay_buffer import ReplayBuffer
from ..env.array_voc_state import VocStateObsNames as ObsNames
from ..env.array_voc_state import VocStateObsNames as VSON
from ..utils.visualize import visualize_model_approximation
from .abc.agent import Agent
from .abc.controller import Controller
from .abc.observation_auto_encoder import ObservationDecoder, ObservationEncoder
from .abc.prior import Prior
from .abc.transition import Transition
from .abc.world import World


class Dreamer(nn.Module):
    """Dreamer model class."""

    # Added attribute from Trainer
    current_step: int = 0
    current_episode: int = 0
    device: torch.device = "cpu"
    dtype: torch.dtype = torch.float32
    tensorboard: SummaryWriter
    log_every_n_steps: int = 1  # intervals for logging

    def __init__(
        self,
        transition: Transition,
        prior: Prior,
        obs_encoder: ObservationEncoder,
        obs_decoder: ObservationDecoder,
        controller: Controller,
        world: partial[World],
        agent: partial[Agent],
        world_optimizer: partial[Optimizer],
        controller_optimizer: partial[Optimizer],
        free_nats: float = 3.0,
        num_collect_experience_steps: int = 100,
        imagination_horizon: int = 32,
        evaluation_steps: int = 44 * 60,
        evaluation_blank_length: int = 44100,
        sample_rate: int = 44100,
        coef_spectrogram_loss: float = 1.0,
        coef_latent_space_loss: float = 0.0,
        coef_mfcc_loss: float = 0.0,
        n_mfcc: int = 40,
        n_mels: int = 80,
        mfcc_dct_norm: Optional[str] = "ortho",
        mfcc_lifter_size: int = 12,
    ) -> None:
        """
        Args:
            transition (Transition): Instance of ransition model class.
            prior (Prior): Instance of prior model class.
            obs_encoder (ObservationEncoder): Instance of ObservationEncoder model class.
            obs_decoder (ObservationDecoder): Instance of ObservationDecoder model class.
            controller (Controller): Instance of Controller model class.
            world (partial[World]): Partial instance of World interface class.
            agent (partial[Agent]): Partial instance of Agent interface class.
            world_optimizer (partial[Optimizer]): Partial instance of Optimizer class.
            controller_optimizer (partial[Optimizer]): Partial instance of Optimizer class.

            free_nats (float): Ignore kl div loss when it is less then this value.
            num_collect_experience_steps: Specifies the number of times the experiences are stored.
            imagination_horizon: Specifies the number of state transitions that controller needs for learning.
            evaluation_steps: Specifies the number of evaluations.
            evaluation_blank_length (int):The blank lengths of generated/target sound.
            sample_rate (int): The generation sampling rate of Vocal Tract Model.
            coef_spectrogram_loss (float): Coefficient for loss between target and generated spectrogram. (default 1.0).
            coef_latent_space_loss (float): Coefficient for loss on latent space between target and generated.
                Default value is 0.0 for backward compatibility.
            coef_mfcc_loss (float): Coefficient for loss between target and generated mfcc.
                Default value is 0.0 for backward compatibility.
            n_mfcc (int): MFCC channels size.
            n_mels (int): Mel Spectrogram channel size.
            mfcc_dct_norm (Optional[str]): Following to torchaudio MFCC implementation.
            mfcc_lifter_size (int): Low path filter for mfcc loss
        """

        super().__init__()
        self.transition = transition
        self.prior = prior
        self.obs_encoder = obs_encoder
        self.obs_decoder = obs_decoder
        self.controller = controller

        self.world = world(
            transition=transition, prior=prior, obs_encoder=obs_encoder, obs_decoder=obs_decoder
        )

        self.agent = agent(
            controller=controller,
            transition=transition,
            obs_encoder=obs_encoder,
        )

        self.world_optimizer = world_optimizer
        self.controller_optimizer = controller_optimizer

        self.free_nats = free_nats
        self.num_collect_experience_steps = num_collect_experience_steps
        self.imagination_horizon = imagination_horizon
        self.evaluation_steps = evaluation_steps
        self.evaluation_blank_length = evaluation_blank_length
        self.sample_rate = sample_rate
        self.coef_spectrogram_loss = coef_spectrogram_loss
        self.coef_latent_space_loss = coef_latent_space_loss
        self.coef_mfcc_loss = coef_mfcc_loss
        self.mfcc_lifter_size = mfcc_lifter_size

        self.dct_mat: Tensor
        self.register_buffer("dct_mat", audioF.create_dct(n_mfcc, n_mels, mfcc_dct_norm), False)

    def configure_optimizers(self) -> tuple[Optimizer, Optimizer]:
        """Configure world optimizer and controller optimizer.

        Returns:
            world_optimizer (Optimizer): Updating Transition, Prior, ObservationEncoder and Decoder.
            controller_optimizer (Optimizer): Updating Controller.
        """
        world_params = (
            list(self.transition.parameters())
            + list(self.prior.parameters())
            + list(self.obs_encoder.parameters())
            + list(self.obs_decoder.parameters())
        )

        world_optim = self.world_optimizer(params=world_params)
        con_optim = self.controller_optimizer(params=self.controller.parameters())

        return [world_optim, con_optim]

    def configure_replay_buffer_space(self, env: gym.Env) -> dict[str, Box]:
        """Configure replay buffer space to store experiences.

        Args:
            env (gym.Env): PynkTrombone environment or its wrapper class.

        Returns:
            spaces: Replay buffer storing spaces.
        """
        action_box = env.action_space
        vocal_state_box = env.observation_space[VSON.VOC_STATE]
        target_sound_box = env.observation_space[VSON.TARGET_SOUND_SPECTROGRAM]
        generated_sound_box = env.observation_space[VSON.GENERATED_SOUND_SPECTROGRAM]
        spaces = {}
        spaces[buffer_names.ACTION] = action_box
        spaces[buffer_names.VOC_STATE] = vocal_state_box
        spaces[buffer_names.GENERATED_SOUND] = target_sound_box
        spaces[buffer_names.TARGET_SOUND] = generated_sound_box
        spaces[buffer_names.DONE] = Box(0, 1, shape=(1,), dtype=bool)

        return spaces

    @torch.no_grad()
    def collect_experiences(self, env: gym.Env, replay_buffer: ReplayBuffer) -> ReplayBuffer:
        """Explorer in env and collect experiences to replay buffer.

        Args:
            env (gym.Env): PynkTrombone environment or its wrapper class.
            replay_buffer (ReplayBuffer): Storing experiences.

        Returns:
            replay_buffer(ReplayBuffer): Same pointer of input replay_buffer.
        """
        device = self.device
        dtype = self.dtype

        obs = env.reset()
        voc_state_np = obs[ObsNames.VOC_STATE]
        generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]
        target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]

        voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device).unsqueeze(0)
        generated = torch.as_tensor(generated_np, dtype=dtype, device=device).unsqueeze(0)
        target = torch.as_tensor(target_np, dtype=dtype, device=device).unsqueeze(0)

        for _ in range(self.num_collect_experience_steps):
            action = self.agent.explore(obs=(voc_state, generated), target=target)
            action = action.cpu().squeeze(0).numpy()
            obs, _, done, _ = env.step(action)

            voc_state_np = obs[ObsNames.VOC_STATE]
            generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]
            target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]
            sample = {
                buffer_names.ACTION: action,
                buffer_names.VOC_STATE: voc_state_np,
                buffer_names.GENERATED_SOUND: generated_np,
                buffer_names.DONE: done,
                buffer_names.TARGET_SOUND: target_np,
            }

            replay_buffer.push(sample)

            if done:
                obs = env.reset()
                voc_state_np = obs[ObsNames.VOC_STATE]
                generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]
                target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]

            voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device).unsqueeze(0)
            generated = torch.as_tensor(generated_np, dtype=dtype, device=device).unsqueeze(0)
            target = torch.as_tensor(target_np, dtype=dtype, device=device).unsqueeze(0)

    def world_training_step(
        self, experiences: dict[str, np.ndarray]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Compute loss for training world model, and add all hiddens and states to `experiences`
        for controller training.

        Args:
            experiences (dict[str, np.ndarray]): Collected experiences.

        Returns:
            loss_dict (dict[str, Any]): loss and some other metric values.
            experiences (dict[str, np.ndarray]): Added `all_hiddens` and `all_states`.
        """
        device = self.device
        dtype = self.dtype

        self.world.train()

        actions = experiences[buffer_names.ACTION]
        voc_states = experiences[buffer_names.VOC_STATE]
        generated_sounds = experiences[buffer_names.GENERATED_SOUND]
        dones = experiences[buffer_names.DONE]

        chunk_size, batch_size = actions.shape[:2]

        hidden = torch.zeros(
            (batch_size, *self.transition.hidden_shape), dtype=dtype, device=device
        )
        state = torch.zeros((batch_size, *self.prior.state_shape), dtype=dtype, device=device)

        all_hiddens = torch.empty((chunk_size, *hidden.shape), dtype=dtype, device="cpu")
        all_states = torch.empty((chunk_size, *state.shape), dtype=dtype, device="cpu")

        rec_voc_state_loss = 0.0
        rec_generated_sound_loss = 0.0
        all_kl_div_loss = 0.0

        for idx in range(chunk_size):
            action = torch.as_tensor(actions[idx], dtype=dtype, device=device)

            voc_stat = torch.as_tensor(voc_states[idx], dtype=dtype, device=device)
            gened_sound = torch.as_tensor(generated_sounds[idx], dtype=dtype, device=device)
            next_obs = (voc_stat, gened_sound)

            next_state_prior, next_state_posterior, next_hidden = self.world.forward(
                hidden, state, action, next_obs
            )

            next_state = next_state_posterior.rsample()
            rec_voc_stat, rec_gened_sound = self.obs_decoder.forward(next_hidden, next_state)
            all_states[idx] = next_state.detach()
            all_hiddens[idx] = next_hidden.detach()

            # compute losses
            kl_div_loss = kl_divergence(next_state_posterior, next_state_prior).view(
                batch_size, -1
            )
            all_kl_div_loss += kl_div_loss.sum(-1).mean()
            rec_voc_state_loss += F.mse_loss(voc_stat, rec_voc_stat)
            rec_generated_sound_loss += F.mse_loss(gened_sound, rec_gened_sound)

            # next step
            is_done = dones[idx].reshape(-1)
            next_state = torch.stack(
                [
                    torch.zeros_like(next_state[i]) if d else next_state[i]
                    for i, d in enumerate(is_done)
                ]
            )  # Initialize with zero.
            next_hidden = torch.stack(
                [
                    torch.zeros_like(next_hidden[i]) if d else next_hidden[i]
                    for i, d in enumerate(is_done)
                ]
            )  # Initialize with zero.

            state = next_state
            hidden = next_hidden

        rec_voc_state_loss /= chunk_size
        rec_generated_sound_loss /= chunk_size
        all_kl_div_loss /= chunk_size
        rec_loss = rec_voc_state_loss + rec_generated_sound_loss

        if all_kl_div_loss.item() < self.free_nats:
            loss = rec_loss
        else:
            loss = all_kl_div_loss + rec_loss

        loss_dict = {
            "loss": loss,
            "rec_loss": rec_loss,
            "rec_voc_state_loss": rec_voc_state_loss,
            "rec_generated_sound_loss": rec_generated_sound_loss,
            "kl_div_loss": all_kl_div_loss,
            "over_free_nat": not all_kl_div_loss.item() < self.free_nats,
        }

        experiences["hiddens"] = all_hiddens
        experiences["states"] = all_states

        prefix = "world_training_step/"
        self.log(prefix + "loss", loss)
        self.log(prefix + "reconstruction loss", rec_loss)
        self.log(prefix + "reconstructed vocal state loss", rec_voc_state_loss)
        self.log(prefix + "reconstructed generated sound loss", rec_generated_sound_loss)
        self.log(prefix + "kl divergence loss", all_kl_div_loss)
        self.log(prefix + "is over free nat", float(all_kl_div_loss.item() > self.free_nats))

        return loss_dict, experiences

    def controller_training_step(
        self, experiences: dict[str, np.ndarray]
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        """Compute loss for training controller model.
        Args:
            experiences (dict[str, np.ndarray]): Collected experiences.

        Returns:
            loss_dict (dict[str, Any]): loss and some other metric values.
            experiences (dict[str, np.ndarray]): Collected experiences (No modification.)
        """

        self.controller.train()
        self.world.eval()

        device = self.device
        dtype = self.dtype

        actions = experiences[buffer_names.ACTION]
        dones = experiences[buffer_names.DONE]
        target_sounds = experiences[buffer_names.TARGET_SOUND]
        old_hiddens = experiences["hiddens"]

        chunk_size, batch_size = actions.shape[:2]
        start_indices = np.random.randint(0, chunk_size - self.imagination_horizon, (batch_size,))
        batch_arange = np.arange(batch_size)
        hidden = torch.as_tensor(
            old_hiddens[start_indices, batch_arange], dtype=dtype, device=device
        )
        controller_hidden = torch.zeros(
            batch_size, *self.controller.controller_hidden_shape, dtype=dtype, device=device
        )
        state = self.prior.forward(hidden).sample()

        latent_space_loss = 0.0
        spectrogram_loss = 0.0
        mfcc_loss = 0.0
        for horizon in range(self.imagination_horizon):
            indices = start_indices + horizon
            target = torch.as_tensor(
                target_sounds[indices, batch_arange], dtype=dtype, device=device
            )
            action, controller_hidden = self.controller.forward(
                hidden, state, target, controller_hidden, probabilistic=True
            )
            next_hidden = self.transition.forward(hidden, state, action)
            next_state = self.prior.forward(next_hidden).sample()
            rec_next_obs = self.obs_decoder.forward(next_hidden, next_state)
            rec_voc_state, rec_gened_sound = rec_next_obs

            target_latent = self.obs_encoder.embed_observation((rec_voc_state, target))

            rec_generated_latent = self.obs_encoder.embed_observation(
                (rec_voc_state, rec_gened_sound)
            )

            spectrogram_loss += F.mse_loss(target, rec_gened_sound)

            latent_space_loss += F.mse_loss(target_latent, rec_generated_latent)

            # MFCC loss
            # shape: (B, C, L) -> (B, L, C) -> (C, L, B)
            target_mfcc = (target.transpose(-1, -2) @ self.dct_mat).transpose(-1, 0)[
                : self.mfcc_lifter_size
            ]
            rec_gened_mfcc = (rec_gened_sound.transpose(-1, -2) @ self.dct_mat).transpose(-1, 0)[
                : self.mfcc_lifter_size
            ]
            mfcc_loss += F.mse_loss(target_mfcc, rec_gened_mfcc)

            hidden = next_hidden

            is_done = dones[indices, batch_arange].reshape(-1)
            controller_hidden = torch.stack(
                [
                    torch.zeros_like(controller_hidden[i]) if d else controller_hidden[i]
                    for i, d in enumerate(is_done)
                ]
            )
            hidden = torch.stack(
                [
                    torch.as_tensor(old_hiddens[indices[i] + 1, i], device=device, dtype=dtype)
                    if d
                    else hidden[i]
                    for i, d in enumerate(is_done)
                ]
            )
            state = self.prior.forward(hidden).sample()

        latent_space_loss /= self.imagination_horizon
        spectrogram_loss /= self.imagination_horizon
        mfcc_loss /= self.imagination_horizon

        loss = (
            self.coef_spectrogram_loss * spectrogram_loss
            + self.coef_latent_space_loss * latent_space_loss
            + self.coef_mfcc_loss * mfcc_loss
        )

        loss_dict = {
            "loss": loss,
            "spectrogram_loss": spectrogram_loss,
            "latent_space_loss": latent_space_loss,
            "mfcc_loss": mfcc_loss,
        }

        prefix = "controller_training_step/"
        self.log(prefix + "loss", loss)
        self.log(prefix + "spectrogram_loss", loss)
        self.log(prefix + "latent_space_loss", latent_space_loss)
        self.log(prefix + "mfcc_loss", mfcc_loss)

        return loss_dict, experiences

    @torch.no_grad()
    def evaluation_step(self, env: gym.Env) -> dict[str, Any]:
        """Evaluation step.
        Args:
            env (gym.Env): PynkTrombone environment or its wrapper class.

        Returns:
            loss_dict (dict[str, Any]): Returned metric values.
        """
        self.world.eval()
        self.controller.eval()

        device = self.device
        dtype = self.dtype

        self.agent.reset()

        obs = env.reset()
        voc_state_np = obs[ObsNames.VOC_STATE]
        generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]
        target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]

        voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device).unsqueeze(0)
        generated = torch.as_tensor(generated_np, dtype=dtype, device=device).unsqueeze(0)
        target = torch.as_tensor(target_np, dtype=dtype, device=device).unsqueeze(0)

        generated_sound_waves = []
        target_sound_waves = []

        blank = np.zeros(self.evaluation_blank_length)

        target_generated_mse = 0.0
        target_generated_mae = 0.0

        for i in range(self.evaluation_steps):
            target_sound_waves.append(obs[ObsNames.TARGET_SOUND_WAVE])

            action = self.agent.act(obs=(voc_state, generated), target=target, probabilistic=False)
            action = action.cpu().squeeze(0).numpy()
            obs, _, done, _ = env.step(action)

            generated_sound_waves.append(obs[ObsNames.GENERATED_SOUND_WAVE])
            generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]

            target_generated_mse += np.mean((target_np - generated_np) ** 2)
            target_generated_mae += np.mean(np.abs(target_np - generated_np))

            if done:
                obs = env.reset()
                generated_sound_waves.append(blank)
                target_sound_waves.append(blank)
                self.agent.reset()

            voc_state_np = obs[ObsNames.VOC_STATE]
            generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]
            target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]

            voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device).unsqueeze(0)
            generated = torch.as_tensor(generated_np, dtype=dtype, device=device).unsqueeze(0)
            target = torch.as_tensor(target_np, dtype=dtype, device=device).unsqueeze(0)

        target_generated_mae /= self.evaluation_steps
        target_generated_mse /= self.evaluation_steps

        generated_sounds_for_log = np.concatenate(generated_sound_waves)
        target_sounds_for_log = np.concatenate(target_sound_waves)

        # logging to tensorboard
        prefix = "evaluation_step/"
        self.tensorboard.add_audio(
            prefix + "generated sounds",
            generated_sounds_for_log,
            self.current_step,
            self.sample_rate,
        )
        self.tensorboard.add_audio(
            prefix + "target sounds", target_sounds_for_log, self.current_step, self.sample_rate
        )
        self.log(prefix + "target generated mse", float(target_generated_mse), True)
        self.log(prefix + "target generated mae", float(target_generated_mae), True)

        loss_dict = {
            "target_generated_mse": target_generated_mse,
            "target_generated_mae": target_generated_mae,
        }

        visualize_model_approximation(
            self.world,
            self.agent,
            env,
            self.tensorboard,
            prefix + "target-generation-imagination-spectrograms",
            self.current_step,
            device=self.device,
            dtype=self.dtype,
        )

        return loss_dict

    def log(self, name: str, value: Any, force_logging: bool = False) -> None:
        """Log scalar value to tensorboard.

        `log_every_n_steps` to reduce log data volume.
        Args:
            name (str): Log value name.
            value (Any): scalar value.
            force_logging (bool): If True, Force tensorboard to log.
        """

        if force_logging or self.current_step % self.log_every_n_steps == 0:
            self.tensorboard.add_scalar(name, value, self.current_step)
