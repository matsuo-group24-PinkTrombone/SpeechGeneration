from collections import OrderedDict
from functools import partial
from typing import Any

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Box
from torch import Tensor
from torch.distributions import kl_divergence
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ..datamodules import buffer_names
from ..datamodules.replay_buffer import ReplayBuffer
from ..env.array_voc_state import VocStateObsNames as ObsNames
from ..env.array_voc_state import VocStateObsNames as VSON
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
    log_every_n_steps: int = 1

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
        evaluation_blank_length: int = 22050,
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

    def configure_replay_buffer(self, env: gym.Env, buffer_size: int) -> ReplayBuffer:
        """Configure replay buffer to store experiences.

        Args:
            env (gym.Env): PynkTrombone environment or its wrapper class.
            buffer_size (int): Max length of experiences you can store.

        Returns:
            ReplayBuffer: Replay buffer that can store experiences.
        """
        action_box = env.action_space
        vocal_state_box = env.observation_space[VSON.VOC_STATE]
        target_sound_box = env.observation_space[VSON.TARGET_SOUND_WAVE]
        generated_sound_box = env.observation_space[VSON.GENERATED_SOUND_WAVE]
        spaces = {}
        spaces[buffer_names.ACTION] = action_box
        spaces[buffer_names.VOC_STATE] = vocal_state_box
        spaces[buffer_names.GENERATED_SOUND] = target_sound_box
        spaces[buffer_names.TARGET_SOUND] = generated_sound_box
        spaces[buffer_names.DONE] = Box(0, 1, shape=(1,), dtype=bool)

        replay_buffer = ReplayBuffer(spaces, buffer_size)

        return replay_buffer

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

            voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device).squeeze(0)
            generated = torch.as_tensor(generated_np, dtype=dtype, device=device).squeeze(0)
            target = torch.as_tensor(target_np, dtype=dtype, device=device).squeeze(0)

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
            next_state[dones[idx]] = 0.0  # Initialize with zero.
            next_hidden[dones[idx]] = 0.0  # Initialize with zero.

            state = next_state
            hidden = next_hidden

        rec_voc_state_loss /= chunk_size
        rec_generated_sound_loss /= chunk_size
        all_kl_div_loss /= chunk_size
        rec_loss = rec_voc_state_loss + rec_generated_sound_loss
        loss = rec_loss + (not all_kl_div_loss.item() < self.free_nats) * all_kl_div_loss

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

        loss = 0.0
        for i in range(self.imagination_horizon):
            indices = start_indices + i
            target = torch.as_tensor(
                target_sounds[indices, batch_arange], dtype=dtype, device=device
            )
            action, controller_hidden = self.controller.forward(
                hidden, state, target, controller_hidden, probabilistic=True
            )
            next_hidden = self.transition.forward(hidden, state, action)
            next_state = self.prior.forward(next_hidden).sample()
            rec_next_obs = self.obs_decoder.forward(next_hidden, next_state)
            _, rec_gened_sound = rec_next_obs

            loss += F.mse_loss(target, rec_gened_sound)

            hidden = next_hidden

            hidden[dones[indices, batch_arange]] = old_hiddens[indices + 1, batch_arange]
            state = self.prior.forward(hidden)

        loss /= self.imagination_horizon

        loss_dict = {"loss": loss}

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
        generated_sounds_for_log
        target_sounds_for_log

        loss_dict = {
            "target_generated_mse": target_generated_mse,
            "target_generated_mae": target_generated_mae,
        }

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
