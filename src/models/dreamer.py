from collections import OrderedDict
from functools import partial
from typing import Any

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import kl_divergence
from torch.optim import Optimizer

from ..datamodules import buffer_names
from ..datamodules.replay_buffer import ReplayBuffer
from ..env.array_voc_state import VocStateObsNames as ObsNames
from .abc.agent import Agent
from .abc.controller import Controller
from .abc.observation_auto_encoder import ObservationDecoder, ObservationEncoder
from .abc.prior import Prior
from .abc.transition import Transition
from .abc.world import World


class Dreamer(nn.Module):
    """Dreamer model class."""

    # Added attribute from Trainer
    current_step: int
    current_episode: int
    device: torch.device
    dtype: torch.dtype

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
        """

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

    def collect_experiences(self, env: gym.Env, replay_buffer: ReplayBuffer) -> ReplayBuffer:
        """Explorer in env and collect experiences to replay buffer.

        Args:
            env (gym.Env): PynkTrombone environment or its wrapper class.
            replay_buffer (ReplayBuffer): Storing experiences.
            num_steps (int): How much experiences to store.

        Returns:
            replay_buffer(ReplayBuffer): Same pointer of input replay_buffer.
        """
        device = self.agent.hidden.device
        dtype = self.agent.hidden.dtype

        obs = env.reset()
        voc_state_np = obs[ObsNames.VOC_STATE]
        generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]
        target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]

        voc_state = torch.as_tensor(voc_state_np, dtype, device).squeeze(0)
        generated = torch.as_tensor(generated_np, dtype, device).squeeze(0)
        target = torch.as_tensor(target_np, dtype, device).squeeze(0)

        for _ in range(self.num_collect_experience_steps):
            action = self.agent.explore(obs=(voc_state, generated), target=target)
            action = action.cpu().unsqueeze(0).numpy()
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

            voc_state = torch.as_tensor(voc_state_np, dtype, device).squeeze(0)
            generated = torch.as_tensor(generated_np, dtype, device).squeeze(0)
            target = torch.as_tensor(target_np, dtype, device).squeeze(0)

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
        device = self.agent.hidden.device
        dtype = self.agent.hidden.dtype

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
            action = torch.as_tensor(actions[idx], dtype, device)

            voc_stat = torch.as_tensor(voc_states[idx], dtype, device)
            gened_sound = torch.as_tensor(generated_sounds[idx], dtype, device)
            next_obs = (voc_stat, gened_sound)

            next_hidden = self.transition.forward(hidden, state, action)
            next_state_prior = self.prior.forward(next_hidden)
            next_state_posterior = self.obs_encoder.forward(next_hidden, next_obs)
            next_state = next_state_posterior.rsample()

            all_states[idx] = next_state.detach()
            all_hiddens[idx] = next_hidden.detach()

            rec_voc_stat, rec_gened_sound = self.obs_decoder.forward(next_hidden, next_state)

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
        kl_div_loss /= chunk_size
        rec_loss = rec_voc_state_loss + rec_generated_sound_loss

        loss = rec_loss + (not kl_div_loss.item() < self.free_nats) * kl_div_loss

        loss_dict = {
            "loss": loss,
            "rec_loss": rec_loss,
            "rec_voc_state_loss": rec_voc_state_loss,
            "rec_generated_sound_loss": rec_generated_sound_loss,
            "kl_div_loss": kl_div_loss,
            "over_free_nat": not kl_div_loss.item() < self.free_nats,
        }

        experiences["hiddens"] = all_hiddens
        experiences["states"] = all_states

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

        device = self.device
        dtype = self.dtype

        actions = experiences[buffer_names.ACTION]
        voc_states = experiences[buffer_names.VOC_STATE]
        generated_sounds = experiences[buffer_names.GENERATED_SOUND]
        dones = experiences[buffer_names.DONE]
        target_sounds = experiences[buffer_names.TARGET_SOUND]
        old_hiddens = experiences["hiddens"]
        old_states = experiences["states"]

        chunk_size, batch_size = actions.shape[:2]

        start_idx = np.random.randint(0, chunk_size - self.imagination_horizon, (chunk_size,))
