import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from librosa.feature import melspectrogram
from torch.utils.tensorboard import SummaryWriter

from ..env.array_voc_state import VocStateObsNames as ObsNames
from ..models.abc.agent import Agent
from ..models.abc.world import World


def make_spectrogram_figure(
    target: np.ndarray,
    generated: np.ndarray,
    generated_prior: np.ndarray,
    generated_posterior: np.ndarray
) -> None:
    fig, axes = plt.subplots(4, 1)
    fig.tight_layout()
    labels = {"xlabel": "timestamp", "ylabel": "Hz"}

    data = {
        "Target": target,
        "Generated": generated,
        "Prediction(prior)": generated_prior,
        "Prediction(posterior)": generated_posterior
    }
    # show target mel spectrogram
    for i, (title, spect) in enumerate(data.items()):
        mappable = axes[i].imshow(spect, aspect="auto")
        axes[i].set(**labels, title=title)
        fig.colorbar(mappable, ax=axes[i])

    return fig


@torch.no_grad()
def visualize_model_approximation(
    world: World,
    agent: Agent,
    env: gym.Env,
    tensorboard: SummaryWriter,
    tag: str,
    global_step: int,
    visualize_per_episode: bool = True,
    visualize_steps: int = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> None:
    """
    Args:
        world_model (torch.nn.Module): model that need to be checked
        env (gym.env): gym.env or it's wrapper class
        tensorboard(SummaryWriter): visualized figures is stored to this tensorboard
        visualize_per_episode(bool): you can select whether to visualize one episode or a fixed number of steps.
    Returns:
        None
    """
    all_target = []
    all_generated_ref = []
    all_generated_prior = []
    all_generated_posterior=[]
    obs = env.reset()

    target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]
    voc_state_np = obs[ObsNames.VOC_STATE]
    generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]

    target = torch.as_tensor(target_np, dtype=dtype, device=device).unsqueeze(0)  # t_1
    voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device).unsqueeze(0)  # v_0
    generated_ref = torch.as_tensor(generated_np, dtype=dtype, device=device).unsqueeze(0)  # g_0

    state_prior = torch.zeros((1, *world.prior.state_shape), dtype=dtype, device=device)  # s_0
    state_posterior = torch.zeros((1, *world.prior.state_shape), dtype=dtype, device=device)

    # init hidden for world model's prediction
    agent.reset()
    hidden = agent.hidden  # h_0

    done = False
    while not done:
        all_target.append(target_np)  # T_t+1

        # append prediction via prior
        action = agent.act(
            obs=(voc_state, generated_ref), target=target, probabilistic=False
        )  # a_t
        hidden = world.transition(hidden, state_prior, action)  # h_t+1
        state_prior = world.prior(hidden).sample()  # s_t+1
        _, generated_prior = world.obs_decoder(hidden, state_prior)  # g_t+1(prior)
        all_generated_prior.append(generated_prior.cpu().numpy())

        # append generated from env
        obs, _, done, _ = env.step(action.squeeze(0).cpu().numpy())  # o_t+1

        all_generated_ref.append(obs[ObsNames.GENERATED_SOUND_SPECTROGRAM])

        generated_ref_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]  # g_t+1(ref)
        voc_state_np = obs[ObsNames.VOC_STATE]
        target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]  # T_t+2

        generated_ref = torch.as_tensor(generated_ref_np, dtype=dtype, device=device).unsqueeze(0)
        target = torch.as_tensor(target_np, dtype=dtype, device=device).unsqueeze(0)
        voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device).unsqueeze(0)

        # append prediction via posterior
        observation = (voc_state, generated_ref)
        state_posterior = world.obs_encoder(hidden, observation).sample() # s_t+1(posterior)
        _, generated_pred_posterior = world.obs_decoder(hidden, state_posterior) # g_t+1(posterior)
        all_generated_posterior.append(generated_pred_posterior.cpu().numpy())

    target_spect = np.concatenate(all_target, axis=-1)
    generated_ref_spect = np.concatenate(all_generated_ref, axis=-1)
    generated_prior_spect = np.concatenate(all_generated_prior, axis=-1).squeeze(0)
    generated_posterior_spect = np.concatenate(all_generated_posterior, axis=-1).squeeze(0)

    fig = make_spectrogram_figure(target_spect, generated_ref_spect, generated_prior_spect, generated_posterior_spect)
    tensorboard.add_figure(tag, fig, global_step=global_step)
