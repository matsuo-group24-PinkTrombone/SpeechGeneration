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
    predicted_generated: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 1)
    fig.tight_layout()
    labels = {"xlabel": "timestamp", "ylabel": "Hz"}

    data = {
        "Target": target,
        "Generated": generated,
        "Predicted Generated": predicted_generated,
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
    all_generated_pred = []
    obs = env.reset()

    target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]
    voc_state_np = obs[ObsNames.VOC_STATE]
    generated_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]

    target = torch.as_tensor(target_np, dtype=dtype, device=device)
    voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device)
    generated_ref = torch.as_tensor(generated_np, dtype=dtype, device=device)

    # init hidden for world model's prediction
    controller_hidden = agent.controller_hidden
    hidden = agent.hidden

    done = False
    with torch.no_grad():
        while not done:
            # append generated from env and target
            all_target.append(obs[ObsNames.GENERATED_SOUND_SPECTROGRAM])

            action = agent.act(obs=(voc_state, generated_ref), target=target, probabilistic=False)
            action = action.cpu().squeeze(0).numpy()
            obs, _, done, _ = env.step(action)

            all_generated_ref.append(obs[ObsNames.GENERATED_SOUND_SPECTROGRAM])

            generated_ref_np = obs[ObsNames.GENERATED_SOUND_SPECTROGRAM]
            voc_state_np = obs[ObsNames.VOC_STATE]
            target_np = obs[ObsNames.TARGET_SOUND_SPECTROGRAM]

            # append prediction by world model
            state = world.prior(hidden).sample()
            action, controller_hidden = agent.controller(
                hidden, state, target, controller_hidden, False
            )
            _, generated_pred = world.obs_decoder(hidden, state)
            all_generated_pred.append(generated_pred.cpu().numpy())
            next_hidden = world.transition(hidden, state, action)

            hidden = next_hidden
            generated_ref = torch.as_tensor(generated_ref_np, dtype=dtype, device=device)
            target = torch.as_tensor(target_np, dtype=dtype, device=device)
            voc_state = torch.as_tensor(voc_state_np, dtype=dtype, device=device)

        target_spect = np.concatenate(all_target, axis=-1)
        generated_ref_spect = np.concatenate(all_generated_ref, axis=-1)
        generated_pred_spect = np.concatenate(all_generated_pred, axis=-1)

        fig = make_spectrogram_figure(target_spect, generated_ref_spect, generated_pred_spect)
        tensorboard.add_figure(tag, fig, global_step=global_step)
