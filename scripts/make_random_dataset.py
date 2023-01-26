import glob
import math
import os
import pathlib
from random import uniform
from typing import Callable

import gym
import numpy as np
import soundfile
import tqdm
from pynktrombonegym.spaces import ObservationSpaceNames as OSN
from pynktrombonegym.wrappers import ActionByAcceleration, Log1pMelSpectrogram

target_sound_files = glob.glob("data/sample_target_sounds/*.wav")
sound_seconds_range = (2.0, 3.0)
data_dir = pathlib.Path(__file__).parent.parent.joinpath("data")
output_dir = "generated_randomly"
sample_dir = "sample_generated_randomly"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def generate_sound(
    environment: gym.Env, action_fn: Callable, file_name: str, generate_chunk: int, sample_rate
) -> None:
    """Generate sound wave with environment and action_fn
    Args:
        environment (env.PynkTrombone): Vocal tract environment.
        action_fn (Callable): Return action for generating waves with environment.
            This function must be able to receive `PynkTrombone` environment and
            to return action.
            Ex:
            >>> def update_fn(environment: gym.Env):
            ...     return action
        file_name (str): The file name of generated sound.
    Returns:
        wave (np.ndarray): Generated wave. 1d array.
    """
    sound_seconds = uniform(*sound_seconds_range)
    roop_num = math.ceil(sound_seconds / (generate_chunk / sample_rate))
    generated_waves = []
    environment.reset()
    done = False
    for _ in range(roop_num):
        if done:
            environment.reset()

        action = action_fn(environment)
        obs, _, done, _ = environment.step(action)  # type: ignore
        generated_sound_wave = obs[OSN.GENERATED_SOUND_WAVE]
        generated_waves.append(generated_sound_wave)

    generated_sound_wave = np.concatenate(generated_waves).astype(np.float32)

    path = os.path.join(data_dir, file_name)
    soundfile.write(path, generated_sound_wave, sample_rate)


if __name__ == "__main__":
    env = Log1pMelSpectrogram(target_sound_files)
    action_scaler = env.generate_chunk / env.sample_rate
    wrapped = ActionByAcceleration(env, action_scaler)

    def action_fn(e: gym.Env) -> dict:
        return e.action_space.sample()

    num_repeat = math.ceil(10 * 60**2 / 2.5)
    # generate sounds for test
    for i in range(5):
        generate_sound(
            wrapped,
            action_fn,
            f"{sample_dir}/{i}.wav",
            env.generate_chunk,
            env.sample_rate,
        )

    # generate sounds for dataset
    for i in tqdm.tqdm(range(num_repeat)):
        generate_sound(
            wrapped, action_fn, f"{output_dir}/{i + 1}.wav", env.generate_chunk, env.sample_rate
        )
    wrapped.close()  # you must call.
