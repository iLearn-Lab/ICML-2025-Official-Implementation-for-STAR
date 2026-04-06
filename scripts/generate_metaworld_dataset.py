import json
import os

import h5py
import hydra
from hydra.utils import instantiate
from moviepy.editor import ImageSequenceClip
import numpy as np
from tqdm import tqdm

import star.utils.metaworld_utils as mu
import star.utils.utils as utils


@hydra.main(config_path="../config", config_name="collect_data", version_base=None)
def main(cfg):
    env_runner = instantiate(cfg.task.env_runner)
    data_dir = os.path.join(cfg.data_prefix, cfg.task.suite_name, cfg.task.benchmark_name, cfg.task.mode)
    os.makedirs(data_dir, exist_ok=True)

    experiment_dir, _ = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    success_rates = {}
    average_returns = {}
    expert = mu.get_expert()

    def noisy_expert(obs, task_id):
        expert_action = expert(obs, task_id)
        return np.clip(np.random.normal(expert_action, cfg.task.demo_noise), -1, 1)

    env_names = mu.get_env_names(cfg.task.benchmark_name, cfg.task.mode, env_names=cfg.task.env_names)
    for env_name in env_names:
        file_path = os.path.join(data_dir, f"{env_name}.hdf5")
        if os.path.exists(file_path):
            print(f"{file_path} already exists. Skipping")
            continue

        video_dir = os.path.join(experiment_dir, env_name)
        os.makedirs(video_dir, exist_ok=True)
        init_hdf5(file_path, env_name)

        completed = 0
        total_return = 0.0
        num_rollouts = 0

        rollouts = env_runner.run_policy_in_env(env_name, noisy_expert)
        for rollout_index, (success, episode_return, episode) in tqdm(
            enumerate(rollouts),
            total=cfg.rollout.rollouts_per_env,
        ):
            completed += int(success)
            total_return += episode_return
            num_rollouts += 1

            save_path = os.path.join(video_dir, f"trial_{rollout_index}.mp4")
            clip = ImageSequenceClip(list(episode["corner_rgb"]), fps=24)
            clip.write_videofile(save_path, fps=24, verbose=False, logger=None)
            dump_demo(episode, file_path, rollout_index)

        if num_rollouts == 0:
            raise RuntimeError(f"No demonstrations were collected for MetaWorld task {env_name}.")

        success_rates[env_name] = completed / num_rollouts
        average_returns[env_name] = total_return / num_rollouts
        print(f"{env_name}: success_rate={success_rates[env_name]:.3f}")

    with open(os.path.join(data_dir, "success_rates.json"), "w", encoding="utf-8") as file:
        json.dump(success_rates, file, indent=2, sort_keys=True)
    with open(os.path.join(data_dir, "returns.json"), "w", encoding="utf-8") as file:
        json.dump(average_returns, file, indent=2, sort_keys=True)


def init_hdf5(file_path, env_name):
    with h5py.File(file_path, "w") as file:
        group_data = file.create_group("data")
        group_data.attrs["total"] = 0
        group_data.attrs["env_args"] = json.dumps(
            {
                "env_name": env_name,
                "env_type": 2,
                "env_kwargs": {"render_mode": "rgb_array", "camera_name": "corner2"},
            }
        )


def dump_demo(demo, file_path, demo_index):
    with h5py.File(file_path, "a") as file:
        group_data = file["data"]
        group = group_data.create_group(f"demo_{demo_index}")

        demo_length = demo["actions"].shape[0]
        group_data.attrs["total"] = group_data.attrs["total"] + demo_length
        group.attrs["num_samples"] = demo_length

        non_obs_keys = ("actions", "terminated", "truncated", "reward", "success")
        group.create_dataset("states", data=np.empty((0,), dtype=np.float32))

        for key, value in demo.items():
            if key in non_obs_keys:
                continue
            group.create_dataset(f"obs/{key}", data=value)

        for key in non_obs_keys:
            group.create_dataset(key, data=demo[key])


if __name__ == "__main__":
    main()
