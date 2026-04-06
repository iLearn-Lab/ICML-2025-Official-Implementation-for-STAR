import numpy as np
from tqdm import tqdm
import wandb

import star.utils.metaworld_utils as mu


class MetaWorldRunner:
    def __init__(
        self,
        env_factory,
        benchmark_name,
        mode,
        rollouts_per_env,
        max_episode_length,
        env_names=None,
        fps=10,
    ):
        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.benchmark = mu.get_benchmark(benchmark_name)
        self.mode = mode
        self.env_names = mu.get_env_names(benchmark_name, mode, env_names=env_names)
        self.rollouts_per_env = rollouts_per_env
        self.max_episode_length = max_episode_length
        self.fps = fps

    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None):
        successes = []
        rewards = []
        per_env_any_success = []
        per_env_success_rates = {}
        videos = {}

        for env_name in tqdm(self.env_names, disable=not do_tqdm):
            any_success = False
            env_successes = []
            env_video = []

            for rollout_index, (success, total_reward, episode) in enumerate(
                self.run_policy_in_env(env_name, policy, render=n_video > 0)
            ):
                any_success = any_success or bool(success)
                successes.append(float(success))
                rewards.append(float(total_reward))
                env_successes.append(float(success))

                if rollout_index < n_video:
                    if save_video_fn is not None:
                        video_hwc = np.asarray(episode["render"])
                        video_chw = video_hwc.transpose((0, 3, 1, 2))
                        save_video_fn(video_chw, env_name, rollout_index)
                    else:
                        env_video.extend(episode["render"])

            per_env_success_rates[env_name] = float(np.mean(env_successes))
            per_env_any_success.append(any_success)

            if env_video:
                video_hwc = np.asarray(env_video)
                video_chw = video_hwc.transpose((0, 3, 1, 2))
                videos[env_name] = wandb.Video(video_chw, fps=self.fps)

        output = {
            "rollout": {
                "overall_success_rate": float(np.mean(successes)),
                "overall_average_reward": float(np.mean(rewards)),
                "environments_solved": int(np.sum(per_env_any_success)),
            },
            "rollout_success_rate": per_env_success_rates,
        }
        if videos:
            output["rollout_videos"] = videos
        return output

    def run_policy_in_env(self, env_name, policy, render=False):
        env = self.env_factory(env_name=env_name)
        try:
            tasks = mu.get_tasks(self.benchmark, self.mode)
            env_tasks = [task for task in tasks if mu.to_repo_env_name(task.env_name) == env_name]

            for rollout_index in range(self.rollouts_per_env):
                if env_tasks:
                    env.set_task(env_tasks[rollout_index % len(env_tasks)])
                yield self.run_episode(env, env_name, policy, render)
        finally:
            env.close()

    def run_episode(self, env, env_name, policy, render=False):
        obs, _ = env.reset()

        if hasattr(policy, "get_action"):
            policy.reset()
            policy_object = policy

            def policy(obs_batch, task_id):
                return policy_object.get_action(obs_batch, task_id)

        success = False
        total_reward = 0.0
        task_id = mu.get_index(env_name)

        episode = {key: [value[-1]] for key, value in obs.items()}
        episode["actions"] = []
        episode["terminated"] = []
        episode["truncated"] = []
        episode["reward"] = []
        episode["success"] = []
        if render:
            episode["render"] = [env.render()]

        for _ in range(self.max_episode_length):
            batched_obs = {key: np.expand_dims(value, 0) for key, value in obs.items()}
            action = policy(batched_obs, task_id).squeeze()
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, info = env.step(action)

            obs = next_obs
            total_reward += reward
            success = success or int(info["success"]) == 1

            for key, value in obs.items():
                episode[key].append(value[-1])
            episode["actions"].append(action)
            episode["terminated"].append(terminated)
            episode["truncated"].append(truncated)
            episode["reward"].append(reward)
            episode["success"].append(info["success"])
            if render:
                episode["render"].append(env.render())

            if terminated or truncated:
                break

        return success, total_reward, {key: np.asarray(value) for key, value in episode.items()}
