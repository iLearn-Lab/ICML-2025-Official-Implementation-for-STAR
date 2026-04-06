import gc
import multiprocessing

import numpy as np
from tqdm import tqdm
import wandb

import star.utils.libero_utils as lu


class LiberoRunner:
    def __init__(
        self,
        env_factory,
        benchmark_name,
        mode,
        rollouts_per_env,
        num_parallel_envs,
        max_episode_length,
        frame_stack=1,
        fps=10,
        task_embedding_format="clip",
        task_embedding_model_path=None,
    ):
        if mode != "all":
            raise ValueError("The cleaned STAR codebase only keeps LIBERO `mode=all`.")

        self.env_factory = env_factory
        self.benchmark_name = benchmark_name
        self.benchmark = lu.get_benchmark(benchmark_name)()
        descriptions = [self.benchmark.get_task(i).language for i in range(self.benchmark.n_tasks)]
        task_embs = lu.get_task_embs(
            task_embedding_format,
            descriptions,
            task_embedding_model_path=task_embedding_model_path,
        )
        self.benchmark.set_task_embs(task_embs)
        self.env_names = self.benchmark.get_task_names()
        self.rollouts_per_env = rollouts_per_env
        self.num_parallel_envs = num_parallel_envs
        self.max_episode_length = max_episode_length
        self.frame_stack = frame_stack
        self.fps = fps

        if num_parallel_envs > 1 and multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)

    def run(self, policy, n_video=0, do_tqdm=False, save_video_fn=None):
        successes = []
        rewards = []
        per_env_any_success = []
        per_env_success_rates = {}
        videos = {}

        for env_name in tqdm(self.env_names, disable=not do_tqdm):
            any_success = False
            env_successes = []
            env_rewards = []
            env_video = []

            for rollout_index, (success, total_reward, episode) in enumerate(
                self.run_policy_in_env(env_name, policy, render=n_video > 0)
            ):
                any_success = any_success or bool(success)
                successes.append(float(success))
                rewards.append(float(total_reward))
                env_successes.append(float(success))
                env_rewards.append(float(total_reward))

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
        env_id = self.env_names.index(env_name)
        all_init_states = self.benchmark.get_task_init_states(env_id)
        total_rollouts = self.rollouts_per_env
        completed = 0

        while completed < total_rollouts:
            current_env_num = min(self.num_parallel_envs, total_rollouts - completed)
            def env_fn():
                return lu.LiberoFrameStack(self.env_factory(env_id, self.benchmark), self.frame_stack)

            env = lu.LiberoVectorWrapper(env_fn, current_env_num)

            indices = np.arange(completed, completed + current_env_num) % all_init_states.shape[0]
            init_states = all_init_states[indices]
            success, total_reward, episode = self.run_episode(
                env,
                env_name,
                policy,
                init_states,
                current_env_num,
                render,
            )
            completed += current_env_num

            for env_index in range(current_env_num):
                episode_slice = {key: value[:, env_index] for key, value in episode.items()}
                yield success[env_index], total_reward[env_index], episode_slice

            env.close()
            gc.collect()

    def run_episode(self, env, env_name, policy, init_states, env_num, render=False):
        obs, _ = env.reset(init_states=init_states)

        if hasattr(policy, "get_action"):
            policy.reset()
            policy_object = policy

            def policy(obs_batch, task_id, task_emb):
                return policy_object.get_action(obs_batch, task_id, task_emb)

        success = np.zeros(env_num, dtype=bool)
        total_reward = np.zeros(env_num, dtype=np.float32)

        episode = {key: [value[:, -1]] for key, value in obs.items()}
        episode["actions"] = []
        if render:
            episode["render"] = [env.render()]

        task_id = self.env_names.index(env_name)
        task_emb = self.benchmark.get_task_emb(task_id).repeat(env_num, 1)

        for _ in range(self.max_episode_length):
            action = policy(obs, task_id, task_emb)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            obs = next_obs

            for key, value in obs.items():
                episode[key].append(value[:, -1])
            episode["actions"].append(action)
            if render:
                episode["render"].append(env.render())

            success = np.logical_or(success, terminated)
            if bool(np.all(success)):
                break

        return success, total_reward, {key: np.asarray(value) for key, value in episode.items()}
