import copy
import json
import os

import hydra
from hydra.utils import instantiate
from moviepy.editor import ImageSequenceClip
from omegaconf import OmegaConf
from pyinstrument import Profiler
import torch

import star.utils.utils as utils


OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_checkpoint_path(cfg):
    if cfg.checkpoint_path is not None:
        return utils.get_latest_checkpoint(cfg.checkpoint_path)

    checkpoint_cfg = copy.deepcopy(cfg)
    checkpoint_cfg.make_unique_experiment_dir = False
    checkpoint_dir, _ = utils.get_experiment_dir(checkpoint_cfg, allow_overlap=True)
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"Could not find STAR checkpoint directory: {checkpoint_dir}. "
            "Set `checkpoint_path` explicitly or match the training config fields."
        )
    return utils.get_latest_checkpoint(checkpoint_dir)


@hydra.main(config_path="config", config_name="evaluate", version_base=None)
def main(cfg):
    torch.manual_seed(cfg.seed)
    OmegaConf.resolve(cfg)

    save_dir, _ = utils.get_experiment_dir(cfg, evaluate=True, allow_overlap=True)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = resolve_checkpoint_path(cfg)
    print(f"loading from checkpoint {checkpoint_path}")
    state_dict = utils.load_state(checkpoint_path)

    saved_config = state_dict.get("config")
    if saved_config is not None:
        print("autoloading policy from saved checkpoint config")
        saved_config["algo"]["policy"]["action_horizon"] = cfg.action_horizon
        model = instantiate(saved_config["algo"]["policy"], shape_meta=cfg.task.shape_meta)
    else:
        model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)

    model.to(cfg.device)
    model.eval()
    model.load_state_dict(state_dict["model"])

    env_runner = instantiate(cfg.task.env_runner)

    print(f"Saving to: {save_dir}")
    print("Running evaluation...")

    def save_video_fn(video_chw, env_name, idx):
        video_dir = os.path.join(save_dir, "videos", env_name)
        os.makedirs(video_dir, exist_ok=True)
        save_path = os.path.join(video_dir, f"{idx}.mp4")
        clip = ImageSequenceClip(list(video_chw.transpose(0, 2, 3, 1)), fps=24)
        clip.write_videofile(save_path, fps=24, verbose=False, logger=None)

    profiler = None
    if cfg.training.do_profile:
        profiler = Profiler()
        profiler.start()

    rollout_results = env_runner.run(
        model,
        n_video=cfg.rollout.n_video,
        do_tqdm=cfg.training.use_tqdm,
        save_video_fn=save_video_fn,
    )

    if profiler is not None:
        profiler.stop()
        profiler.print()

    print(
        f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} "
        f"| environments solved: {rollout_results['rollout']['environments_solved']}"
    )

    with open(os.path.join(save_dir, "data.json"), "w", encoding="utf-8") as file:
        json.dump(rollout_results, file, indent=2)


if __name__ == "__main__":
    main()
