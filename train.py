import os
import time

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from pyinstrument import Profiler
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from star.utils.logger import Logger
import star.utils.utils as utils


OmegaConf.register_new_resolver("eval", eval, replace=True)


def resolve_checkpoint_path(cfg, experiment_dir):
    train_cfg = cfg.training

    if train_cfg.auto_continue:
        if cfg.stage is None or cfg.stage <= 0:
            raise ValueError("training.auto_continue only supports stage-1 CST training.")
        return utils.get_stage_dir(cfg, stage=cfg.stage - 1)

    if train_cfg.resume:
        return experiment_dir

    return cfg.checkpoint_path


@hydra.main(config_path="config", version_base=None)
def main(cfg):
    device = cfg.device
    device_type = utils.get_torch_device_type(device)
    use_amp = bool(cfg.training.use_amp and device_type == "cuda")
    max_train_batches = cfg.training.max_train_batches

    torch.manual_seed(cfg.seed)

    model = instantiate(cfg.algo.policy, shape_meta=cfg.task.shape_meta)
    model.to(device)
    model.train()

    optimizers = model.get_optimizers()
    schedulers = model.get_schedulers(optimizers)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    experiment_dir, experiment_name = utils.get_experiment_dir(cfg)
    os.makedirs(experiment_dir, exist_ok=True)

    start_epoch = 0
    steps = 0
    wandb_id = None

    checkpoint_path = resolve_checkpoint_path(cfg, experiment_dir)
    if checkpoint_path is not None:
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        print(f"loading from checkpoint {checkpoint_path}")
        state_dict = utils.load_state(checkpoint_path)
        utils.soft_load_state_dict(model, state_dict["model"])

        if cfg.stage == state_dict["stage"]:
            for optimizer, optimizer_state in zip(optimizers, state_dict["optimizers"]):
                optimizer.load_state_dict(optimizer_state)
            for scheduler, scheduler_state in zip(schedulers, state_dict["schedulers"]):
                scheduler.load_state_dict(scheduler_state)
            scaler.load_state_dict(state_dict["scaler"])
            start_epoch = state_dict["epoch"] + 1
            steps = state_dict["steps"]
            wandb_id = state_dict["wandb_id"]
    else:
        print("starting from scratch")

    dataset = instantiate(cfg.task.dataset)
    model.preprocess_dataset(dataset, use_tqdm=cfg.training.use_tqdm)
    train_dataloader = instantiate(cfg.train_dataloader, dataset=dataset)

    env_runner = instantiate(cfg.task.env_runner) if cfg.rollout.enabled else None

    print(f"Saving to: {experiment_dir}")
    print(f"Experiment name: {experiment_name}")

    wandb.init(
        dir=experiment_dir,
        name=experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=wandb_id,
        **cfg.logging,
    )

    logger = Logger(cfg.training.log_interval)
    print("Training...")

    for epoch in range(start_epoch, cfg.training.n_epochs + 1):
        epoch_start_time = time.time()
        training_loss = 0.0
        profiler = None

        model.train()
        if cfg.training.do_profile:
            profiler = Profiler()
            profiler.start()

        batch_index = 0
        for batch_index, data in enumerate(tqdm(train_dataloader, disable=not cfg.training.use_tqdm), start=1):
            data = utils.map_tensor_to_device(data, device)

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device_type,
                dtype=torch.float16 if device_type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                loss, info = model.compute_loss(data)

            scaler.scale(loss).backward()

            grad_norm = None
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            if cfg.training.grad_clip is not None:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()

            info["epoch"] = epoch
            if grad_norm is not None:
                info["grad_norm"] = grad_norm.item()

            training_loss += loss.item()
            steps += 1
            logger.update({cfg.logging_folder: info}, steps)

            if max_train_batches and batch_index >= max_train_batches:
                break

        if profiler is not None:
            profiler.stop()
            profiler.print()

        training_loss /= max(1, batch_index)
        epoch_minutes = (time.time() - epoch_start_time) / 60
        print(f"[info] Epoch: {epoch:3d} | train loss: {training_loss:5.5f} | time: {epoch_minutes:4.2f}")

        if epoch > 0 and epoch % cfg.training.save_interval == 0:
            if cfg.training.save_all_checkpoints:
                checkpoint_name = os.path.join(experiment_dir, f"multitask_model_epoch_{epoch:04d}.pth")
            else:
                checkpoint_name = os.path.join(experiment_dir, "multitask_model.pth")

            utils.save_state(
                {
                    "model": model,
                    "optimizers": optimizers,
                    "schedulers": schedulers,
                    "scaler": scaler,
                    "epoch": epoch,
                    "stage": cfg.stage,
                    "steps": steps,
                    "wandb_id": wandb.run.id,
                    "experiment_dir": experiment_dir,
                    "experiment_name": experiment_name,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                checkpoint_name,
            )

        if env_runner is not None and epoch > 0 and epoch % cfg.rollout.interval == 0:
            rollout_results = env_runner.run(model, n_video=cfg.rollout.n_video, do_tqdm=cfg.training.use_tqdm)
            print(
                f"[info]     success rate: {rollout_results['rollout']['overall_success_rate']:1.3f} "
                f"| environments solved: {rollout_results['rollout']['environments_solved']}"
            )
            logger.log(rollout_results, step=steps)

        for scheduler in schedulers:
            scheduler.step(epoch)

    print("[info] finished learning")
    wandb.finish()


if __name__ == "__main__":
    main()
