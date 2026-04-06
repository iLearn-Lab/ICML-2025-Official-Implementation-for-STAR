import os
import warnings
from pathlib import Path

from natsort import natsorted
import torch

import star.utils.tensor_utils as TensorUtils


def _get_skill_quantizer_cfg(policy_cfg):
    if hasattr(policy_cfg, "skill_quantizer"):
        return policy_cfg.skill_quantizer
    if hasattr(policy_cfg, "autoencoder"):
        return policy_cfg.autoencoder
    return None


def _build_experiment_parts(cfg, evaluate=False):
    prefix = os.path.join(cfg.output_prefix, "evaluate") if evaluate else cfg.output_prefix
    parts = [prefix, cfg.task.suite_name, cfg.task.benchmark_name, cfg.algo.name, cfg.exp_name]

    if cfg.variant_name is not None:
        parts.append(cfg.variant_name)

    if cfg.seed != 10000:
        parts.append(f"seed_{cfg.seed}")

    policy_cfg = getattr(cfg.algo, "policy", None)
    skill_quantizer_cfg = _get_skill_quantizer_cfg(policy_cfg) if policy_cfg is not None else None

    if skill_quantizer_cfg is not None:
        if hasattr(skill_quantizer_cfg, "use_rotation_augmentation"):
            parts.append(f"rotation_{skill_quantizer_cfg.use_rotation_augmentation}")
        elif hasattr(skill_quantizer_cfg, "use_rotation_trick"):
            parts.append(f"rotation_{skill_quantizer_cfg.use_rotation_trick}")

        if hasattr(skill_quantizer_cfg, "codebook_size"):
            parts.append(f"codebook_{skill_quantizer_cfg.codebook_size}")
        elif hasattr(skill_quantizer_cfg, "vqvae_n_embed"):
            parts.append(f"codebook_{skill_quantizer_cfg.vqvae_n_embed}")

        if hasattr(skill_quantizer_cfg, "quantization_depth"):
            parts.append(f"depth_{skill_quantizer_cfg.quantization_depth}")
        elif hasattr(skill_quantizer_cfg, "vqvae_groups"):
            parts.append(f"depth_{skill_quantizer_cfg.vqvae_groups}")

    if policy_cfg is not None and hasattr(policy_cfg, "use_cross_entropy_loss"):
        parts.append(f"ce_loss_{policy_cfg.use_cross_entropy_loss}")

    if hasattr(cfg.algo, "lr"):
        parts.append(f"lr_{cfg.algo.lr}")

    if hasattr(cfg.algo, "action_refinement_loss_weight"):
        parts.append(f"refine_{cfg.algo.action_refinement_loss_weight}")

    return parts


def get_stage_dir(cfg, evaluate=False, stage=None):
    parts = _build_experiment_parts(cfg, evaluate=evaluate)
    resolved_stage = getattr(cfg, "stage", None) if stage is None else stage
    if resolved_stage is not None:
        parts.append(f"stage_{resolved_stage}")
    return os.path.join(*parts)


def get_experiment_dir(cfg, evaluate=False, allow_overlap=False):
    stage_dir = get_stage_dir(cfg, evaluate=evaluate)
    prefix = os.path.join(cfg.output_prefix, "evaluate") if evaluate else cfg.output_prefix
    resume_enabled = bool(getattr(getattr(cfg, "training", {}), "resume", False))

    if cfg.make_unique_experiment_dir:
        if resume_enabled:
            if not os.path.exists(stage_dir):
                raise FileNotFoundError(f"Cannot resume because no run directory exists at: {stage_dir}")
            experiment_dir = get_latest_run_dir(stage_dir)
            if experiment_dir == stage_dir:
                raise FileNotFoundError(f"Cannot resume because no run_* directory exists at: {stage_dir}")
        else:
            experiment_id = 0
            if os.path.exists(stage_dir):
                for path in Path(stage_dir).glob("run_*"):
                    if not path.is_dir():
                        continue
                    try:
                        folder_id = int(path.name.split("run_")[-1])
                    except ValueError:
                        continue
                    experiment_id = max(experiment_id, folder_id)
                experiment_id += 1
            experiment_dir = os.path.join(stage_dir, f"run_{experiment_id:03d}")
    else:
        experiment_dir = stage_dir
        if not allow_overlap and not resume_enabled and os.path.exists(experiment_dir):
            raise AssertionError(
                f"cfg.make_unique_experiment_dir=false but {experiment_dir} is already occupied"
            )

    experiment_name = os.path.relpath(experiment_dir, start=prefix).replace(os.sep, "_")
    return experiment_dir, experiment_name


def get_latest_run_dir(path):
    run_dirs = [entry for entry in os.listdir(path) if entry.startswith("run_") and os.path.isdir(os.path.join(path, entry))]
    if not run_dirs:
        return path
    return os.path.join(path, natsorted(run_dirs)[-1])


def get_latest_checkpoint(checkpoint_dir):
    if os.path.isfile(checkpoint_dir):
        return checkpoint_dir

    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_dir}")

    checkpoint_dir = get_latest_run_dir(checkpoint_dir)
    checkpoint_files = [
        file_name
        for file_name in os.listdir(checkpoint_dir)
        if os.path.isfile(os.path.join(checkpoint_dir, file_name)) and file_name.endswith(".pth")
    ]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files were found in: {checkpoint_dir}")

    return os.path.join(checkpoint_dir, natsorted(checkpoint_files)[-1])


def soft_load_state_dict(model, loaded_state_dict):
    current_model_dict = model.state_dict()
    new_state_dict = {}

    for key in current_model_dict:
        if key not in loaded_state_dict:
            warnings.warn(f"Model parameter {key} does not exist in checkpoint. Skipping")
            new_state_dict[key] = current_model_dict[key]
            continue

        value = loaded_state_dict[key]
        if hasattr(value, "size") and value.size() != current_model_dict[key].size():
            warnings.warn(
                f"Cannot load checkpoint parameter {key} with shape {value.shape} "
                f"into model with corresponding parameter shape {current_model_dict[key].shape}. Skipping"
            )
            new_state_dict[key] = current_model_dict[key]
            continue

        new_state_dict[key] = value

    for key in loaded_state_dict:
        if key not in current_model_dict:
            warnings.warn(f"Loaded checkpoint parameter {key} does not exist in model. Skipping")

    model.load_state_dict(new_state_dict)


def get_torch_device_type(device):
    device = torch.device(device)
    return device.type


def map_tensor_to_device(data, device):
    return TensorUtils.map_tensor(data, lambda x: safe_device(x, device=device))


def safe_device(x, device):
    if not isinstance(x, torch.Tensor):
        return x

    target_device = torch.device(device)
    if x.device == target_device:
        return x

    try:
        return x.to(target_device)
    except RuntimeError as exc:
        warnings.warn(f"Failed to move tensor to {target_device}: {exc}. Falling back to CPU.")
        return x.cpu()


def extract_state_dicts(inp):
    if not isinstance(inp, (dict, list)):
        if hasattr(inp, "state_dict"):
            return inp.state_dict()
        return inp

    if isinstance(inp, list):
        return [extract_state_dicts(value) for value in inp]

    return {key: extract_state_dicts(value) for key, value in inp.items()}


def save_state(state_dict, path):
    torch.save(extract_state_dicts(state_dict), path)


def load_state(path):
    return torch.load(path, map_location="cpu")
