"""Dataset file helpers adapted from robomimic."""

from collections import OrderedDict
import os

import h5py

import star.utils.obs_utils as ObsUtils


def get_shape_metadata_from_dataset(dataset_path, all_obs_keys=None, verbose=False):
    """Read observation and action shapes from a demonstration HDF5 file."""

    dataset_path = os.path.expanduser(dataset_path)
    with h5py.File(dataset_path, "r") as dataset:
        demo_id = next(iter(dataset["data"].keys()))
        demo = dataset[f"data/{demo_id}"]

        if all_obs_keys is None:
            all_obs_keys = list(demo["obs"].keys())

        all_shapes = OrderedDict()
        for obs_key in sorted(all_obs_keys):
            input_shape = demo[f"obs/{obs_key}"].shape[1:]
            if verbose:
                print(f"obs key {obs_key} with shape {input_shape}")
            all_shapes[obs_key] = ObsUtils.get_processed_shape(
                obs_modality=ObsUtils.OBS_KEYS_TO_MODALITIES[obs_key],
                input_shape=input_shape,
            )

        return {
            "ac_dim": dataset[f"data/{demo_id}/actions"].shape[1],
            "all_shapes": all_shapes,
            "all_obs_keys": all_obs_keys,
            "use_images": ObsUtils.has_modality("rgb", all_obs_keys),
            "use_depths": ObsUtils.has_modality("depth", all_obs_keys),
        }
