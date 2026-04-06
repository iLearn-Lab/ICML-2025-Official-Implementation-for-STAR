import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import gymnasium
from gymnasium.vector.utils import batch_space
from hydra.utils import to_absolute_path
from libero.libero.benchmark import get_benchmark
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import trange
from transformers import AutoModel, AutoTokenizer, logging

import star.utils.file_utils as FileUtils
import star.utils.obs_utils as ObsUtils
from star.utils.dataset import SequenceDataset
from star.utils.frame_stack import FrameStackObservationFixed

DEFAULT_LOCAL_CLIP_MODEL_PATH = "/home/li_hao/projects/base_model/clip-vit-base-patch32"
DEFAULT_CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


def _resolve_clip_model_source(task_embedding_model_path=None):
    candidate_paths = [
        task_embedding_model_path,
        os.environ.get("STAR_CLIP_MODEL_PATH"),
        DEFAULT_LOCAL_CLIP_MODEL_PATH,
    ]
    for candidate in candidate_paths:
        if not candidate:
            continue
        resolved_candidate = to_absolute_path(candidate)
        if os.path.isdir(resolved_candidate):
            return resolved_candidate
    return DEFAULT_CLIP_MODEL_NAME


class LiberoVectorWrapper(gymnasium.Env):
    def __init__(self, env_factory, env_num):
        from libero.libero.envs import DummyVectorEnv, SubprocVectorEnv

        env_creation = False
        retry_count = 0
        last_error = None
        while not env_creation and retry_count < 5:
            try:
                if env_num == 1:
                    env = DummyVectorEnv([env_factory])
                else:
                    env = SubprocVectorEnv([env_factory for _ in range(env_num)])
                env_creation = True
            except Exception as exc:
                last_error = exc
                time.sleep(5)
                retry_count += 1

        if not env_creation:
            raise RuntimeError("Failed to create LIBERO vector environment") from last_error

        self._env = env
        self.action_space = batch_space(self._env.action_space[0], env_num)
        self.observation_space = batch_space(self._env.observation_space[0], env_num)

    def reset(self, init_states=None, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        if init_states is not None:
            obs = self._env.set_init_state(init_states)
        return self.process_obs(obs), info

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = self._env.step(*args, **kwargs)
        return self.process_obs(obs), reward, terminated, truncated, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    def close(self):
        self._env.close()

    @staticmethod
    def process_obs(obs):
        """Convert LIBERO vector-env observations into stacked numpy arrays."""
        obs_out = {key: [] for key in obs[0]}
        for env_obs in obs:
            for key in obs_out:
                obs_out[key].append(env_obs[key])
        return {key: np.asarray(values) for key, values in obs_out.items()}


class LiberoFrameStack(FrameStackObservationFixed):
    def set_init_state(self, *args, **kwargs):
        return self.env.set_init_state(*args, **kwargs)


class LiberoWrapper(gymnasium.Env):
    def __init__(
        self,
        task_id,
        benchmark,
        shape_meta,
        obs_key_mapping,
        img_height=128,
        img_width=128,
        cameras=("agentview", "robot0_eye_in_hand"),
    ):
        from libero.libero.envs import OffScreenRenderEnv

        self.img_width = img_width
        self.img_height = img_height
        self.obs_key_mapping = obs_key_mapping

        obs_meta = shape_meta["observation"]
        self.rgb_outputs = list(obs_meta["rgb"])
        self.lowdim_outputs = list(obs_meta["lowdim"])
        self.cameras = cameras

        env = OffScreenRenderEnv(
            bddl_file_name=benchmark.get_task_bddl_file_path(task_id),
            camera_heights=img_height,
            camera_widths=img_width,
            camera_names=cameras,
        )
        self.env = env

        obs_space_dict = {}
        for key in self.rgb_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8,
            )
        for key in self.lowdim_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_meta["lowdim"][key],),
                dtype=np.float32,
            )
        self.observation_space = gymnasium.spaces.Dict(obs_space_dict)
        self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.render_out = None

    def reset(self, init_states=None, **kwargs):
        raw_obs = self.env.reset()
        if init_states is not None:
            raw_obs = self.env.set_init_state(init_states)

        dummy_action = np.zeros((7,), dtype=np.float32)
        for _ in range(5):
            raw_obs, _, _, _ = self.env.step(dummy_action)
        return self.make_obs(raw_obs), {}

    def step(self, action):
        raw_obs, reward, truncated, info = self.env.step(action)
        obs = self.make_obs(raw_obs)
        info["success"] = self.env.check_success()
        terminated = info["success"]
        return obs, reward, terminated, truncated, info

    def set_init_state(self, *args, **kwargs):
        return self.env.set_init_state(*args, **kwargs)

    def make_obs(self, raw_obs):
        obs = {}
        self.render_out = raw_obs[f"{self.cameras[0]}_image"][::-1]

        for key in self.rgb_outputs:
            obs[key] = raw_obs[self.obs_key_mapping[key]]
        for key in self.lowdim_outputs:
            obs[key] = raw_obs[self.obs_key_mapping[key]]

        return obs

    def render(self, *args, **kwargs):
        return self.render_out


def build_dataset(
    data_prefix,
    suite_name,
    benchmark_name,
    mode,
    seq_len,
    frame_stack,
    shape_meta,
    n_demos,
    extra_obs_modality=None,
    obs_seq_len=1,
    load_obs=True,
    task_embedding_format="clip",
    task_embedding_model_path=None,
):
    if mode != "all":
        raise ValueError("The cleaned STAR codebase only keeps LIBERO `mode=all`.")

    benchmark = get_benchmark(benchmark_name)()
    descriptions = []
    manip_datasets = []

    obs_modality = {
        "rgb": list(shape_meta["observation"]["rgb"].keys()),
        "low_dim": list(shape_meta["observation"]["lowdim"].keys()),
    }
    if extra_obs_modality is not None:
        for key, value in extra_obs_modality.items():
            obs_modality[key] = obs_modality[key] + value

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for task_index in trange(benchmark.n_tasks):
        task_dataset = get_dataset(
            dataset_path=os.path.join(data_prefix, benchmark.get_task_demonstration(task_index)),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            frame_stack=frame_stack,
            load_obs=load_obs,
            n_demos=n_demos,
        )
        manip_datasets.append(task_dataset)
        descriptions.append(benchmark.get_task(task_index).language)

    task_embs = get_task_embs(
        task_embedding_format,
        descriptions,
        task_embedding_model_path=task_embedding_model_path,
    )
    benchmark.set_task_embs(task_embs)

    datasets = [
        SequenceVLDataset(dataset, task_emb, task_id)
        for task_id, (dataset, task_emb) in enumerate(zip(manip_datasets, task_embs))
    ]

    n_demo_list = [dataset.n_demos for dataset in datasets]
    n_sequence_list = [dataset.total_num_sequences for dataset in datasets]
    concat_dataset = ConcatDataset(datasets)

    print("\n===================  Benchmark Information  ===================")
    print(f" Name: {benchmark.name}")
    print(f" # Tasks: {benchmark.n_tasks}")
    print(" # demonstrations: " + " ".join(f"({value})" for value in n_demo_list))
    print(" # sequences: " + " ".join(f"({value})" for value in n_sequence_list))
    print("=======================================================================\n")

    return concat_dataset


def get_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    load_obs=True,
    n_demos=None,
):
    all_obs_keys = []
    for modality_list in obs_modality.values():
        all_obs_keys += modality_list

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=all_obs_keys,
        verbose=False,
    )
    obs_keys = shape_meta["all_obs_keys"] if load_obs else []

    return SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,
        obs_seq_length=obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,
        hdf5_use_swmr=False,
        hdf5_normalize_obs=False,
        filter_by_attribute=filter_key,
        n_demos=n_demos,
    )


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset[idx]
        return_dict["task_emb"] = self.task_emb
        return_dict["task_id"] = self.task_id
        return return_dict


def get_task_embs(task_embedding_format, descriptions, task_embedding_model_path=None):
    logging.set_verbosity_error()

    if task_embedding_format != "clip":
        raise ValueError(
            "The cleaned STAR codebase only keeps CLIP text embeddings for LIBERO."
        )

    model_source = _resolve_clip_model_source(task_embedding_model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_source, clean_up_tokenization_spaces=True)
    model = AutoModel.from_pretrained(model_source)
    tokens = tokenizer(
        text=descriptions,
        add_special_tokens=True,
        max_length=25,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return model.get_text_features(**tokens).detach()
