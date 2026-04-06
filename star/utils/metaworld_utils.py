from collections import OrderedDict
import pickle

import numpy as np
import star.utils.file_utils as FileUtils
import star.utils.obs_utils as ObsUtils
from star.utils.dataset import SequenceDataset
from torch.utils.data import Dataset
from star.utils.frame_stack import FrameStackObservationFixed
import gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
import mujoco
from torch.utils.data import ConcatDataset
import metaworld

try:
    from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_GOAL_OBSERVABLE_ENVIRONMENTS
    _RUNTIME_ENV_VERSION = "-v2"
except ImportError:
    from metaworld import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE as ALL_GOAL_OBSERVABLE_ENVIRONMENTS
    _RUNTIME_ENV_VERSION = "-v3"

from metaworld.policies import *


def _resolve_policy_class(*candidate_names):
    for candidate_name in candidate_names:
        policy_cls = globals().get(candidate_name)
        if policy_cls is not None:
            return policy_cls
    raise ImportError(f"Unable to resolve any MetaWorld policy class from: {candidate_names}")


def _convert_env_name_version(env_name, version_suffix):
    goal_suffix = "-goal-observable"
    for current_suffix in ("-v2", "-v3"):
        full_goal_suffix = f"{current_suffix}{goal_suffix}"
        if env_name.endswith(full_goal_suffix):
            return env_name[: -len(full_goal_suffix)] + f"{version_suffix}{goal_suffix}"
    for current_suffix in ("-v2", "-v3"):
        if env_name.endswith(current_suffix):
            return env_name[: -len(current_suffix)] + version_suffix
    return env_name


def to_runtime_env_name(env_name):
    return _convert_env_name_version(env_name, _RUNTIME_ENV_VERSION)


def to_repo_env_name(env_name):
    return _convert_env_name_version(env_name, "-v2")


_POLICY_CLASS_NAMES = OrderedDict(
    [
        ("assembly-v2", ("SawyerAssemblyV2Policy", "SawyerAssemblyV3Policy")),
        ("basketball-v2", ("SawyerBasketballV2Policy", "SawyerBasketballV3Policy")),
        ("bin-picking-v2", ("SawyerBinPickingV2Policy", "SawyerBinPickingV3Policy")),
        ("box-close-v2", ("SawyerBoxCloseV2Policy", "SawyerBoxCloseV3Policy")),
        ("button-press-topdown-v2", ("SawyerButtonPressTopdownV2Policy", "SawyerButtonPressTopdownV3Policy")),
        ("button-press-topdown-wall-v2", ("SawyerButtonPressTopdownWallV2Policy", "SawyerButtonPressTopdownWallV3Policy")),
        ("button-press-v2", ("SawyerButtonPressV2Policy", "SawyerButtonPressV3Policy")),
        ("button-press-wall-v2", ("SawyerButtonPressWallV2Policy", "SawyerButtonPressWallV3Policy")),
        ("coffee-button-v2", ("SawyerCoffeeButtonV2Policy", "SawyerCoffeeButtonV3Policy")),
        ("coffee-pull-v2", ("SawyerCoffeePullV2Policy", "SawyerCoffeePullV3Policy")),
        ("coffee-push-v2", ("SawyerCoffeePushV2Policy", "SawyerCoffeePushV3Policy")),
        ("dial-turn-v2", ("SawyerDialTurnV2Policy", "SawyerDialTurnV3Policy")),
        ("disassemble-v2", ("SawyerDisassembleV2Policy", "SawyerDisassembleV3Policy")),
        ("door-close-v2", ("SawyerDoorCloseV2Policy", "SawyerDoorCloseV3Policy")),
        ("door-lock-v2", ("SawyerDoorLockV2Policy", "SawyerDoorLockV3Policy")),
        ("door-open-v2", ("SawyerDoorOpenV2Policy", "SawyerDoorOpenV3Policy")),
        ("door-unlock-v2", ("SawyerDoorUnlockV2Policy", "SawyerDoorUnlockV3Policy")),
        ("drawer-close-v2", ("SawyerDrawerCloseV2Policy", "SawyerDrawerCloseV3Policy")),
        ("drawer-open-v2", ("SawyerDrawerOpenV2Policy", "SawyerDrawerOpenV3Policy")),
        ("faucet-close-v2", ("SawyerFaucetCloseV2Policy", "SawyerFaucetCloseV3Policy")),
        ("faucet-open-v2", ("SawyerFaucetOpenV2Policy", "SawyerFaucetOpenV3Policy")),
        ("hammer-v2", ("SawyerHammerV2Policy", "SawyerHammerV3Policy")),
        ("hand-insert-v2", ("SawyerHandInsertV2Policy", "SawyerHandInsertV3Policy")),
        ("handle-press-side-v2", ("SawyerHandlePressSideV2Policy", "SawyerHandlePressSideV3Policy")),
        ("handle-press-v2", ("SawyerHandlePressV2Policy", "SawyerHandlePressV3Policy")),
        ("handle-pull-v2", ("SawyerHandlePullV2Policy", "SawyerHandlePullV3Policy")),
        ("handle-pull-side-v2", ("SawyerHandlePullSideV2Policy", "SawyerHandlePullSideV3Policy")),
        ("peg-insert-side-v2", ("SawyerPegInsertionSideV2Policy", "SawyerPegInsertionSideV3Policy")),
        ("lever-pull-v2", ("SawyerLeverPullV2Policy", "SawyerLeverPullV3Policy")),
        ("peg-unplug-side-v2", ("SawyerPegUnplugSideV2Policy", "SawyerPegUnplugSideV3Policy")),
        ("pick-out-of-hole-v2", ("SawyerPickOutOfHoleV2Policy", "SawyerPickOutOfHoleV3Policy")),
        ("pick-place-v2", ("SawyerPickPlaceV2Policy", "SawyerPickPlaceV3Policy")),
        ("pick-place-wall-v2", ("SawyerPickPlaceWallV2Policy", "SawyerPickPlaceWallV3Policy")),
        ("plate-slide-back-side-v2", ("SawyerPlateSlideBackSideV2Policy", "SawyerPlateSlideBackSideV3Policy")),
        ("plate-slide-back-v2", ("SawyerPlateSlideBackV2Policy", "SawyerPlateSlideBackV3Policy")),
        ("plate-slide-side-v2", ("SawyerPlateSlideSideV2Policy", "SawyerPlateSlideSideV3Policy")),
        ("plate-slide-v2", ("SawyerPlateSlideV2Policy", "SawyerPlateSlideV3Policy")),
        ("reach-v2", ("SawyerReachV2Policy", "SawyerReachV3Policy")),
        ("reach-wall-v2", ("SawyerReachWallV2Policy", "SawyerReachWallV3Policy")),
        ("push-back-v2", ("SawyerPushBackV2Policy", "SawyerPushBackV3Policy")),
        ("push-v2", ("SawyerPushV2Policy", "SawyerPushV3Policy")),
        ("push-wall-v2", ("SawyerPushWallV2Policy", "SawyerPushWallV3Policy")),
        ("shelf-place-v2", ("SawyerShelfPlaceV2Policy", "SawyerShelfPlaceV3Policy")),
        ("soccer-v2", ("SawyerSoccerV2Policy", "SawyerSoccerV3Policy")),
        ("stick-pull-v2", ("SawyerStickPullV2Policy", "SawyerStickPullV3Policy")),
        ("stick-push-v2", ("SawyerStickPushV2Policy", "SawyerStickPushV3Policy")),
        ("sweep-into-v2", ("SawyerSweepIntoV2Policy", "SawyerSweepIntoV3Policy")),
        ("sweep-v2", ("SawyerSweepV2Policy", "SawyerSweepV3Policy")),
        ("window-close-v2", ("SawyerWindowCloseV2Policy", "SawyerWindowCloseV3Policy")),
        ("window-open-v2", ("SawyerWindowOpenV2Policy", "SawyerWindowOpenV3Policy")),
    ]
)
_policies = OrderedDict((env_name, _resolve_policy_class(*policy_names)) for env_name, policy_names in _POLICY_CLASS_NAMES.items())
_env_names = list(_policies)

DEFAULT_CORNER2_CAMERA_POS = [0.75, 0.075, 0.7]

def get_index(env_name):
    return _env_names.index(env_name)

def get_expert():
    env_experts = {env_name: _policies[env_name]() for env_name in _policies}

    def expert(obs, task_id):
        obs_gt = obs['obs_gt'].squeeze()
        return env_experts[_env_names[task_id]].get_action(obs_gt)

    return expert

def get_benchmark(benchmark_name):
    if benchmark_name.upper() != "MT50":
        raise ValueError(f"Unsupported MetaWorld benchmark: {benchmark_name}")
    return metaworld.MT50()

def get_env_names(benchmark=None, mode=None, env_names=None):
    if benchmark is None:
        benchmark_env_names = list(_env_names)
    elif isinstance(benchmark, str) and benchmark.upper() == "MT50":
        benchmark_env_names = list(_env_names) if mode == "train" else []
    else:
        benchmark_env_names = list(benchmark.train_classes if mode == "train" else benchmark.test_classes)
        benchmark_env_names.sort()
        benchmark_env_names = [to_repo_env_name(env_name) for env_name in benchmark_env_names]

    if env_names is None:
        return benchmark_env_names

    if isinstance(env_names, str):
        env_names = [env_names]
    filtered_env_names = list(env_names)
    invalid_env_names = [
        env_name for env_name in filtered_env_names
        if env_name not in benchmark_env_names
    ]
    if invalid_env_names:
        raise ValueError(
            f"Unknown MetaWorld env_names for benchmark={benchmark} mode={mode}: {invalid_env_names}"
        )
    return filtered_env_names
    
def get_tasks(benchmark, mode):
    if benchmark is None:
        return []
    return benchmark.train_tasks if mode == "train" else benchmark.test_tasks


class MetaWorldFrameStack(FrameStackObservationFixed):
    def __init__(self, env_name, env_factory, num_stack):
        env = env_factory(env_name)
        super().__init__(env, num_stack)

    def set_task(self, task):
        self.env.set_task(task)
    

class MetaWorldWrapper(gymnasium.Wrapper):
    def __init__(
        self,
        env_name: str,
        shape_meta,
        img_height: int = 128,
        img_width: int = 128,
        cameras=("corner2",),
        env_kwargs=None,
    ):
        if env_kwargs is None:
            env_kwargs = {}
        runtime_env_name = to_runtime_env_name(env_name)
        env = ALL_GOAL_OBSERVABLE_ENVIRONMENTS[f"{runtime_env_name}-goal-observable"](**env_kwargs)
        env._freeze_rand_vec = False
        super().__init__(env)
        self.env.model.cam_pos[2] = DEFAULT_CORNER2_CAMERA_POS

        self.img_width = img_width
        self.img_height = img_height
        obs_meta = shape_meta["observation"]
        self.rgb_outputs = list(obs_meta["rgb"])
        self.lowdim_outputs = list(obs_meta["lowdim"])

        self.cameras = cameras
        self.viewer = OffScreenViewer(
            env.model,
            env.data,
            img_width,
            img_height,
            env.mujoco_renderer.max_geom,
            env.mujoco_renderer._vopt,
        )

        obs_space_dict = {}
        for key in self.rgb_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            )
        for key in self.lowdim_outputs:
            obs_space_dict[key] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_meta['lowdim'][key],),
                dtype=np.float32
            )
        obs_space_dict["obs_gt"] = env.observation_space
        self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def step(self, action):
        obs_gt, reward, terminated, truncated, info = super().step(action)
        obs_gt = obs_gt.astype(np.float32)
        info["obs_gt"] = obs_gt

        next_obs = self.make_obs(obs_gt)

        terminated = info["success"] == 1
        return next_obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        obs_gt, info = super().reset(seed=seed, options=options)
        obs_gt = obs_gt.astype(np.float32)
        info["obs_gt"] = obs_gt
        return self.make_obs(obs_gt), info

    def make_obs(self, obs_gt):
        obs = {
            "robot_states": np.concatenate((obs_gt[:4], obs_gt[18:22])),
            "obs_gt": obs_gt,
        }

        image_dict = {}
        for camera_name in self.cameras:
            image_obs = self.render(camera_name=camera_name, mode="rgb_array")
            image_dict[camera_name] = image_obs
        for key in self.rgb_outputs:
            camera_name = f"{key[:-4]}2"
            obs[key] = image_dict[camera_name][::-1]

        return obs

    def render(self, camera_name=None, mode="rgb_array"):
        if camera_name is None:
            camera_name = self.cameras[0]
        cam_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        return self.viewer.render(render_mode=mode, camera_id=cam_id)

    def set_task(self, task):
        task_data = pickle.loads(task.data)
        if "partially_observable" in task_data:
            task_data["partially_observable"] = False
            task = task._replace(data=pickle.dumps(task_data))
        self.env.set_task(task)
        self.env._partially_observable = False

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.viewer.close()


def build_dataset(
    data_prefix,
    suite_name,
    benchmark_name,
    mode,
    seq_len,
    frame_stack,
    shape_meta,
    extra_obs_modality=None,
    obs_seq_len=1,
    lowdim_obs_seq_len=None,
    load_obs=True,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=("actions",),
    env_names=None,
):
    task_names = get_env_names(benchmark_name, mode, env_names=env_names)
    n_tasks = len(task_names)
    datasets = []

    obs_modality = {
        'rgb': list(shape_meta['observation']['rgb'].keys()),
        'low_dim': list(shape_meta['observation']['lowdim'].keys())
    }
    if extra_obs_modality is not None:
        for key, value in extra_obs_modality.items():
            obs_modality[key] = obs_modality[key] + value

    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})
    for task_name in task_names:
        task_i_dataset = get_task_dataset(
            dataset_path=os.path.join(
                data_prefix,
                suite_name,
                benchmark_name,
                mode,
                f"{task_name}.hdf5",
            ),
            obs_modality=obs_modality,
            seq_len=seq_len,
            obs_seq_len=obs_seq_len,
            lowdim_obs_seq_len=lowdim_obs_seq_len,
            load_obs=load_obs,
            frame_stack=frame_stack,
            n_demos=n_demos,
            load_next_obs=load_next_obs,
            dataset_keys=dataset_keys,
        )
        task_id = get_index(task_name)
        datasets.append(SequenceVLDataset(task_i_dataset, task_id))
    n_demos = [dataset.n_demos for dataset in datasets]
    n_sequences = [dataset.total_num_sequences for dataset in datasets]
    concat_dataset = ConcatDataset(datasets)
    print("\n===================  Benchmark Information  ===================")
    print(f" Name: MetaWorld")
    print(f" # Tasks: {n_tasks}")
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=======================================================================\n")

    return concat_dataset


def get_task_dataset(
    dataset_path,
    obs_modality,
    seq_len=1,
    obs_seq_len=1,
    lowdim_obs_seq_len=None,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    load_obs=True,
    n_demos=None,
    load_next_obs=False,
    dataset_keys=None,
):
    all_obs_keys = []
    for modality_list in obs_modality.values():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )
    if load_obs:
        obs_keys = shape_meta["all_obs_keys"]
    else:
        obs_keys = []

    if dataset_keys is None:
        dataset_keys = ["actions"]
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        dataset_keys=dataset_keys,
        load_next_obs=load_next_obs,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        obs_seq_length=obs_seq_len,
        lowdim_obs_seq_length=lowdim_obs_seq_len,
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,
        hdf5_use_swmr=False,
        hdf5_normalize_obs=False,
        filter_by_attribute=filter_key,
        n_demos=n_demos,
    )
    return dataset


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_id"] = self.task_id
        return return_dict
