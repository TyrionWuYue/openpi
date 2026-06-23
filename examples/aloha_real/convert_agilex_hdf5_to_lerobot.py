"""
Convert AgileX ALOHA-style HDF5 episodes to a LeRobot dataset.

Expected input layout:

    raw_dir/
      episode_1.hdf5
      episode_2.hdf5
      ...

Each episode should contain:

    /action
    /observations/qpos
    /observations/images/cam_high
    /observations/images/cam_left_wrist
    /observations/images/cam_right_wrist

Optional:

    /observations/qvel
    /observations/effort

Example:

    uv run examples/aloha_real/convert_agilex_hdf5_to_lerobot.py \
      --raw-dir /inspire/qb-ilm2/project/embodied-intelligent-robot-system/public/Cube_in_Bowl \
      --repo-id sii_team9/cube_in_bowl \
      --task "put the cube in the bowl"
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
import re
import shutil
from typing import Literal

import h5py
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro


DEFAULT_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DEFAULT_MOTORS = (
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
)
_EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$")


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = False
    tolerance_s: float = 0.0001
    image_writer_processes: int = 5
    image_writer_threads: int = 10
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def _episode_sort_key(path: Path) -> tuple[int, str]:
    match = _EPISODE_RE.fullmatch(path.name)
    if match is None:
        return (10**12, path.name)
    return (int(match.group(1)), path.name)


def _find_hdf5_files(raw_dir: Path, max_episodes: int | None) -> list[Path]:
    files = sorted(raw_dir.glob("episode_*.hdf5"), key=_episode_sort_key)
    if max_episodes is not None:
        files = files[:max_episodes]
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {raw_dir}")
    return files


def _has_dataset(ep_path: Path, key: str) -> bool:
    with h5py.File(ep_path, "r") as ep:
        return key in ep


def _decode_compressed_image(data: np.ndarray) -> np.ndarray:
    import cv2

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed image from HDF5 data")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _infer_camera_shape(ep_path: Path, camera: str) -> tuple[int, int, int]:
    with h5py.File(ep_path, "r") as ep:
        dataset = ep[f"/observations/images/{camera}"]
        if dataset.ndim == 4:
            _, height, width, channels = dataset.shape
            return channels, height, width

        image = _decode_compressed_image(dataset[0])
        height, width, channels = image.shape
        return channels, height, width


def _load_images_per_camera(ep: h5py.File, cameras: tuple[str, ...]) -> dict[str, np.ndarray]:
    images_per_camera = {}
    for camera in cameras:
        dataset = ep[f"/observations/images/{camera}"]
        if dataset.ndim == 4:
            images = dataset[:]
        else:
            images = np.asarray([_decode_compressed_image(data) for data in dataset])
        images_per_camera[camera] = images
    return images_per_camera


def _validate_episode(ep_path: Path, cameras: tuple[str, ...]) -> None:
    required = ["/action", "/observations/qpos", *(f"/observations/images/{camera}" for camera in cameras)]
    with h5py.File(ep_path, "r") as ep:
        missing = [key for key in required if key not in ep]
        if missing:
            raise KeyError(f"{ep_path} is missing required dataset(s): {missing}")

        action_len = ep["/action"].shape[0]
        qpos_len = ep["/observations/qpos"].shape[0]
        if action_len != qpos_len:
            raise ValueError(f"{ep_path} has mismatched action/qpos lengths: {action_len} vs {qpos_len}")

        for camera in cameras:
            image_len = ep[f"/observations/images/{camera}"].shape[0]
            if image_len != action_len:
                raise ValueError(f"{ep_path} camera {camera} has {image_len} frames, expected {action_len}")


def create_empty_dataset(
    repo_id: str,
    ep_path: Path,
    *,
    robot_type: str,
    fps: int,
    cameras: tuple[str, ...],
    mode: Literal["video", "image"],
    has_velocity: bool,
    has_effort: bool,
    overwrite: bool,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [DEFAULT_MOTORS],
        },
        "action": {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [DEFAULT_MOTORS],
        },
    }

    if has_velocity:
        features["observation.velocity"] = {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [DEFAULT_MOTORS],
        }

    if has_effort:
        features["observation.effort"] = {
            "dtype": "float32",
            "shape": (len(DEFAULT_MOTORS),),
            "names": [DEFAULT_MOTORS],
        }

    for camera in cameras:
        channels, height, width = _infer_camera_shape(ep_path, camera)
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }

    output_path = LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def load_episode(
    ep_path: Path,
    cameras: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    _validate_episode(ep_path, cameras)
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/action"][:])

        velocity = None
        if "/observations/qvel" in ep:
            velocity = torch.from_numpy(ep["/observations/qvel"][:])

        effort = None
        if "/observations/effort" in ep:
            effort = torch.from_numpy(ep["/observations/effort"][:])

        images_per_camera = _load_images_per_camera(ep, cameras)

    return images_per_camera, state, action, velocity, effort


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[Path],
    *,
    task: str,
    cameras: tuple[str, ...],
) -> LeRobotDataset:
    for ep_path in tqdm.tqdm(hdf5_files, desc="Converting episodes"):
        images_per_camera, state, action, velocity, effort = load_episode(ep_path, cameras)
        num_frames = state.shape[0]

        for frame_idx in range(num_frames):
            frame = {
                "observation.state": state[frame_idx],
                "action": action[frame_idx],
            }
            for camera, images in images_per_camera.items():
                frame[f"observation.images.{camera}"] = images[frame_idx]
            if velocity is not None:
                frame["observation.velocity"] = velocity[frame_idx]
            if effort is not None:
                frame["observation.effort"] = effort[frame_idx]
            dataset.add_frame(frame)

        dataset.save_episode(task=task)

    return dataset


def convert_agilex_hdf5_to_lerobot(
    raw_dir: Path,
    repo_id: str = "sii_team9/cube_in_bowl",
    task: str = "put the cube in the bowl",
    *,
    max_episodes: int | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    robot_type: str = "agilex_aloha",
    fps: int = 50,
    cameras: tuple[str, ...] = DEFAULT_CAMERAS,
    overwrite: bool = True,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> None:
    hdf5_files = _find_hdf5_files(raw_dir, max_episodes)
    for ep_path in hdf5_files:
        _validate_episode(ep_path, cameras)

    dataset = create_empty_dataset(
        repo_id,
        hdf5_files[0],
        robot_type=robot_type,
        fps=fps,
        cameras=cameras,
        mode=mode,
        has_velocity=_has_dataset(hdf5_files[0], "/observations/qvel"),
        has_effort=_has_dataset(hdf5_files[0], "/observations/effort"),
        overwrite=overwrite,
        dataset_config=dataclasses.replace(dataset_config, use_videos=mode == "video"),
    )
    dataset = populate_dataset(dataset, hdf5_files, task=task, cameras=cameras)
    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()

    print(f"Converted {len(hdf5_files)} episode(s) to {LEROBOT_HOME / repo_id}")


if __name__ == "__main__":
    tyro.cli(convert_agilex_hdf5_to_lerobot)
