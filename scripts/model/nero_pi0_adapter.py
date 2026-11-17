"""NERO rad/m <-> pi0 data transforms. NumPy only; no hardware or model loading."""
from dataclasses import dataclass

import numpy as np


CAMERAS = {
    "base": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}
ACTION_DIM = 16
JOINT_MASK = np.array([True] * 7 + [False] + [True] * 7 + [False])


def vector16(value, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (ACTION_DIM,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite and have shape (16,), got {value.shape}")
    return value.copy()


def rgb_image(value):
    value = np.asarray(value)
    if value.ndim != 3:
        raise ValueError(f"RGB image must have 3 axes, got {value.shape}")
    if value.shape[0] == 3 and value.shape[-1] != 3:
        value = value.transpose(1, 2, 0)
    if value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError(f"Invalid RGB image: {value.shape}")
    if np.issubdtype(value.dtype, np.floating):
        if value.min() < 0 or value.max() > 1:
            raise ValueError("Floating RGB images must use the LeRobot [0, 1] range")
        value = np.rint(value * 255).astype(np.uint8)
    elif value.dtype != np.uint8:
        raise ValueError(f"Expected uint8 or floating RGB, got {value.dtype}")
    return np.ascontiguousarray(value).copy()


@dataclass(frozen=True)
class NeroInputs:
    """Accept packed state/images/actions/prompt; keep NERO signs and rad/m units."""

    def __call__(self, data):
        images = data["images"]
        if set(images) != set(CAMERAS):
            raise ValueError(f"NERO requires exactly these cameras: {tuple(CAMERAS)}")
        result = {
            "state": vector16(data["state"], "state"),
            "image": {dest: rgb_image(images[src]) for src, dest in CAMERAS.items()},
            "image_mask": {dest: np.bool_(True) for dest in CAMERAS.values()},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] == 0:
                raise ValueError(f"actions must have shape (H, 16), got {actions.shape}")
            if not np.isfinite(actions).all():
                raise ValueError("actions contain NaN/Inf")
            result["actions"] = actions.copy()
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclass(frozen=True)
class NeroOutputs:
    """Keep all 16 NERO channels after OpenPI's unnormalization/inverse delta."""

    def __call__(self, data):
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < ACTION_DIM or actions.shape[0] == 0:
            raise ValueError(f"Expected (H, >=16) output actions, got {actions.shape}")
        result = actions[:, :ACTION_DIM].copy()
        if not np.isfinite(result).all():
            raise ValueError("Output actions contain NaN/Inf")
        return {"actions": result}


