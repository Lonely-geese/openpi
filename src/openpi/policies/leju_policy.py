import dataclasses

import cv2
import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_leju_example() -> dict:
    """Creates a random input example for the Leju policy (TASK1-ToySorting / kuavo4pro)."""
    return {
        "observation/state": np.random.rand(16).astype(np.float32),
        "observation/image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "observation/right_wrist_image": np.random.randint(256, size=(3, 480, 640), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C), resize to 224x224. Handles float (C,H,W) from LeRobot."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


@dataclasses.dataclass(frozen=True)
class LejuInputs(transforms.DataTransformFn):
    """Inputs for Leju policy (kuavo4pro / TASK1-ToySorting).

    Expects RepackTransform output: observation/image (head_cam_h), observation/wrist_image
    (wrist_cam_l), observation/right_wrist_image (wrist_cam_r), observation/state [16], actions [16].
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        right_wrist = _parse_image(data["observation/right_wrist_image"])

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)[:, :16]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LejuOutputs(transforms.DataTransformFn):
    """Outputs for Leju policy. Returns actions [:, :16] as-is (kuavo4pro 16d)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :16], dtype=np.float32)}
