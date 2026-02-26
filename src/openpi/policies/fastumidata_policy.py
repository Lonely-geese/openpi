import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    """Parse image to uint8 (H,W,C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FastUmiDataInputs(transforms.DataTransformFn):

    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:

        state = transforms.pad_to_dim(data["observation/state"], self.action_dim)

        robot_0_rgb = _parse_image(data["observation/images/robot_0"])
        
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": robot_0_rgb,
                "left_wrist_0_rgb": robot_0_rgb,
                "right_wrist_0_rgb": np.zeros_like(robot_0_rgb),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }


        action_key = "actions" if "actions" in data else "action"
        if action_key in data:
            actions = transforms.pad_to_dim(data[action_key], self.action_dim)
            inputs["actions"] = actions

        if "task" in data:
            inputs["prompt"] = data["task"]

        return inputs


@dataclasses.dataclass(frozen=True)
class FastUmiDataOutputs(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
