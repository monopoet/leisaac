import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ...template import SingleArmTaskDirectEnv, SingleArmTaskDirectEnvCfg
from .. import mdp
from ..lift_cube2_env_cfg import LiftCube2SceneCfg


@configclass
class LiftCube2EnvCfg(SingleArmTaskDirectEnvCfg):
    """Direct env configuration for the lift cube2 task.

    Task: Pick up the red cube and place it on the blue target paper.
    Cameras: Top-view camera and wrist camera.
    """

    scene: LiftCube2SceneCfg = LiftCube2SceneCfg(env_spacing=8.0)

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        self.viewer.eye = (0.0, -1.5, 1.5)
        self.viewer.lookat = (0.9, 0.0, 0.88)

        # Update cameras list to use top and wrist
        self.cameras = ["top", "wrist"]

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                # Randomize cube position - offset to robot's left side (y positive)
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (0.05, 0.15),  # Offset to robot's left (y positive)
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
                # Randomize target paper position (smaller range, stays on right side)
                randomize_object_uniform(
                    "target_paper",
                    pose_range={
                        "x": (-0.03, 0.03),
                        "y": (-0.03, 0.03),
                        "z": (0.0, 0.0),
                        "yaw": (-10 * torch.pi / 180, 10 * torch.pi / 180),
                    },
                ),
                # Randomize top camera
                randomize_camera_uniform(
                    "top",
                    pose_range={
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    },
                    convention="opengl",
                ),
                # Randomize wrist camera
                randomize_camera_uniform(
                    "wrist",
                    pose_range={
                        "x": (-0.003, 0.003),
                        "y": (-0.003, 0.003),
                        "z": (-0.003, 0.003),
                        "roll": (-0.03 * torch.pi / 180, 0.03 * torch.pi / 180),
                        "pitch": (-0.03 * torch.pi / 180, 0.03 * torch.pi / 180),
                        "yaw": (-0.03 * torch.pi / 180, 0.03 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )


class LiftCube2Env(SingleArmTaskDirectEnv):
    """Direct env for the lift cube2 task.

    Task: Pick up the red cube and place it on the blue target paper.
    """

    cfg: LiftCube2EnvCfg

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        # Add subtask observations
        obs["subtask_terms"] = {
            "pick_cube": mdp.object_grasped(
                self,
                robot_cfg=SceneEntityCfg("robot"),
                ee_frame_cfg=SceneEntityCfg("ee_frame"),
                object_cfg=SceneEntityCfg("cube"),
            ),
            "place_cube": mdp.cube_above_target(
                self,
                cube_cfg=SceneEntityCfg("cube"),
                target_cfg=SceneEntityCfg("target_paper"),
            ),
        }
        return obs

    def _check_success(self) -> torch.Tensor:
        return mdp.cube_placed_on_target(
            env=self,
            cube_cfg=SceneEntityCfg("cube"),
            target_cfg=SceneEntityCfg("target_paper"),
            xy_threshold=0.04,
            z_threshold=0.03,
            velocity_threshold=0.1,
        )
