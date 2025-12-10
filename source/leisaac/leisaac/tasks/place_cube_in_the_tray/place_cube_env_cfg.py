"""Configuration for the place cube in tray environment."""

import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.domain_randomization import (
    domain_randomization,
    randomize_camera_uniform,
    randomize_object_uniform,
)
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from . import mdp


# Target position relative to robot base (tray location)
# Robot is at (0.35, -0.64, 0.01)
# Cube starts on the left (positive y), tray is on the right (negative y from cube)
TARGET_POS_LOCAL = (0.15, -0.15, 0.0)  # x forward, y right of robot base


@configclass
class PlaceCubeSceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the place cube in tray task.

    Uses default camera configuration from SingleArmTaskSceneCfg
    (wrist + front cameras).
    """

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


@configclass
class ObservationsCfg(SingleArmObservationsCfg):
    """Observation configuration for place cube task."""

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        # Subtask 1: Pick up the cube
        pick_cube = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        # Subtask 2: Place cube at target (tray)
        place_cube = ObsTerm(
            func=mdp.cube_at_target,
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "robot_cfg": SceneEntityCfg("robot"),
                "target_pos_local": TARGET_POS_LOCAL,
                "distance_threshold": 0.05,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    """Termination configuration for place cube task."""

    success = DoneTerm(
        func=mdp.cube_placed_at_target,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "target_pos_local": TARGET_POS_LOCAL,
            "distance_threshold": 0.05,
            "release_threshold": 0.35,
        },
    )


@configclass
class PlaceCubeEnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the place cube in tray environment."""

    scene: PlaceCubeSceneCfg = PlaceCubeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # Viewer settings
        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        # Robot position
        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        # Parse USD and create subassets (cube)
        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        # Domain randomization for cube position
        # Cube starts on the LEFT side of robot (positive y offset)
        domain_randomization(
            self,
            random_options=[
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        # Cube positioned to the left of robot (positive y from robot)
                        "x": (-0.05, 0.05),
                        "y": (0.1, 0.2),  # Left side of robot in y direction
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
                randomize_camera_uniform(
                    "front",
                    pose_range={
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    },
                    convention="ros",
                ),
                randomize_camera_uniform(
                    "wrist",
                    pose_range={
                        "x": (-0.005, 0.005),
                        "y": (-0.005, 0.005),
                        "z": (-0.005, 0.005),
                        "roll": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "pitch": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                        "yaw": (-0.05 * torch.pi / 180, 0.05 * torch.pi / 180),
                    },
                    convention="ros",
                ),
            ],
        )
