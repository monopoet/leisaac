import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.enhance.envs.manager_based_rl_digital_twin_env_cfg import (
    ManagerBasedRLDigitalTwinEnvCfg,
)
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


@configclass
class LiftCube2SceneCfg(SingleArmTaskSceneCfg):
    """Scene configuration for the lift cube2 task.

    This task uses:
    - Top-view camera: Looking down from above
    - Wrist camera: Mounted on the robot gripper
    """

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # Top-view camera - positioned above looking down
    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/top_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.8),  # Positioned above the workspace
            rot=(0.0, 0.70711, 0.0, 0.70711),  # Looking straight down (90 degrees pitch)
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=38.11,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    # Wrist camera - mounted on the gripper (inherited from template but redefined for clarity)
    wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.001, 0.1, -0.04),
            rot=(-0.404379, -0.912179, -0.0451242, 0.0486914),
            convention="ros",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=36.5,
            focus_distance=400.0,
            horizontal_aperture=36.83,
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    # Blue target paper - spawned as a visual marker
    target_paper: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/target_paper",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.10, 0.002),  # 10cm x 10cm, 2mm thick paper
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                kinematic_enabled=True,  # Static object
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.3, 0.8),  # Blue color
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.15, 0.0, 0.005),  # Positioned next to cube, on table surface
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )

    def __post_init__(self):
        super().__post_init__()
        # Remove the default front camera - we use top and wrist instead
        if hasattr(self, "front"):
            delattr(self, "front")


@configclass
class ObservationsCfg(SingleArmObservationsCfg):
    """Observation configuration for lift cube2 task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - using top and wrist cameras."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        # Top-view camera
        top = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False},
        )

        # Wrist camera
        wrist = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False},
        )

        ee_frame_state = ObsTerm(
            func=mdp.ee_frame_state,
            params={"ee_frame_cfg": SceneEntityCfg("ee_frame"), "robot_cfg": SceneEntityCfg("robot")},
        )
        joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""

        pick_cube = ObsTerm(
            func=mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube"),
            },
        )

        place_cube = ObsTerm(
            func=mdp.cube_above_target,
            params={
                "cube_cfg": SceneEntityCfg("cube"),
                "target_cfg": SceneEntityCfg("target_paper"),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(SingleArmTerminationsCfg):
    """Termination configuration for lift cube2 task."""

    success = DoneTerm(
        func=mdp.cube_placed_on_target,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "target_cfg": SceneEntityCfg("target_paper"),
            "xy_threshold": 0.04,
            "z_threshold": 0.03,
            "velocity_threshold": 0.1,
        },
    )


@configclass
class LiftCube2EnvCfg(SingleArmTaskEnvCfg):
    """Configuration for the lift cube2 environment.

    Task: Pick up the red cube and place it on the blue target paper.
    Cameras: Top-view camera and wrist camera.
    """

    scene: LiftCube2SceneCfg = LiftCube2SceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (-0.4, -0.6, 0.7)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        self.scene.robot.init_state.pos = (0.35, -0.64, 0.01)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        domain_randomization(
            self,
            random_options=[
                # Randomize cube position
                randomize_object_uniform(
                    "cube",
                    pose_range={
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "z": (0.0, 0.0),
                        "yaw": (-30 * torch.pi / 180, 30 * torch.pi / 180),
                    },
                ),
                # Randomize target paper position (smaller range)
                randomize_object_uniform(
                    "target_paper",
                    pose_range={
                        "x": (-0.02, 0.02),
                        "y": (-0.02, 0.02),
                        "z": (0.0, 0.0),
                        "yaw": (-15 * torch.pi / 180, 15 * torch.pi / 180),
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


@configclass
class LiftCube2DigitalTwinEnvCfg(LiftCube2EnvCfg, ManagerBasedRLDigitalTwinEnvCfg):
    """Configuration for the lift cube2 digital twin environment."""

    rgb_overlay_mode: str = "background"

    rgb_overlay_paths: dict[str, str] = {
        "top": "greenscreen/background-lift-cube2-top.png",
        "wrist": "greenscreen/background-lift-cube2-wrist.png",
    }

    render_objects: list[SceneEntityCfg] = [
        SceneEntityCfg("cube"),
        SceneEntityCfg("target_paper"),
        SceneEntityCfg("robot"),
    ]
