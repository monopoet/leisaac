import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer


def object_grasped(
    env: ManagerBasedRLEnv | DirectRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    diff_threshold: float = 0.02,
    grasp_threshold: float = 0.26,
) -> torch.Tensor:
    """Check if an object is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    object_pos = object.data.root_pos_w
    end_effector_pos = ee_frame.data.target_pos_w[:, 1, :]
    pos_diff = torch.linalg.vector_norm(object_pos - end_effector_pos, dim=1)

    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, -1] < grasp_threshold)

    return grasped


def cube_above_target(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target_paper"),
    xy_threshold: float = 0.03,
    height_threshold: float = 0.02,
) -> torch.Tensor:
    """Check if the cube is above the target paper.

    Args:
        env: The RL environment instance.
        cube_cfg: Configuration for the cube entity.
        target_cfg: Configuration for the target paper entity.
        xy_threshold: Maximum horizontal distance from target center.
        height_threshold: Minimum height above target.

    Returns:
        Boolean tensor indicating which environments have cube above target.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    cube_pos = cube.data.root_pos_w
    target_pos = target.data.root_pos_w

    # Check XY distance (horizontal)
    xy_diff = torch.linalg.vector_norm(cube_pos[:, :2] - target_pos[:, :2], dim=1)

    # Check if cube is above target (z-axis)
    height_diff = cube_pos[:, 2] - target_pos[:, 2]

    above_target = torch.logical_and(xy_diff < xy_threshold, height_diff > height_threshold)

    return above_target
