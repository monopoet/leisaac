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


def cube_at_target(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_pos_local: tuple[float, float, float] = (0.15, -0.15, 0.0),
    distance_threshold: float = 0.05,
) -> torch.Tensor:
    """Check if the cube is at the target position (tray) relative to robot base.

    Args:
        env: The RL environment instance.
        object_cfg: Configuration for the cube entity.
        robot_cfg: Configuration for the robot entity.
        target_pos_local: Target position (x, y, z) relative to robot base position.
        distance_threshold: Maximum distance to consider cube as placed.

    Returns:
        Boolean tensor indicating which environments have cube at target.
    """
    cube: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Get robot base position (assuming base is at index 0)
    robot_base_pos = robot.data.root_pos_w

    # Calculate target position in world frame
    target_pos_world = robot_base_pos.clone()
    target_pos_world[:, 0] += target_pos_local[0]
    target_pos_world[:, 1] += target_pos_local[1]
    target_pos_world[:, 2] += target_pos_local[2]

    # Get cube position
    cube_pos = cube.data.root_pos_w

    # Calculate horizontal distance (x, y only)
    distance_xy = torch.linalg.vector_norm(cube_pos[:, :2] - target_pos_world[:, :2], dim=1)

    at_target = distance_xy < distance_threshold

    return at_target
