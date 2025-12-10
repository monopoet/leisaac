from __future__ import annotations

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_placed_at_target(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    target_pos_local: tuple[float, float, float] = (0.15, -0.15, 0.0),
    distance_threshold: float = 0.05,
    release_threshold: float = 0.35,
) -> torch.Tensor:
    """Determine if the cube is placed at the target (tray) position.

    Success conditions:
    1. Cube is at target position (within distance threshold)
    2. Gripper is released (open)

    Args:
        env: The RL environment instance.
        cube_cfg: Configuration for the cube entity.
        robot_cfg: Configuration for the robot entity.
        target_pos_local: Target position (x, y, z) relative to robot base.
        distance_threshold: Maximum XY distance to consider cube at target.
        release_threshold: Gripper joint position above which gripper is considered open.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    # Get robot base position
    robot_base_pos = robot.data.root_pos_w

    # Calculate target position in world frame
    target_pos_world = robot_base_pos.clone()
    target_pos_world[:, 0] += target_pos_local[0]
    target_pos_world[:, 1] += target_pos_local[1]
    target_pos_world[:, 2] += target_pos_local[2]

    # Get cube position
    cube_pos = cube.data.root_pos_w

    # Check XY distance to target
    distance_xy = torch.linalg.vector_norm(cube_pos[:, :2] - target_pos_world[:, :2], dim=1)
    at_target = distance_xy < distance_threshold

    # Check if gripper is released (open)
    gripper_released = robot.data.joint_pos[:, -1] > release_threshold

    # Task is done when cube is at target AND gripper is released
    done = torch.logical_and(at_target, gripper_released)

    return done
