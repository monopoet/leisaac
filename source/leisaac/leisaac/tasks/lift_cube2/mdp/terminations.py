from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_placed_on_target(
    env: ManagerBasedRLEnv | DirectRLEnv,
    cube_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    xy_threshold: float = 0.03,
    z_threshold: float = 0.02,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Determine if the cube is placed on the target paper.

    This function checks whether all success conditions for the task have been met:
    1. Cube is horizontally aligned with target (within xy_threshold)
    2. Cube is on target surface (within z_threshold of target height + cube half-height)
    3. Cube is relatively stationary (velocity below threshold)

    Args:
        env: The RL environment instance.
        cube_cfg: Configuration for the cube entity.
        target_cfg: Configuration for the target paper entity.
        xy_threshold: Maximum horizontal distance from target center.
        z_threshold: Tolerance for height matching.
        velocity_threshold: Maximum velocity for cube to be considered placed.

    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    cube_pos = cube.data.root_pos_w
    target_pos = target.data.root_pos_w
    cube_vel = cube.data.root_lin_vel_w

    # Check XY distance (horizontal alignment)
    xy_diff = torch.linalg.vector_norm(cube_pos[:, :2] - target_pos[:, :2], dim=1)
    xy_aligned = xy_diff < xy_threshold

    # Check Z height (cube should be on target, accounting for cube size ~0.025m half-height)
    # Target paper is thin, so cube bottom should be near target surface
    cube_bottom_height = cube_pos[:, 2] - 0.025  # Approximate cube half-height
    height_diff = torch.abs(cube_bottom_height - target_pos[:, 2])
    z_aligned = height_diff < z_threshold

    # Check velocity (cube should be stationary)
    velocity_mag = torch.linalg.vector_norm(cube_vel, dim=1)
    is_stationary = velocity_mag < velocity_threshold

    # All conditions must be met
    done = torch.logical_and(xy_aligned, torch.logical_and(z_aligned, is_stationary))

    return done
