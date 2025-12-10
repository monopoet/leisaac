"""Configuration for the place cube in tray task with mimic environment."""

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from .place_cube_env_cfg import PlaceCubeEnvCfg


@configclass
class PlaceCubeMimicEnvCfg(PlaceCubeEnvCfg, MimicEnvCfg):
    """
    Configuration for the place cube in tray task with mimic environment.

    Subtasks:
    1. pick_cube: Pick up the cube from the left side
    2. place_cube: Place the cube at the target (tray) position on the right
    """

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "place_cube_in_tray_leisaac_task_v0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.generation_relative = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        subtask_configs = []
        """
        subtask: pick_cube -> place_cube (at tray position)
        """

        # Subtask 1: Pick up the cube
        subtask_configs.append(
            SubTaskConfig(
                # Each subtask involves manipulation with respect to a single object frame.
                object_ref="cube",
                # This key corresponds to the binary indicator in "datagen_info" that signals
                # when this subtask is finished (e.g., on a 0 to 1 edge).
                subtask_term_signal="pick_cube",
                # Specifies time offsets for data generation when splitting a trajectory into
                # subtask segments. Random offsets are added to the termination boundary.
                subtask_term_offset_range=(10, 20),
                # Selection strategy for the source subtask segment during data generation
                selection_strategy="nearest_neighbor_object",
                # Optional parameters for the selection strategy function
                selection_strategy_kwargs={"nn_k": 3},
                # Amount of action noise to apply during this subtask
                action_noise=0.003,
                # Number of interpolation steps to bridge to this subtask segment
                num_interpolation_steps=5,
                # Additional fixed steps for the robot to reach the necessary pose
                num_fixed_steps=0,
                # If True, apply action noise during the interpolation phase and execution
                apply_noise_during_interpolation=False,
                description="Pick cube from left side",
                next_subtask_description="Place cube at tray",
            )
        )

        # Subtask 2: Place the cube at the target (tray) position
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="place_cube",
                subtask_term_offset_range=(10, 20),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.003,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Place cube at tray position",
                next_subtask_description="Task complete",
            )
        )

        # Final subtask: End state
        subtask_configs.append(
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0001,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
            )
        )

        self.subtask_configs["so101_follower"] = subtask_configs
