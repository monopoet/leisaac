import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-PlaceCubeInTray-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_cube_env_cfg:PlaceCubeEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-PlaceCubeInTray-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_cube_mimic_env_cfg:PlaceCubeMimicEnvCfg",
    },
)
