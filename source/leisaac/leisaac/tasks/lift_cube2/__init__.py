import gymnasium as gym

gym.register(
    id="LeIsaac-SO101-LiftCube2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube2_env_cfg:LiftCube2EnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube2-DigitalTwin-v0",
    entry_point="leisaac.enhance.envs:ManagerBasedRLDigitalTwinEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube2_env_cfg:LiftCube2DigitalTwinEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube2-Mimic-v0",
    entry_point=f"leisaac.enhance.envs:ManagerBasedRLLeIsaacMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.lift_cube2_mimic_env_cfg:LiftCube2MimicEnvCfg",
    },
)

gym.register(
    id="LeIsaac-SO101-LiftCube2-Direct-v0",
    entry_point=f"{__name__}.direct.lift_cube2_env:LiftCube2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.direct.lift_cube2_env:LiftCube2EnvCfg",
    },
)
