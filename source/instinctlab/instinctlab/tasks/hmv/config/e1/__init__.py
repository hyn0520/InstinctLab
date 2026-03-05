import gymnasium as gym

from . import agent as agents

task_entry = "instinctlab.tasks.hmv.config.e1"

gym.register(
    id="Instinct-Parkour-Target-Amp-E1-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.e1_parkour_target_amp_cfg:E1ParkourRoughEnvCfg",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:E1ParkourPPORunnerCfg",
    },
)

gym.register(
    id="Instinct-Parkour-Target-Amp-E1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{task_entry}.e1_parkour_target_amp_cfg:E1ParkourRoughEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_amp_cfg:E1ParkourPPORunnerCfg",
    },
)
