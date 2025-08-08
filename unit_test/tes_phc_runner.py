# OS PATH
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

# Issac Lab : Parser
import argparse

# Issac Lab : Library
from isaaclab.app import AppLauncher

# Simulation Setting
parser = argparse.ArgumentParser(description="Unit test of Humanoid AMP Environments")

# Gym
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Seed of Randomness.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# Wandb
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--wandb-project-name", type=str, default=None, help="the wandb's project name")
parser.add_argument("--wandb-entity", type=str, default=None, help="the entity (team) of wandb's project")
parser.add_argument("--wandb-name", type=str, default=None, help="the name of wandb's run")
parser.add_argument(
    "--track",
    type=lambda x: bool(strtobool(x)),
    default=False,
    nargs="?",
    const=True,
    help="if toggled, this experiment will be tracked with Weights and Biases",
)

# GPU
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

# Output
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")


# Simulation Open
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

####################################################################
# Robot Config
from robot.h1_cfg import H1_CFG

# Env
from env.chan_task.humanoid_amp_env import HumanoidAmpEnv
from env.chan_task.humanoid_amp_env_cfg import HumanoidAmpEnvCfg

# Torch
import torch

# Python Standard
import math
import math
import os
import random
from datetime import datetime

# Gym
import gymnasium as gym

# Hydra
import hydra

# DictConfig
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf

# RL Games
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper



@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="test_agent",
)
def main(cfg_hydra: DictConfig):
    # To Do 
    # 1. wandb
    # 2. vector - env
    # 3. agent_cfg -> runner

    agent_cfg = EasyDict(OmegaConf.to_container(cfg_hydra, resolve=True))
    agent_cfg["params"]["config"]["num_actors"] = args_cli.num_envs
    agent_cfg["params"]["seed"] = args_cli.seed
    
    env_name = agent_cfg["params"]["config"]["name"]

    # read configurations about the agent-training
    rl_device = agent_cfg["params"]["config"]["device"]
    # clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    # clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    clip_obs = 5.0
    clip_actions = 1.0


    # Env Cfg : Custom
    env_cfg = HumanoidAmpEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed
    
    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None


    gym.register(id="Custom-Humanoid-v0", entry_point="env.chan_task.humanoid_amp_env:HumanoidAmpEnv")
    env = gym.make(env_name, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})


    # create runner from rl-games
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": train_sigma})
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

