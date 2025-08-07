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

def main():
    # To Do 
    # 1. wandb
    # 2. vector - env
    # 3. agent_cfg -> runner

    # Env Cfg : Custom
    env_cfg = HumanoidAmpEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    gym.register(id="Custom-Humanoid-v0", entry_point="env.chan_task.humanoid_amp_env:HumanoidAmpEnv")
    env = gym.make("Custom-Humanoid-v0", cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    obs, _ = env.reset()
    count = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            joint_efforts = torch.randn(env.unwrapped.num_envs, env.unwrapped.robot.data.joint_pos.shape[1], device=env.unwrapped.sim.device)
            # step the environment
            obs_buf, reward_buf, reset_terminated, reset_time_outs, extras = env.step(joint_efforts)
            # print current orientation of pole
            print("[Env 0]: Reward: ", reward_buf[0])
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()