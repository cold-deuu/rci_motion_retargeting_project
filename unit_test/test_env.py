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
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to spawn.")
parser.add_argument("--seed", type=int, default=42, help="Seed of Randomness.")

# Simulation Open
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Robot Config
from robot.h1_cfg import H1_CFG

# Env
from env.chan_task.humanoid_amp_env import HumanoidAmpEnv
from env.chan_task.humanoid_amp_env_cfg import HumanoidAmpEnvCfg


# Torch
import torch

# Python Standard
import math



def main():
    env_cfg = HumanoidAmpEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.seed = args_cli.seed

    env = HumanoidAmpEnv(env_cfg)
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
            joint_efforts = torch.randn(env.num_envs, env.robot.data.joint_pos.shape[1], device=env.sim.device)
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