# Author : cold_deuu

# Issac Lab
from isaaclab.app import AppLauncher
from isaaclab.sim import SimulationCfg, SimulationContext


# Argument for Issac Lab
import argparse
from omegaconf import OmegaConf

# Hydra
import hydra

# Torch : GPU ACC
import torch

# Gym
import gymnasium as gym

# Python Standard
import math
import os
import random
from datetime import datetime

# RL Games
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner


# Default Args
# headless = False
# livestream = -1
# enable_cameras = False
# xr = False
# device = cuda:0
# cpu = False
# verbose = False
# info = False
# experience = 
# rendering_mode = None
# kit_args = 


# Chan
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

@hydra.main(version_base=None, config_path="../../cfg", config_name="config")
def main(cfg):
    issac_cfg = cfg.issaclab
    vars(args_cli).update(issac_cfg)

    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
#     # Play the simulator
    sim.reset()
#     # Now we are ready!
    print("[INFO]: Setup complete...")

#     # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # # close sim app
    simulation_app.close()
