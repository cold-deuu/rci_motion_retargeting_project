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

# PHC Learning
from learning import im_amp
from learning import im_amp_players
from learning import amp_agent
from learning import amp_players
from learning import amp_models
from learning import amp_network_builder
from learning import amp_network_mcp_builder
from learning import amp_network_pnn_builder

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


def build_alg_runner(algo_observer):
    runner = Runner(algo_observer)
    runner.player_factory.register_builder('amp_discrete', lambda **kwargs: amp_players.AMPPlayerDiscrete(**kwargs))
    
    runner.algo_factory.register_builder('amp', lambda **kwargs: amp_agent.AMPAgent(**kwargs))
    runner.player_factory.register_builder('amp', lambda **kwargs: amp_players.AMPPlayerContinuous(**kwargs))

    runner.model_builder.model_factory.register_builder('amp', lambda network, **kwargs: amp_models.ModelAMPContinuous(network))
    runner.model_builder.network_factory.register_builder('amp', lambda **kwargs: amp_network_builder.AMPBuilder())
    runner.model_builder.network_factory.register_builder('amp_mcp', lambda **kwargs: amp_network_mcp_builder.AMPMCPBuilder())
    runner.model_builder.network_factory.register_builder('amp_pnn', lambda **kwargs: amp_network_pnn_builder.AMPPNNBuilder())
    
    runner.algo_factory.register_builder('im_amp', lambda **kwargs: im_amp.IMAmpAgent(**kwargs))
    runner.player_factory.register_builder('im_amp', lambda **kwargs: im_amp_players.IMAMPPlayerContinuous(**kwargs))
    
    return runner

@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="phc_learning",
)
def main(cfg_hydra: DictConfig):
    # To Do 
    # 1. wandb

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
    runner = build_alg_runner(runner)
    runner.load(agent_cfg)
    runner.reset()
    runner.run({"train": True, "play": False, "sigma": train_sigma})
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()


# Error
# 2025-08-08 10:22:41 [25,371ms] [Warning] [omni.hydra.scene_delegate.plugin] Calling getBypassRenderSkelMeshProcessing for prim /World/envs/env_0/Robot/left_shoulder_yaw_link/visuals.proto_mesh_id0 that has not been populated
# [INFO]: Time taken for simulation start : 9.394737 seconds
# Creating window for environment.
# ManagerLiveVisualizer cannot be created for manager: action_manager, Manager does not exist
# ManagerLiveVisualizer cannot be created for manager: observation_manager, Manager does not exist
# [INFO]: Completed setting up the environment...
# self.seed = 42
# Started to train
# seq_length: 4
# current training device: cuda:0
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/rl_games/common/a2c_common.py:254: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
#   self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
# build mlp: 50
# build mlp: 50
# RunningMeanStd:  (1,)
# RunningMeanStd:  (50,)
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:130: UserWarning: WARN: The obs returned by the `reset()` method was expecting a numpy array, actual type: <class 'dict'>
#   logger.warn(
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/spaces/box.py:424: UserWarning: WARN: Casting input x to numpy array.
#   gym.logger.warn("Casting input x to numpy array.")
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:158: UserWarning: WARN: The obs returned by the `reset()` method is not within the observation space.
#   logger.warn(f"{pre} is not within the observation space.")
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:227: UserWarning: WARN: Expects `terminated` signal to be a boolean, actual type: <class 'torch.Tensor'>
#   logger.warn(
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:231: UserWarning: WARN: Expects `truncated` signal to be a boolean, actual type: <class 'torch.Tensor'>
#   logger.warn(
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:130: UserWarning: WARN: The obs returned by the `step()` method was expecting a numpy array, actual type: <class 'dict'>
#   logger.warn(
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:158: UserWarning: WARN: The obs returned by the `step()` method is not within the observation space.
#   logger.warn(f"{pre} is not within the observation space.")
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/gymnasium/utils/passive_env_checker.py:245: UserWarning: WARN: The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: <class 'torch.Tensor'>
#   logger.warn(
# /home/server1/anaconda3/envs/cold_deuu/lib/python3.10/site-packages/rl_games/algos_torch/a2c_continuous.py:106: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
#   with torch.cuda.amp.autocast(enabled=self.mixed_precision):

# GEMINI
# 이 로그는 치명적인 에러(Error)가 아니라, 경고(Warning) 메시지들입니다. 프로그램이 멈추지는 않지만, 코드의 잠재적인 문제나 비표준적인 사용 방식을 알려주고 있습니다.

# 핵심 원인은 직접 만드신 커스텀 환경(HumanoidAmpEnv)이 Gymnasium 라이브러리가 기대하는 표준 데이터 형식(Numpy 배열, bool 등)을 따르지 않고, PyTorch 텐서(torch.Tensor)나 딕셔너리(dict)를 반환하고 있기 때문입니다.

# ## 주요 경고 메시지 분석 및 해결 방안

# 이 경고들은 gymnasium의 PassiveEnvChecker라는 기능이 "이 환경, 표준에 맞게 잘 만들어졌나?"를 자동으로 검사하면서 발생합니다.

# 1. 관측(obs) 데이터 타입 및 구조 문제

#     경고 메시지:

#     WARN: The obs returned by the `reset()` method was expecting a numpy array, actual type: <class 'dict'>
#     WARN: The obs returned by the `reset()` method is not within the observation space.

#     원인:
#     환경의 _get_observations 함수가 {"policy": obs} 형태의 **딕셔너리(dict)**를 반환하고 있습니다. 하지만 gymnasium의 표준 observation_space는 보통 Box와 같은 단일 공간이며, reset()이나 step() 메소드는 이 공간에 맞는 Numpy 배열을 반환해야 합니다. rl-games와 같은 라이브러리는 딕셔너리 형태의 관측을 처리할 수 있지만, 순수한 gymnasium 표준에서는 벗어난 것입니다.

#     해결 방안 (권장):
#     _get_observations 함수가 딕셔너리가 아닌, 관측 텐서 자체를 반환하도록 수정해야 합니다.
#     Python

#     # HumanoidAmpEnv 클래스의 _get_observations 함수 수정

#     def _get_observations(self) -> torch.Tensor: # 반환 타입을 딕셔너리에서 텐서로 변경
#         # ... obs 계산 로직 ...
#         # return {"policy": obs} # 딕셔너리 대신
#         return obs # 텐서 자체를 반환

#     이후 reset()과 step() 메소드에서 이 관측 텐서를 .cpu().numpy()로 변환하여 반환하면 경고가 사라집니다.

# 2. terminated 및 truncated 신호 타입 문제

#     경고 메시지:

#     WARN: Expects `terminated` signal to be a boolean, actual type: <class 'torch.Tensor'>
#     WARN: Expects `truncated` signal to be a boolean, actual type: <class 'torch.Tensor'>

#     원인:
#     _get_dones 함수가 bool 타입의 died와 time_out이 아닌, PyTorch 텐서를 반환하고 있습니다.

#     해결 방안:
#     반환하기 전에 .cpu().numpy().astype(bool) 등을 사용하여 타입을 변환해줘야 합니다. 하지만 rl-games와 같은 VecEnv 기반의 래퍼가 이 변환을 내부적으로 처리해주는 경우가 많아, 실제 훈련에는 큰 문제가 되지 않을 수 있습니다.

# 3. 보상(reward) 타입 문제

#     경고 메시지:

#     WARN: The reward returned by `step()` must be a float, int, np.integer or np.floating, actual type: <class 'torch.Tensor'>

#     원인:
#     _get_rewards 함수가 단일 숫자(float)나 Numpy 숫자가 아닌, PyTorch 텐서를 반환하고 있습니다.

#     해결 방안:
#     terminated/truncated와 마찬가지로, 반환 전에 .cpu().numpy()로 변환해주는 것이 표준적인 방법입니다.

# 4. autocast 관련 FutureWarning

#     경고 메시지:

#     FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.

#     원인:
#     rl-games 라이브러리 내부에서 사용하고 있는 torch.cuda.amp.autocast 함수가 PyTorch의 최신 버전에서 사용이 권장되지 않는(deprecated) 이전 방식이라는 경고입니다.

#     해결 방안:
#     이것은 사용자의 코드 문제가 아니라 rl-games 라이브러리 자체의 코드에 관한 것입니다. 라이브러리 개발자들이 다음 버전에서 수정할 내용이므로, 사용자 입장에서는 무시하셔도 괜찮습니다. 훈련 성능이나 결과에는 아무런 영향을 미치지 않습니다.

# ### 결론

# 현재 로그에 보이는 경고들은 대부분 데이터 타입 불일치에서 비롯됩니다. Isaac Lab과 같은 고성능 시뮬레이터는 모든 연산을 GPU 상의 torch.Tensor로 처리하는 반면, gymnasium의 기본 표준은 CPU 기반의 numpy 배열을 기준으로 하기 때문에 이런 경고가 발생합니다.

# RlGamesVecEnvWrapper와 같은 래퍼가 중간에서 이런 타입 변환을 상당 부분 처리해주기 때문에 훈련 자체는 진행될 수 있지만, 가장 깔끔한 방법은 환경 코드(HumanoidAmpEnv)의 최종 반환 값들을 gymnasium 표준에 맞게 수정하는 것입니다.