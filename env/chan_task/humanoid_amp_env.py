# Code for Making RL Environments : Humanoid Imitation
# Author : cold-deuu
from __future__ import annotations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../..")))


import gymnasium as gym
import numpy as np

# Torch
import torch
from torch_utils import torch_utils
from torch_utils.gym_utils import *

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg

# from .motions import MotionLoader


class HumanoidAmpEnv(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # action offset and scale
        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # load motion
        # self._motion_loader = MotionLoader(motion_file=self.cfg.motion_file, device=self.device)

        
        # DOF and key body indexes
        # Key Body Name 수정 
        key_body_names = ["pelvis","right_elbow_link", "left_elbow_link", "right_ankle_link", "left_ankle_link"]


        # Data 상에 있는 Reference Body 의 인덱스를 가져옴
        # H1 : Pelvis Link Index
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)

        # Data 상에 있는 Key Body 의 인덱스를 가져옴
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        
        # Motion Loader 는 나중에 확인하기
        # self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        # self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        # self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        
        # 이걸 잘 모르겠네. Num_amp_obs = 2, num_obs_space = 81. 어떻게 2, 81 이 나오는지? 그리고 그 둘을 왜 곱하는지?
        # self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        # self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        # self.amp_observation_buffer = torch.zeros(
        #     (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        # )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        # add ground plane
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target(target)

    def _get_observations(self) -> dict:
        # build task observation : Default
        # obs = compute_obs(
        #     self.robot.data.joint_pos,
        #     self.robot.data.joint_vel,
        #     # self.robot.data.body_pos_w[:, self.ref_body_index],
        #     # self.robot.data.body_quat_w[:, self.ref_body_index],
        #     # self.robot.data.body_lin_vel_w[:, self.ref_body_index],
        #     # self.robot.data.body_ang_vel_w[:, self.ref_body_index],
        #     self.robot.data.body_pos_w[:, self.key_body_indexes],
        # )

        # Compute Observation (2025.08.25)
        # Editor : cold-deuu
        # 여기 ... 에서 문제생김
        # Pos 
        root_pos_w = self.robot.data.body_pos_w[..., self.ref_body_index] # Root
        kp_pos_w = self.robot.data.body_pos_w[..., self.key_body_indexes] # Key-Point
        
        # Rot
        root_quat_w = self.robot.data.body_quat_w[..., self.ref_body_index] # Root   
        kp_quat_w = self.robot.data.body_quat_w[..., self.key_body_indexes] # Key-Point

        # Linear Velocity
        root_linvel_w = self.robot.body_lin_vel_w[..., self.ref_body_index] # Root
        kp_linvel_w = self.robot.body_lin_vel_w[..., self.key_body_indexes] # Key-Point

        # Angular Velocity
        root_angvel_w = self.robot.body_ang_vel_w[..., self.ref_body_index] # Root
        kp_angvel_w = self.robot.body_ang_vel_w[..., self.key_body_indexes] # Key-Point

        root_param = {"pos" : root_pos_w, "quat" : root_quat_w, "linvel" : root_linvel_w, "angvel" : root_angvel_w}
        kp_param = {"pos" : kp_pos_w, "quat" : kp_quat_w, "linvel" : kp_linvel_w, "angvel" : kp_angvel_w}
        # obs = compute_self_obs(root_param, kp_param) # (B, 3*4 + 4*5 + 3*5 + 3*5 + 1) --> B, 63
        obs = compute_self_obs(root_pos_w, root_quat_w, kp_pos_w, kp_quat_w, kp_linvel_w, kp_angvel_w) # (B, 3*4 + 4*5 + 3*5 + 3*5 + 1) --> B, 63




        # Motion
        # # update AMP observation history
        # for i in reversed(range(self.cfg.num_amp_observations - 1)):
        #     self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # # build AMP observation
        # self.amp_observation_buffer[:, 0] = obs.clone()
        # self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}


        # Logger
        # print(f"[Chan_Logger] Body Pos Data : {self.robot.data.body_pos_w[:, self.ref_body_index]}")
        # print(f"[Chan_Logger] Body Pos Shape : {(self.robot.data.body_pos_w[:, self.ref_body_index]).shape}") # B, num_ref, 3 (Expect) -> B, 1, 3 --> B, 3
        # print(f"[Chan_Logger] Body Rot Data : {self.robot.data.body_quat_w[:, self.ref_body_index]}")
        # print(f"[Chan_Logger] Body Rot Shape : {(self.robot.data.body_quat_w[:, self.ref_body_index]).shape}") # B, num_ref, 4 (Expect) -> B, 1, 4 --> B, 4
        # print(f"[Chan_Logger] Body Pos Data : {self.robot.data.body_pos_w[:, self.key_body_indexes]}")
        # print(f"[Chan_Logger] Body Pos Shape : {(self.robot.data.body_pos_w[:, self.key_body_indexes]).shape}") # B, num_ref, 3
        # print(f"[Chan_Logger] Body Rot Data : {self.robot.data.body_quat_w[:, self.key_body_indexes]}")
        # print(f"[Chan_Logger] Body Rot Shape : {(self.robot.data.body_quat_w[:, self.key_body_indexes]).shape}") # B, num_ref, 4
        print(f"[Chan_Logger] Obs Shape :{obs.shape}") # 1(B), 298

        


        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # if self.cfg.early_termination:
        #     died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height
        # else:
        #     died = torch.zeros_like(time_out)
        died = torch.zeros_like(time_out)

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # if self.cfg.reset_strategy == "default":
        #     root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        # elif self.cfg.reset_strategy.startswith("random"):
        #     start = "start" in self.cfg.reset_strategy
        #     root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        # else:
        #     raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")
        root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        self.robot.write_root_link_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_com_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        return root_state, joint_pos, joint_vel

    # def _reset_strategy_random(
    #     self, env_ids: torch.Tensor, start: bool = False
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     # sample random motion times (or zeros if start is True)
    #     num_samples = env_ids.shape[0]
    #     times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
    #     # sample random motions
    #     (
    #         dof_positions,
    #         dof_velocities,
    #         body_positions,
    #         body_rotations,
    #         body_linear_velocities,
    #         body_angular_velocities,
    #     ) = self._motion_loader.sample(num_samples=num_samples, times=times)

    #     # get root transforms (the humanoid torso)
    #     motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
    #     root_state = self.robot.data.default_root_state[env_ids].clone()
    #     root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
    #     root_state[:, 2] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
    #     root_state[:, 3:7] = body_rotations[:, motion_torso_index]
    #     root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
    #     root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
    #     # get DOFs state
    #     dof_pos = dof_positions[:, self.motion_dof_indexes]
    #     dof_vel = dof_velocities[:, self.motion_dof_indexes]

    #     # update AMP observation
    #     amp_observations = self.collect_reference_motions(num_samples, times)
    #     self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

    #     return root_state, dof_pos, dof_vel

    # env methods

    # def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
    #     # sample random motion times (or use the one specified)
    #     if current_times is None:
    #         current_times = self._motion_loader.sample_times(num_samples)
    #     times = (
    #         np.expand_dims(current_times, axis=-1)
    #         - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
    #     ).flatten()
    #     # get motions
    #     (
    #         dof_positions,
    #         dof_velocities,
    #         body_positions,
    #         body_rotations,
    #         body_linear_velocities,
    #         body_angular_velocities,
    #     ) = self._motion_loader.sample(num_samples=num_samples, times=times)
    #     # compute AMP observation
    #     amp_observation = compute_obs(
    #         dof_positions[:, self.motion_dof_indexes],
    #         dof_velocities[:, self.motion_dof_indexes],
    #         body_positions[:, self.motion_ref_body_index],
    #         body_rotations[:, self.motion_ref_body_index],
    #         body_linear_velocities[:, self.motion_ref_body_index],
    #         body_angular_velocities[:, self.motion_ref_body_index],
    #         body_positions[:, self.motion_key_body_indexes],
    #     )
    #     return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    # root_positions: torch.Tensor,
    # root_rotations: torch.Tensor,
    # root_linear_velocities: torch.Tensor,
    # root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            # root_positions[:, 2:3],  # root body height
            # quaternion_to_tangent_and_normal(root_rotations),
            # root_linear_velocities,
            # root_angular_velocities,
            # (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),

            (key_body_positions).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs

# PHC
# @torch.jit.script
def compute_self_obs_v2(root_param, kp_param):
    root_pos_w = root_param["pos"] # B,3
    root_quat_w = root_param["quat"] # B,4
    
    kp_pos_w = kp_param["pos"] # B,J,3
    kp_quat_w = kp_param["quat"] # B,J,4
    kp_linvel_w = kp_param["linvel"] # B,J,3
    kp_angvel_w = kp_param["angvel"] # B,J,3
    
    # Param
    num_env, num_kp, _ = kp_pos_w.shape

    root_h = root_pos_w[...,2]
    heading_quat_inv = torch_utils.calc_heading_quat_inv(root_quat_w)
    # print(f"Heading Quat Inv : {heading_quat_inv}") # 1,4
    heading_quat_inv_expand = heading_quat_inv.unsqueeze(1).repeat((1,kp_pos_w.shape[1],1))
    
    # (B,J,3) - (B,3)
    local_body_pos = kp_pos_w - root_pos_w

    # local_body_pos : (B,J,3)
    # heading_quat_inv_expand : (B,J,4)
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], -1)
    flat_heading_quat_inv = heading_quat_inv_expand.reshape(heading_quat_inv_expand.shape[0] * heading_quat_inv_expand.shape[1], -1)
    local_body_pos_obs = torch_utils.my_quat_rotate(flat_heading_quat_inv, flat_local_body_pos)
    
    local_body_pos_obs = local_body_pos_obs.reshape(local_body_pos.shape[0], local_body_pos.shape[1]*3) # 1, 20*3
    local_body_pos_obs = local_body_pos_obs[...,3:].clone() # Root Link 제외
    # print(f"local_body_pos_obs :{local_body_pos_obs.shape}") # 1(B), 57

    # local_body_rot = quat_mul(heading_quat_inv, self.link_pose_tensor[...,3:])
    # local_body_rot_obs = torch_utils.quat_to_tan_norm(local_body_rot).view(B, time_steps, self.link_pose_tensor.shape[0] * 6)

    body_rot = kp_quat_w
    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_quat_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)

    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    # print(f"local_body_rot_obs :{local_body_rot_obs.shape}") # 1(B), 6*20

    body_vel = kp_linvel_w
    body_ang_vel = kp_angvel_w

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_quat_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    # print(f"local_body_vel :{local_body_vel.shape}") # 1(B), 57

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_quat_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    # print(f"local_body_ang_vel :{local_body_ang_vel.shape}") # 1(B), 57


    # return tensor : batch_size, --
    obs = torch.cat([root_h, local_body_pos_obs, local_body_rot_obs, local_body_vel, local_body_ang_vel], dim=-1)

    return obs

@torch.jit.script
def compute_self_obs(root_pos : torch.Tensor, root_quat : torch.Tensor, kp_pos : torch.Tensor, kp_quat : torch.Tensor, kp_vel : torch.Tensor, kp_angvel : torch.Tensor):
    root_pos_w = root_pos # B,3
    root_quat_w = root_quat # B,4
    
    kp_pos_w = kp_pos["pos"] # B,J,3
    kp_quat_w = kp_quat["quat"] # B,J,4
    kp_linvel_w = kp_vel["linvel"] # B,J,3
    kp_angvel_w = kp_angvel["angvel"] # B,J,3
    
    # Param
    num_env, num_kp, _ = kp_pos_w.shape

    root_h = root_pos_w[...,2]
    heading_quat_inv = torch_utils.calc_heading_quat_inv(root_quat_w)
    # print(f"Heading Quat Inv : {heading_quat_inv}") # 1,4
    heading_quat_inv_expand = heading_quat_inv.unsqueeze(1).repeat((1,kp_pos_w.shape[1],1))
    
    # (B,J,3) - (B,3)
    local_body_pos = kp_pos_w - root_pos_w

    # local_body_pos : (B,J,3)
    # heading_quat_inv_expand : (B,J,4)
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], -1)
    flat_heading_quat_inv = heading_quat_inv_expand.reshape(heading_quat_inv_expand.shape[0] * heading_quat_inv_expand.shape[1], -1)
    local_body_pos_obs = torch_utils.my_quat_rotate(flat_heading_quat_inv, flat_local_body_pos)
    
    local_body_pos_obs = local_body_pos_obs.reshape(local_body_pos.shape[0], local_body_pos.shape[1]*3) # 1, 20*3
    local_body_pos_obs = local_body_pos_obs[...,3:].clone() # Root Link 제외
    # print(f"local_body_pos_obs :{local_body_pos_obs.shape}") # 1(B), 57

    # local_body_rot = quat_mul(heading_quat_inv, self.link_pose_tensor[...,3:])
    # local_body_rot_obs = torch_utils.quat_to_tan_norm(local_body_rot).view(B, time_steps, self.link_pose_tensor.shape[0] * 6)

    body_rot = kp_quat_w
    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])  # This is global rotation of the body
    flat_local_body_rot = quat_mul(flat_heading_quat_inv, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)

    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    # print(f"local_body_rot_obs :{local_body_rot_obs.shape}") # 1(B), 6*20

    body_vel = kp_linvel_w
    body_ang_vel = kp_angvel_w

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = torch_utils.my_quat_rotate(flat_heading_quat_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    # print(f"local_body_vel :{local_body_vel.shape}") # 1(B), 57

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = torch_utils.my_quat_rotate(flat_heading_quat_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    # print(f"local_body_ang_vel :{local_body_ang_vel.shape}") # 1(B), 57


    # return tensor : batch_size, --
    obs = torch.cat([root_h, local_body_pos_obs, local_body_rot_obs, local_body_vel, local_body_ang_vel], dim=-1)

    return obs
