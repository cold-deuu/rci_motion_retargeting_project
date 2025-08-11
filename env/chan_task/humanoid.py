# Python Standard
import enum
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

# Linear Algebra
import torch
import numpy as np

# Data
from datetime import datetime
import imageio
from utils.flags import flags
from collections import defaultdict
import aiohttp, cv2, asyncio
import json
from collections import deque
import threading
from tqdm import tqdm

# IsaacLab
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply


# Robot 
from .humanoid_amp_env_cfg import HumanoidAmpEnvCfg

# Humanoid ENV 생성
class Humanoid(DirectRLEnv):
    cfg: HumanoidAmpEnvCfg

    def __init__(self, cfg: HumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        # Original Isaac Lab Start
        super().__init__(cfg, render_mode, **kwargs)

        dof_lower_limits = self.robot.data.soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = self.robot.data.soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # To Do : Key Body Names Load from config
        # Child Class 로 옮기기
        # Observation 에 사용?
        # key_body_names = ["right_elbow_link", "left_elbow_link", "right_ankle_link", "left_ankle_link"]
        # self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        # Original Isaac Lab End

        # 20250810
        # Gym to Lab : PHC
        # 1. This : Camera Rendering 
        # 1-1. render mode --> do not need
        # self.headless = cfg['headless']
        # if self.headless == False and not flags.no_virtual_display:
        #     from pyvirtualdisplay.smartdisplay import SmartDisplay
        #     self.virtual_display = SmartDisplay(size=(1800, 990), visible=True)
        #     self.virtual_display.start()


        # self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        # self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        # self.amp_observation_buffer = torch.zeros(
        #     (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        # )
        
        # Basic Config - base_task.py
        # self.device = cuda:0
        self.num_obs = self.cfg.observation_space
        self.num_actions = self.cfg.action_space
        self.num_state = self.cfg.state_space
        self.is_discrete = self.cfg.is_discrete
        self.control_freq_inv = self.cfg.decimation

        # allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # Humanoid Config - Humanoid.py
        # Standard H1 
        # To Do : Add H12
        self.dt = cfg.decimation * cfg.sim.dt


    # Setup Scene
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
        # build task observation
        obs = compute_obs(
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            # self.robot.data.body_pos_w[:, self.ref_body_index],
            # self.robot.data.body_quat_w[:, self.ref_body_index],
            # self.robot.data.body_lin_vel_w[:, self.ref_body_index],
            # self.robot.data.body_ang_vel_w[:, self.ref_body_index],
            self.robot.data.body_pos_w[:, self.key_body_indexes],
        )

        # Motion
        # # update AMP observation history
        # for i in reversed(range(self.cfg.num_amp_observations - 1)):
        #     self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # # build AMP observation
        # self.amp_observation_buffer[:, 0] = obs.clone()
        # self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        pass
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    def _reset_idx(self, env_ids: torch.Tensor | None):
        pass
    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass


    def load_robot_configs(self, cfg):
        self.load_common_humanoid_configs(cfg)
        self._has_upright_start = cfg["robot"].get("has_upright_start", True) # 똑바로 일어서서 시작할건지? Recovery 아니면 대부분 True 일듯
        self._real_weight = True
        self._body_names_orig = cfg["robot"].get("body_names", []) # 로봇 링크 이름
        
        _body_names_orig_copy = self._body_names_orig.copy() 
        self._full_track_bodies = _body_names_orig_copy

        _body_names_orig_copy = self._body_names_orig.copy()
        self._eval_bodies = _body_names_orig_copy # default eval bodies
        self._body_names = self._body_names_orig
        self._masterfoot_config = None
        self.dof_subset = torch.tensor([]).long()
        
        self._dof_names = cfg["robot"].get("dof_names", [])
        self.limb_weight_group = cfg["robot"].get("limb_weight_group", []) 
        self.limb_weight_group = [[self._body_names.index(g) for g in group] for group in self.limb_weight_group]
        # 링크 인덱스를 구하는 과정

    def load_common_humanoid_configs(self, cfg):
        self._divide_group = cfg["env"].get("divide_group", False) # False
        self._group_obs = cfg["env"].get("group_obs", False) # False
        self._disable_group_obs = cfg["env"].get("disable_group_obs", False) # False
        if self._divide_group:
            self._group_num_people = group_num_people = min(cfg['env'].get("num_env_group", 128), cfg['env']['num_envs'])
            self._group_ids = torch.tensor(np.arange(cfg["env"]["num_envs"] / group_num_people).repeat(group_num_people).astype(int))

        # To Do : Load FT Sensor
        self.force_sensor_joints = cfg["env"].get("force_sensor_joints", ["L_Ankle", "R_Ankle"]) # FT 
        
        ##### Robot Configs #####
        self._has_shape_obs = cfg.robot.get("has_shape_obs", False)
        self._has_shape_obs_disc = cfg.robot.get("has_shape_obs_disc", False)
        self._has_limb_weight_obs = cfg.robot.get("has_weight_obs", False)
        self._has_limb_weight_obs_disc = cfg.robot.get("has_weight_obs_disc", False)
        self.has_shape_variation = cfg.robot.get("has_shape_variation", False)
        self._bias_offset = cfg.robot.get("bias_offset", False)
        self._has_self_collision = cfg.robot.get("has_self_collision", False)
        self._has_mesh = cfg.robot.get("has_mesh", True)
        self._replace_feet = cfg.robot.get("replace_feet", True)  # replace feet or not
        self._has_jt_limit = cfg.robot.get("has_jt_limit", True)
        self._has_dof_subset = cfg.robot.get("has_dof_subset", False)
        self._has_smpl_pd_offset = cfg.robot.get("has_smpl_pd_offset", False)
        self._masterfoot = cfg.robot.get("masterfoot", False)
        self._freeze_toe = cfg.robot.get("freeze_toe", True)
        ##### Robot Configs #####
        
        
        self.shape_resampling_interval = cfg["env"].get("shape_resampling_interval", 100)
        self.getup_schedule = cfg["env"].get("getup_schedule", False) # Getup 일땐 True 아니면 False << 이거 마지막 failure recovery 같은데
        self._kp_scale = cfg["env"].get("kp_scale", 1.0)
        self._kd_scale = cfg["env"].get("kd_scale", self._kp_scale)
        
        self.hard_negative = cfg["env"].get("hard_negative", False)  # False
        self.cycle_motion = cfg["env"].get("cycle_motion", False)  # 얘도 get_up 일때 True 아니면 False
        self.power_reward = cfg["env"].get("power_reward", False) # 대부분 True
        self.obs_v = cfg["env"].get("obs_v", 1) # obs_v = 6
        self.amp_obs_v = cfg["env"].get("amp_obs_v", 1) # 대부분 1
        
        
        ## Kin stuff
        # Kinematic Loss 관련 텀들이라고 함
        # AMP 를 위한 세팅이라고 함
        # Humanoid.py 에서 세팅해놓지 뭐
        # 대부분 False
        self.kin_loss = cfg["env"].get("kin_loss", False)
        self.kin_lr = cfg["env"].get("kin_lr", 5e-4)
        self.z_readout = cfg["env"].get("z_readout", False)
        self.z_read = cfg["env"].get("z_read", False)
        self.z_uniform = cfg["env"].get("z_uniform", False)
        self.z_model = cfg["env"].get("z_model", False)
        self.distill = cfg["env"].get("distill", False)
        self.remove_disc_rot = cfg["env"].get("remove_disc_rot", False)
        
         ## ZL Devs
         # 연구/실험용 옵션이라고 함
        #################### Devs ####################
        self.fitting = cfg["env"].get("fitting", False) # get up 일때 True
        self.zero_out_far = cfg["env"].get("zero_out_far", False) # 일단 h1, g1 은 False
        self.zero_out_far_train = cfg["env"].get("zero_out_far_train", True) # 일단 h1, g1 은 False

        
        self.max_len = cfg["env"].get("max_len", -1) # 대부분 -1 아마 episode length?
        self.cycle_motion_xp = cfg["env"].get("cycle_motion_xp", False)  # Cycle motion, but cycle farrrrr.
                                                                         # 얘도 False
        self.models_path = cfg["env"].get("models", ['output/dgx/smpl_im_fit_3_1/Humanoid_00185000.pth', 'output/dgx/smpl_im_fit_3_2/Humanoid_00198750.pth']) # 얘는 학습 
        
        self.eval_full = cfg["env"].get("eval_full", False) # F
        self.auto_pmcp = cfg["env"].get("auto_pmcp", False) # F
        self.auto_pmcp_soft = cfg["env"].get("auto_pmcp_soft", False) # T
        self.strict_eval = cfg["env"].get("strict_eval", False) # F
        self._occl_training = cfg["env"].get("occl_training", False)  # F, Cycle motion, but cycle farrrrr.
        self._occl_training_prob = cfg["env"].get("occl_training_prob", 0.1)  # F, Cycle motion, but cycle farrrrr.
        self._sim_occlu = False
        self._res_action = cfg["env"].get("res_action", False) # F
        self.close_distance = cfg["env"].get("close_distance", 0.25) # 0.25
        self.far_distance = cfg["env"].get("far_distance", 3) # 3
        self._zero_out_far_steps = cfg["env"].get("zero_out_far_steps", 90) # 90
        self.past_track_steps = cfg["env"].get("past_track_steps", 5) # 5 << 이게 amp buffer 인건가
        #################### Devs ####################

        #################### Collect Dataset ####################
        self.add_obs_noise = cfg["env"].get("add_obs_noise", False) # F
        self.start_idx = cfg["env"].get("start_idx", 0) # 0
        self.add_action_noise = cfg["env"].get("add_action_noise", False) # F
        self.mlp_model_path = cfg["env"].get("mlp_model_path", "") 
        self.action_noise_std = cfg["env"].get("action_noise_std", 0.05)
        self.collect_dataset = cfg.get("collect_dataset", False)
        self.mlp_bypass = cfg["env"].get("mlp_bypass", False)


    def _setup_tensors(self):
        # --- Isaac Lab에서는 상태가 self.robot.data.* 로 바로 제공됩니다. ---
        data = self.robot.data  # 편의 참조

        # 1) Root state (월드 기준) : [num_envs, 13] (pos[3], quat[4], lin_vel[3], ang_vel[3])
        self._humanoid_root_states = data.root_state_w
        self._initail_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0.0  # 초기 속도/각속도 0

        # 2) DOF 상태 : 위치/속도 [num_envs, num_dof]
        self._dof_pos = data.joint_pos
        self._dof_vel = data.joint_vel

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device)

        # 3) (선택) 조인트 힘/토크: 구현에 따라 아래 중 하나를 사용
        #    - 제어기에서 실제로 가한 토크 값이 필요하면 applied_torque/commanded_torque 계열
        #    - 피직스가 보고하는 조인트 반력/힘이면 joint_force/actuator_force 계열
        # 프로젝트에 맞게 확인해서 사용하세요.
        if hasattr(data, "joint_effort"):
            self.dof_force_tensor = data.joint_effort                     # [E, dof]
        elif hasattr(data, "applied_torque"):
            self.dof_force_tensor = data.applied_torque                   # [E, dof]
        elif hasattr(data, "joint_actuator_force"):
            self.dof_force_tensor = data.joint_actuator_force             # [E, dof]
        else:
            self.dof_force_tensor = torch.zeros_like(self._dof_pos)

        # 4) 리지드 바디 상태 (월드 기준): 보통 [E, B, 13] 형태로 제공
        #    (바디별 pos/quat/lin_vel/ang_vel를 개별 텐서로도 제공)
        if hasattr(data, "body_state_w"):
            rb_state = data.body_state_w                                  # [E, B, 13]
            self._rigid_body_state_reshaped = rb_state
            self._rigid_body_pos = rb_state[..., :self.num_bodies, 0:3]
            self._rigid_body_rot = rb_state[..., :self.num_bodies, 3:7]
            self._rigid_body_vel = rb_state[..., :self.num_bodies, 7:10]
            self._rigid_body_ang_vel = rb_state[..., :self.num_bodies, 10:13]
        else:
            # 개별 텐서로 제공되는 경우
            self._rigid_body_pos = data.body_pos_w                        # [E, B, 3]
            self._rigid_body_rot = data.body_quat_w                       # [E, B, 4]
            self._rigid_body_vel = data.body_vel_w                        # [E, B, 3]
            self._rigid_body_ang_vel = data.body_ang_vel_w                # [E, B, 3]
            self._rigid_body_state_reshaped = torch.cat(
                [self._rigid_body_pos, self._rigid_body_rot,
                self._rigid_body_vel, self._rigid_body_ang_vel], dim=-1) # [E, B, 13]

        # 5) (옵션) 과거 히스토리 버퍼
        if self.self_obs_v == 2:
            self._rigid_body_pos_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device
            )
            self._rigid_body_rot_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 4), device=self.device
            )
            self._rigid_body_vel_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device
            )
            self._rigid_body_ang_vel_hist = torch.zeros(
                (self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device
            )

        # 6) 접촉 힘 (바디 당 순접촉력): [E, B, 3]
        self._contact_forces = data.net_contact_forces_w[..., :self.num_bodies, :3]

        # 7) (옵션) 포스 센서: Isaac Lab에서는 센서를 cfg에 등록해야 하며,
        #    등록 후 self.sensors["<name>"].data 를 통해 접근합니다.
        if self.self_obs_v == 3 and hasattr(self, "force_sensor_joints"):
            # 예시) ForceSensorCfg들을 "fs"라는 이름으로 등록했다고 가정
            fs_data = self.sensors["fs"].data  # 구성에 따라 net_forces_w, torques_w 등 제공
            # 센서당 6D(힘+토크)를 일렬로 편성
            # 실제 키/필드는 프로젝트의 센서 설정에 맞춰 교체
            vecs = []
            if hasattr(fs_data, "net_forces_w") and hasattr(fs_data, "net_torques_w"):
                vecs = [fs_data.net_forces_w, fs_data.net_torques_w]  # [E, S, 3] 각각
                self.vec_sensor_tensor = torch.cat(vecs, dim=-1).view(self.num_envs, -1)  # [E, S*6]
            else:
                # 최소 안전 처리
                self.vec_sensor_tensor = torch.zeros(self.num_envs, len(self.force_sensor_joints)*6, device=self.device)

        # 8) 종료 버퍼
        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # 9) 종료 높이 등 커스텀 전처리
        self._build_termination_heights()

        # 10) 바디/컨택트 인덱스: body_names에서 이름→인덱스 매핑
        body_names = data.body_names  # list[str]
        self._key_body_ids = torch.tensor(
            [body_names.index(nm) for nm in self.key_bodies],
            device=self.device, dtype=torch.long
        )
        contact_bodies = self.cfg.env.contact_bodies
        self._contact_body_ids = torch.tensor(
            [body_names.index(nm) for nm in contact_bodies],
            device=self.device, dtype=torch.long
        )

        # 11) 뷰어/카메라
        if self.viewer is not None:
            self._init_camera()  # Lab에서도 똑같이 내부 카메라 세팅 함수 호출 (구현은 Lab API로)



    def _setup_tensors_gym(self):
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        # ZL: needs to put this back
        if self.self_obs_v == 3:
            sensors_per_env = len(self.force_sensor_joints)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)
        

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._root_states = gymtorch.wrap_tensor(actor_root_state)
        num_actors = self.get_num_actors_per_env()

        self._humanoid_root_states = self._root_states.view(self.num_envs, num_actors, actor_root_state.shape[-1])[..., 0, :]
        self._initial_humanoid_root_states = self._humanoid_root_states.clone()
        self._initial_humanoid_root_states[:, 7:13] = 0

        self._humanoid_actor_ids = num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        dofs_per_env = self._dof_state.shape[0] // self.num_envs
        self._dof_pos = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 0]
        self._dof_vel = self._dof_state.view(self.num_envs, dofs_per_env, 2)[..., :self.num_dof, 1]

        self._initial_dof_pos = torch.zeros_like(self._dof_pos, device=self.device, dtype=torch.float)
        self._initial_dof_vel = torch.zeros_like(self._dof_vel, device=self.device, dtype=torch.float)

        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_state)
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        self._rigid_body_state_reshaped = self._rigid_body_state.view(self.num_envs, bodies_per_env, 13)

        self._rigid_body_pos = self._rigid_body_state_reshaped[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self._rigid_body_state_reshaped[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state_reshaped[..., :self.num_bodies, 10:13]
        
        if self.self_obs_v == 2:
            self._rigid_body_pos_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device, dtype=torch.float)
            self._rigid_body_rot_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 4), device=self.device, dtype=torch.float)
            self._rigid_body_vel_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device, dtype=torch.float)
            self._rigid_body_ang_vel_hist = torch.zeros((self.num_envs, self.past_track_steps, self.num_bodies, 3), device=self.device, dtype=torch.float)

        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., :self.num_bodies, :]

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        self._build_termination_heights()
        
        contact_bodies = self.cfg["env"]["contact_bodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(self.key_bodies)

        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        if self.viewer != None or flags.server_mode:
            self._init_camera()
