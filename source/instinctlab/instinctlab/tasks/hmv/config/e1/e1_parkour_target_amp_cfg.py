"""E1 25-DOF parkour environment configuration.

对标: parkour/config/g1/g1_parkour_target_amp_cfg.py
"""

import copy
import os

from isaaclab.envs import ViewerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import instinctlab.tasks.hmv.mdp as mdp
from instinctlab.assets.noetix_e1 import (
    E1_25DOF_CFG,
    E1_25DOF_ACTION_SCALE,
    E1_25DOF_symmetric_augmentation_joint_mapping,
    E1_25DOF_symmetric_augmentation_joint_reverse_buf,
    E1_25DOF_symmetric_augmentation_link_mapping,
    e1_25dof_delayed_actuators,
)
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import (
    AmassMotionCfg as AmassMotionCfgBase,
)
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.sensors.noisy_camera import NoisyGroupedRayCasterCameraCfg
from instinctlab.tasks.hmv.config.parkour_env_cfg import (
    ROUGH_TERRAINS_CFG,
    ParkourEnvCfg,
)

# ============================================================
# Robot config
# ============================================================
__file_dir__ = os.path.dirname(os.path.realpath(__file__))
E1_CFG = copy.deepcopy(E1_25DOF_CFG)
E1_CFG.spawn.merge_fixed_joints = True
E1_CFG.init_state.pos = (0.0, 0.0, 0.85)

# ============================================================
# AmassMotionCfg
# ============================================================
@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    path = os.path.expanduser("/home/user/hyn/InstinctLab/e1_25dof")
    retargetting_func = None  # 数据已在 E1 关节空间
    filtered_motion_selection_filepath = os.path.expanduser(
        "/home/user/hyn/InstinctLab/e1_25dof/e1_parkour_motion.yaml"
    )
    motion_start_from_middle_range = [0.0, 0.9]
    motion_start_height_offset = 0.0
    ensure_link_below_zero_ground = False
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontward"


# ============================================================
# MotionReferenceManagerCfg
# ============================================================
motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_link",  # G1: torso_link
    robot_model_path=E1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_.*/RobotReference/base_link",
    symmetric_augmentation_link_mapping=E1_25DOF_symmetric_augmentation_link_mapping,
    symmetric_augmentation_joint_mapping=E1_25DOF_symmetric_augmentation_joint_mapping,
    symmetric_augmentation_joint_reverse_buf=E1_25DOF_symmetric_augmentation_joint_reverse_buf,
    frame_interval_s=0.02,
    update_period=0.02,
    num_frames=10,
    motion_buffers={
        "run_walk": AmassMotionCfg(),
    },
    # 14 个 link，左右成对排列（与 G1 同样 14 个）
    link_of_interests=[
        "base_link",                    # 0  G1: pelvis
        "waist_yaw_link",              # 1  G1: torso_link
        "l_arm_shoulder_roll_link",    # 2  ↔ 3
        "r_arm_shoulder_roll_link",    # 3  ↔ 2
        "l_arm_elbow_pitch_link",      # 4  ↔ 5
        "r_arm_elbow_pitch_link",      # 5  ↔ 4
        # G1 此处有 wrist_yaw_link × 2，E1 无腕关节
        "l_leg_hip_roll_link",         # 6  ↔ 7
        "r_leg_hip_roll_link",         # 7  ↔ 6
        "l_leg_knee_link",             # 8  ↔ 9
        "r_leg_knee_link",             # 9  ↔ 8
        "l_leg_ankle_roll_link",       # 10 ↔ 11
        "r_leg_ankle_roll_link",       # 11 ↔ 10
        "head_yaw_link",               # 12 E1 特有
        "head_pitch_link",             # 13 E1 特有
    ],
    mp_split_method="Even",
)


# ============================================================
# Play terrain variant
# ============================================================
ROUGH_TERRAINS_CFG_PLAY = copy.deepcopy(ROUGH_TERRAINS_CFG)
for _sub_name, _sub_cfg in ROUGH_TERRAINS_CFG_PLAY.sub_terrains.items():
    _sub_cfg.wall_prob = [0.0, 0.0, 0.0, 0.0]


# ============================================================
# E1ParkourRoughEnvCfg
# ============================================================
@configclass
class E1ParkourRoughEnvCfg(ParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- (a) 机器人 ---
        self.scene.robot = E1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators = e1_25dof_delayed_actuators

        # --- (b) Action Scale（基类硬编码了 G1 的 beyondmimic_action_scale）---
        self.actions.joint_pos.scale = E1_25DOF_ACTION_SCALE

        # --- (c) 深度相机（torso_link → head_pitch_link）---
        self.scene.camera.prim_path = "{ENV_REGEX_NS}/Robot/head_pitch_link"
        self.scene.camera.offset = NoisyGroupedRayCasterCameraCfg.OffsetCfg(
            pos=(0.0584, 0.0, 0.0324),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="world",  # ⚠️ 可能需要验证，追球项目用 "ros"
        )
        self.scene.camera.mesh_prim_paths = [
            "/World/ground/",
            "/World/envs/env_.*/Robot/(?!.*head_pitch_link).*",
        ]
        # E1 没有 rubber_hand 等 G1 特有 link
        self.scene.camera.aux_mesh_and_link_names = {}

        # --- (d) 高度扫描器 prim_path ---
        self.scene.left_height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/l_leg_ankle_roll_link"
        )
        self.scene.right_height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/r_leg_ankle_roll_link"
        )

        # --- (e) pelvis_orientation_l2: "pelvis" → "base_link" ---
        self.rewards.rewards.pelvis_orientation_l2.params["asset_cfg"] = (
            SceneEntityCfg("robot", body_names="base_link")
        )

        # --- (f) freeze_upper_body: 去 wrist，改 waist，加 head ---
        self.rewards.rewards.freeze_upper_body.params["asset_cfg"] = (
            SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*",
                    ".*_elbow_.*",
                    "waist_yaw_joint",
                    "head_yaw_joint",
                    "head_pitch_joint",
                ],
            )
        )

        # --- (g) base_contact 终止: "torso_link" → "waist_yaw_link" ---
        self.terminations.base_contact.params["sensor_cfg"] = (
            SceneEntityCfg("contact_forces", body_names="waist_yaw_link")
        )

        # --- (h) Motion Reference ---
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.motion_reference = motion_reference_cfg


# ============================================================
# Play config
# ============================================================
@configclass
class E1ParkourRoughEnvCfg_PLAY(E1ParkourRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG_PLAY
        self.scene.num_envs = 10
        self.viewer = ViewerCfg(
            eye=[4.0, 0.75, 1.0],
            lookat=[0.0, 0.75, 0.0],
            origin_type="asset_root",
            asset_name="robot",
        )
        self.scene.env_spacing = 2.5
        self.episode_length_s = 10
        self.terminations.root_height = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 4
            self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.leg_volume_points.debug_vis = True
        self.commands.base_velocity.debug_vis = True
        self.events.physics_material = None
        self.events.reset_robot_joints.params = {
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        }