"""E1 25-DOF robot asset configuration for parkour project."""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg

try:
    from isaaclab.actuators import DelayedImplicitActuatorCfg
except ImportError:  # IsaacLab versions without delayed implicit actuator
    from isaaclab.actuators import DelayedPDActuatorCfg as DelayedImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# ============================================================
# 执行器参数常量（与追球项目 e1.py 完全一致）
# ============================================================
ARMATURE_ANKLE_ROLL = 0.02
ARMATURE_ANKLE_PITCH = 0.033
ARMATURE_4340 = 0.032

NATURAL_FREQ = 3 * 2.0 * 3.1415926535  # 3Hz
NATURAL_FREQ_SQ = NATURAL_FREQ**2

# ============================================================
# URDF 路径
# ============================================================
__file_dir__ = os.path.dirname(os.path.realpath(__file__))
E1_URDF_PATH = os.path.abspath(
    os.path.join(__file_dir__, "resources", "noetix_e1", "urdf", "e1_25dof.urdf")
)

# ============================================================
# E1_25DOF_CFG — ArticulationCfg
# ============================================================
E1_25DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=E1_URDF_PATH,
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),#0.85
        joint_pos={
            # === 腿部 ===
            "l_leg_hip_yaw_joint": 0.0,
            "l_leg_hip_roll_joint": 0.0,
            "l_leg_hip_pitch_joint": -0.3,
            "l_leg_knee_joint": 0.55,
            "l_leg_ankle_pitch_joint": -0.35,
            "l_leg_ankle_roll_joint": 0.0,
            "r_leg_hip_yaw_joint": 0.0,
            "r_leg_hip_roll_joint": 0.0,
            "r_leg_hip_pitch_joint": -0.3,
            "r_leg_knee_joint": 0.55,
            "r_leg_ankle_pitch_joint": -0.35,
            "r_leg_ankle_roll_joint": 0.0,
            # === 腰部 ===
            "waist_yaw_joint": 0.0,
            # === 手臂 ===
            "l_arm_shoulder_pitch_joint": 0.0,
            "l_arm_shoulder_roll_joint": 0.2618,
            "l_arm_shoulder_yaw_joint": 0.0,
            "l_arm_elbow_pitch_joint": 0.0,
            "l_arm_elbow_yaw_joint": 0.0,
            "r_arm_shoulder_pitch_joint": 0.0,
            "r_arm_shoulder_roll_joint": -0.2618,
            "r_arm_shoulder_yaw_joint": 0.0,
            "r_arm_elbow_pitch_joint": 0.0,
            "r_arm_elbow_yaw_joint": 0.0,
            # === 头部（轻微俯视看前方地形）===
            "head_yaw_joint": 0.0,
            "head_pitch_joint": 0.87,  # pitch 0.87=49.8 0.95=54.4 0.52=30.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={},  # 由下方 e1_25dof_delayed_actuators 填充
)

# ============================================================
# 有延迟执行器（训练时使用）
# ============================================================
e1_25dof_delayed_actuators = {
    "legs": DelayedImplicitActuatorCfg(
        joint_names_expr=[
            ".*_hip_yaw_joint",
            ".*_hip_roll_joint",
            ".*_hip_pitch_joint",
            ".*_knee_joint",
        ],
        effort_limit_sim={
            ".*_hip_yaw_joint": 80.0,
            ".*_hip_roll_joint": 80.0,
            ".*_hip_pitch_joint": 140.0,
            ".*_knee_joint": 140.0,
        },
        velocity_limit_sim={
            ".*_hip_yaw_joint": 10.0,
            ".*_hip_roll_joint": 10.0,
            ".*_hip_pitch_joint": 14.0,
            ".*_knee_joint": 14.0,
        },
        stiffness={
            ".*_hip_yaw_joint": 0.6 * NATURAL_FREQ_SQ,
            ".*_hip_roll_joint": 0.4 * NATURAL_FREQ_SQ,
            ".*_hip_pitch_joint": 0.6 * NATURAL_FREQ_SQ,
            ".*_knee_joint": 0.2 * NATURAL_FREQ_SQ,
        },
        damping={
            ".*_hip_yaw_joint": 2 * 0.6 * NATURAL_FREQ,
            ".*_hip_roll_joint": 2 * 0.4 * NATURAL_FREQ,
            ".*_hip_pitch_joint": 2 * 0.6 * NATURAL_FREQ,
            ".*_knee_joint": 2 * 0.2 * NATURAL_FREQ,
        },
        min_delay=0,
        max_delay=2,
    ),
    "feet": DelayedImplicitActuatorCfg(
        joint_names_expr=[
            ".*_ankle_pitch_joint",
            ".*_ankle_roll_joint",
        ],
        effort_limit_sim=70.0,
        velocity_limit_sim=10.0,
        stiffness={
            ".*_ankle_pitch_joint": 2 * ARMATURE_ANKLE_PITCH * NATURAL_FREQ_SQ,
            ".*_ankle_roll_joint": 2 * ARMATURE_ANKLE_ROLL * NATURAL_FREQ_SQ,
        },
        damping={
            ".*_ankle_pitch_joint": 2 * 2 * ARMATURE_ANKLE_PITCH * NATURAL_FREQ,
            ".*_ankle_roll_joint": 2 * 2 * ARMATURE_ANKLE_ROLL * NATURAL_FREQ,
        },
        armature={
            ".*_ankle_pitch_joint": ARMATURE_ANKLE_PITCH,
            ".*_ankle_roll_joint": ARMATURE_ANKLE_ROLL,
        },
        friction={
            ".*_ankle_pitch_joint": 1.2,
            ".*_ankle_roll_joint": 0.6,
        },
        min_delay=0,
        max_delay=2,
    ),
    "waist": DelayedImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint"],
        effort_limit_sim=60.0,
        velocity_limit_sim=12.0,
        stiffness=150,
        damping=5,
        min_delay=0,
        max_delay=2,
    ),
    "arms": DelayedImplicitActuatorCfg(
        joint_names_expr=[
            ".*_shoulder_pitch_joint",
            ".*_shoulder_roll_joint",
            ".*_shoulder_yaw_joint",
            ".*_elbow_pitch_joint",
            ".*_elbow_yaw_joint",
        ],
        effort_limit_sim=20.0,
        velocity_limit_sim=10.0,
        stiffness=30,
        damping=1,
        armature=ARMATURE_4340,
        min_delay=0,
        max_delay=2,
    ),
    "head": DelayedImplicitActuatorCfg(
        joint_names_expr=[
            "head_yaw_joint",
            "head_pitch_joint",
        ],
        effort_limit_sim=10.0,
        velocity_limit_sim=8.0,
        stiffness=20,
        damping=1,
        min_delay=0,
        max_delay=2,
    ),
}

# ============================================================
# 无延迟执行器（部署 / 调试时使用）
# ============================================================
# 与 e1_25dof_delayed_actuators 相同参数，
# 但用 ImplicitActuatorCfg 替代 DelayedImplicitActuatorCfg，
# 去掉 min_delay / max_delay / armature / friction 字段。
# （此处省略，使用时按需构造）

# ============================================================
# E1_25DOF_ACTION_SCALE
# ============================================================
E1_25DOF_ACTION_SCALE = {}
for _a in e1_25dof_delayed_actuators.values():
    _e = _a.effort_limit_sim
    _s = _a.stiffness
    _names = _a.joint_names_expr
    if not isinstance(_e, dict):
        _e = {n: _e for n in _names}
    if not isinstance(_s, dict):
        _s = {n: _s for n in _names}
    for n in _names:
        if n in _e and n in _s and _s[n]:
            E1_25DOF_ACTION_SCALE[n] = 0.25 * _e[n] / _s[n]

# ============================================================
# 对称增强映射（IsaacLab 25-DOF 顺序）
# ============================================================

# --- 关节映射 (25 维): 左右对调，中轴→自身 ---
E1_25DOF_symmetric_augmentation_joint_mapping = [
    1,   # [0]  l_hip_yaw    ↔ [1]  r_hip_yaw
    0,   # [1]  r_hip_yaw    ↔ [0]  l_hip_yaw
    2,   # [2]  waist_yaw    → 自身
    4,   # [3]  l_hip_roll   ↔ [4]  r_hip_roll
    3,   # [4]  r_hip_roll   ↔ [3]  l_hip_roll
    5,   # [5]  head_yaw     → 自身
    7,   # [6]  l_shoulder_p ↔ [7]  r_shoulder_p
    6,   # [7]  r_shoulder_p ↔ [6]  l_shoulder_p
    9,   # [8]  l_hip_pitch  ↔ [9]  r_hip_pitch
    8,   # [9]  r_hip_pitch  ↔ [8]  l_hip_pitch
    10,  # [10] head_pitch   → 自身
    12,  # [11] l_shoulder_r ↔ [12] r_shoulder_r
    11,  # [12] r_shoulder_r ↔ [11] l_shoulder_r
    14,  # [13] l_knee       ↔ [14] r_knee
    13,  # [14] r_knee       ↔ [13] l_knee
    16,  # [15] l_shoulder_y ↔ [16] r_shoulder_y
    15,  # [16] r_shoulder_y ↔ [15] l_shoulder_y
    18,  # [17] l_ankle_p    ↔ [18] r_ankle_p
    17,  # [18] r_ankle_p    ↔ [17] l_ankle_p
    20,  # [19] l_elbow_p    ↔ [20] r_elbow_p
    19,  # [20] r_elbow_p    ↔ [19] l_elbow_p
    22,  # [21] l_ankle_r    ↔ [22] r_ankle_r
    21,  # [22] r_ankle_r    ↔ [21] l_ankle_r
    24,  # [23] l_elbow_y    ↔ [24] r_elbow_y
    23,  # [24] r_elbow_y    ↔ [23] l_elbow_y
]

# --- 关节符号翻转 (25 维): yaw/roll → -1, pitch → +1 ---
E1_25DOF_symmetric_augmentation_joint_reverse_buf = [
    -1,  # [0]  l_hip_yaw
    -1,  # [1]  r_hip_yaw
    -1,  # [2]  waist_yaw
    -1,  # [3]  l_hip_roll
    -1,  # [4]  r_hip_roll
    -1,  # [5]  head_yaw
     1,  # [6]  l_shoulder_pitch
     1,  # [7]  r_shoulder_pitch
     1,  # [8]  l_hip_pitch
     1,  # [9]  r_hip_pitch
     1,  # [10] head_pitch
    -1,  # [11] l_shoulder_roll
    -1,  # [12] r_shoulder_roll
     1,  # [13] l_knee (pitch)
     1,  # [14] r_knee
    -1,  # [15] l_shoulder_yaw
    -1,  # [16] r_shoulder_yaw
     1,  # [17] l_ankle_pitch
     1,  # [18] r_ankle_pitch
     1,  # [19] l_elbow_pitch
     1,  # [20] r_elbow_pitch
    -1,  # [21] l_ankle_roll
    -1,  # [22] r_ankle_roll
    -1,  # [23] l_elbow_yaw
    -1,  # [24] r_elbow_yaw
]

# --- Link 映射 (14 维): 对应 link_of_interests 的 L/R 互换 ---
E1_25DOF_symmetric_augmentation_link_mapping = [
    0,   # base_link → 自身
    1,   # waist_yaw_link → 自身
    3,   # l_arm_shoulder_roll ↔ r_arm_shoulder_roll
    2,
    5,   # l_arm_elbow_pitch ↔ r_arm_elbow_pitch
    4,
    7,   # l_leg_hip_roll ↔ r_leg_hip_roll
    6,
    9,   # l_leg_knee ↔ r_leg_knee
    8,
    11,  # l_leg_ankle_roll ↔ r_leg_ankle_roll
    10,
    12,  # head_yaw → 自身
    13,  # head_pitch → 自身
]
