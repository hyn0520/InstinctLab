import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates different legged robots.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
parser.add_argument("--live_plot", action="store_true", default=False, help="Plot some critical lines alive")
parser.add_argument("--video", type=str, default=None, help="Path to save the video.")
parser.add_argument(
    "--motion_npz",
    type=str,
    default=None,
    help="Optional: play a single retargeted motion .npz (e.g. e1_*_retargeted.npz). If set, overrides the default G1/AMASS setup.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=4000,
    help="Maximum simulation steps to run in headless mode (or when recording video).",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# Video export requires the rendering experience (headless rendering) to have camera pipelines enabled.
# AppLauncher selects the experience file based on `enable_cameras`.
if args_cli.video is not None:
    # This arg is injected by AppLauncher.add_app_launcher_args(parser).
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
from collections import deque

if args_cli.video:
    import imageio.v2 as iio

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass

from instinctlab.assets.unitree_g1 import G1_29DOF_TORSOBASE_CFG
from instinctlab.motion_reference import MotionReferenceManager
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.tasks.shadowing.whole_body.config.g1.plane_shadowing_cfg import (
    motion_reference_cfg as g1_motion_reference_cfg,
)

# from instinctlab.utils.retarget_smpl_to_joint import retarget_smpl_to_g1_29dof_joints
from instinctlab.utils.humanoid_ik import HumanoidSmplRotationalIK
from instinctlab.utils.live_plotter import LivePlotter

# wait for attach if in debug mode
if args_cli.debug:
    # import typing; typing.TYPE_CHECKING = True
    import debugpy

    ip_address = ("0.0.0.0", 6789)
    print("Process: " + " ".join(sys.argv[:]))
    print("Is waiting for attach at address: %s:%d" % ip_address, flush=True)
    debugpy.listen(ip_address)
    debugpy.wait_for_client()
    debugpy.breakpoint()

VIEWER_CFG = ViewerCfg()
VIEWER_CFG.resolution = (640, 360)

# ratio between the step_dt and sim_dt
DECIMATION = 4


@configclass
class AmassMotionCfg(AmassMotionCfgBase):
    # Default: G1 + AMASS retargeting (original behavior).
    clip_joint_ref_to_robot_limits = True
    path = os.path.expanduser("~/Datasets/AMASS/")
    retargetting_func = HumanoidSmplRotationalIK
    retargetting_func_kwargs = dict(
        # NOTE: Keep the original defaults. These are overridden when --motion_npz is used.
        robot_chain=G1_29DOF_TORSOBASE_CFG.spawn.asset_path,
        smpl_root_in_robot_link_name="pelvis",
        translation_scaling=0.75,
        translation_height_offset=0.0,
    )
    filtered_motion_selection_filepath = os.path.expanduser("~/Datasets/AMASS_selections/amass_test_motion_files.yaml")
    motion_start_from_middle_range = (0.0, 0.0)
    buffer_device = "cpu"


def _build_single_motion_selection_yaml(motion_npz: str) -> str:
    motion_npz = os.path.abspath(os.path.expanduser(motion_npz))
    if not motion_npz.endswith(".npz"):
        raise ValueError(f"--motion_npz must be a .npz file, got: {motion_npz}")
    if not os.path.exists(motion_npz):
        raise FileNotFoundError(motion_npz)
    motion_dir = os.path.dirname(motion_npz)
    motion_base = os.path.basename(motion_npz)
    out_yaml = os.path.join("/tmp", f"single_motion_{os.getpid()}.yaml")
    with open(out_yaml, "w", encoding="utf-8") as f:
        f.write("selected_files:\n")
        f.write(f"  - {motion_base}\n")
        f.write("weights:\n")
        f.write("  - 1.0\n")
    return out_yaml


@configclass
class SceneCfg(InteractiveSceneCfg):
    # common scene entities
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robots
    robot = G1_29DOF_TORSOBASE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # motion reference (default: original behavior)
    motion_reference = g1_motion_reference_cfg.replace(
        frame_interval_s=0.02,
        motion_buffers={
            "amass": AmassMotionCfg(),
        },
    )

    def __post_init__(self):
        if args_cli.motion_npz:
            # E1 retargeted motion playback (no SMPL retargeting needed).
            import copy

            from instinctlab.assets.noetix_e1 import E1_25DOF_CFG
            from instinctlab.motion_reference import MotionReferenceManagerCfg

            motion_npz = os.path.abspath(os.path.expanduser(args_cli.motion_npz))
            # Build a YAML selection file so AmassMotion only loads the requested file.
            selection_yaml = _build_single_motion_selection_yaml(motion_npz)

            # Configure E1 robot.
            e1_cfg = copy.deepcopy(E1_25DOF_CFG)
            e1_cfg.spawn.merge_fixed_joints = True
            e1_cfg.init_state.pos = (0.0, 0.0, 0.85)
            self.robot = e1_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

            # Configure motion buffer.
            @configclass
            class E1SingleMotionCfg(AmassMotionCfgBase):
                path = os.path.dirname(motion_npz)
                retargetting_func = None
                filtered_motion_selection_filepath = selection_yaml
                motion_start_from_middle_range = (0.0, 0.0)
                motion_start_height_offset = 0.0
                ensure_link_below_zero_ground = False
                buffer_device = "output_device"
                motion_interpolate_func = motion_interpolate_bilinear
                velocity_estimation_method = "frontward"

            # Build a minimal motion reference config for visualization.
            # Keep it self-contained to avoid task-level imports.
            self.motion_reference = MotionReferenceManagerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base_link",
                robot_model_path=e1_cfg.spawn.asset_path,
                frame_interval_s=0.02,
                update_period=0.02,
                num_frames=10,
                motion_buffers={"run_walk": E1SingleMotionCfg()},
                # Required for building forward-kinematics targets when loading retargeted motions.
                # Keep consistent with e1_parkour_target_amp_cfg.py.
                link_of_interests=[
                    "base_link",
                    "waist_yaw_link",
                    "l_arm_shoulder_roll_link",
                    "r_arm_shoulder_roll_link",
                    "l_arm_elbow_pitch_link",
                    "r_arm_elbow_pitch_link",
                    "l_leg_hip_roll_link",
                    "r_leg_hip_roll_link",
                    "l_leg_knee_link",
                    "r_leg_knee_link",
                    "l_leg_ankle_roll_link",
                    "r_leg_ankle_roll_link",
                    "head_yaw_link",
                    "head_pitch_link",
                ],
                mp_split_method="None",
            )

        # Normalize motion buffer sampling behavior for visualization.
        if hasattr(self.motion_reference, "motion_buffers"):
            for _, v in self.motion_reference.motion_buffers.items():
                v.motion_bin_length_s = None
                v.motion_start_from_middle_range = (0.0, 0.0)


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""

    sim_dt = sim.get_physics_dt()
    simulation_timestamp = 0
    robot: Articulation = scene["robot"]
    motion_reference: MotionReferenceManager = scene["motion_reference"]

    motion_reference.match_scene(scene)

    # prepare annotator product to record and save video
    if args_cli.video is not None:
        video_writer = iio.get_writer(
            args_cli.video,
            fps=1 / sim_dt / DECIMATION,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        # The IsaacLab headless experience may not load replicator by default.
        # Enable it explicitly for video rendering.
        try:
            import omni.kit.app

            ext_mgr = omni.kit.app.get_app().get_extension_manager()
            ext_mgr.set_extension_enabled_immediate("omni.replicator.core", True)
            ext_mgr.set_extension_enabled_immediate("omni.syntheticdata", True)
        except Exception:
            # If extension enable fails, the import below will raise a clearer error.
            pass

        import omni.replicator.core as rep
        # Some headless experiences don't create the Render/PostProcess prims that syntheticdata expects.
        # Pre-create them so SDG can attach its pipeline graph cleanly.
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                UsdGeom.Xform.Define(stage, "/Render")
                UsdGeom.Xform.Define(stage, "/Render/PostProcess")
        except Exception:
            pass

        _render_product = rep.create.render_product(VIEWER_CFG.cam_prim_path, VIEWER_CFG.resolution)
        _rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        _rgb_annotator.attach([_render_product])
        _video_interval_counter = 0

    # prepare live plots
    if args_cli.live_plot:
        plotter = LivePlotter(keys=["1"] * 12)
        _plotter_counter = 0

    # In headless runs, SimulationApp.is_running() may return False immediately.
    # For visualization/video export, run a fixed number of steps instead.
    use_fixed_steps = bool(getattr(args_cli, "headless", False) or args_cli.video is not None)
    fixed_steps = int(args_cli.max_steps) if use_fixed_steps else None

    step_count = 0
    while True:
        if (not use_fixed_steps) and (not simulation_app.is_running()):
            break
        if use_fixed_steps and fixed_steps is not None and step_count >= fixed_steps:
            break
        # Write data to sim

        # write robot data based on motion reference
        motion_reference_frame = motion_reference.reference_frame
        # robot.root_physx_view.set_dof_positions(
        #     motion_reference_frame.joint_pos[:, 0],
        #     indices=robot._ALL_INDICES,
        # )
        # robot.root_physx_view.set_dof_velocities(
        #     motion_reference_frame.joint_vel[:, 0],
        #     indices=robot._ALL_INDICES,
        # )
        # robot.root_physx_view.set_root_transforms(
        #     torch.concatenate(
        #         [
        #             motion_reference_frame.base_pos_w[:, 0],
        #             math_utils.convert_quat(motion_reference_frame.base_quat_w[:, 0], to="xyzw"),
        #         ],
        #         dim=-1,
        #     ),
        #     indices=robot._ALL_INDICES,
        # )
        robot.write_root_pose_to_sim(
            torch.concatenate(
                [
                    motion_reference_frame.base_pos_w[:, 0],
                    motion_reference_frame.base_quat_w[:, 0],
                ],
                dim=-1,
            ),
        )
        robot.write_joint_state_to_sim(
            motion_reference_frame.joint_pos[:, 0],
            motion_reference_frame.joint_vel[:, 0],
        )
        robot.write_root_velocity_to_sim(torch.zeros(robot.num_instances, 6, device=torch.device("cuda")))

        # reset motion reference if motion reference is exhausted
        reset_mask = torch.logical_not(motion_reference.data.validity.any(dim=-1))
        if reset_mask.any():
            reset_env_ids = torch.where(reset_mask)[0]
            motion_reference.reset(reset_env_ids)
            print("[INFO] current motion", motion_reference.get_current_motion_identifiers(reset_env_ids))

        # Perform render not physics steps
        sim.render()
        # Update buffers
        scene.update(sim_dt)

        if args_cli.video is not None:
            if _video_interval_counter % DECIMATION == 0:
                # obtain the rgb data
                rgb_data = _rgb_annotator.get_data()
                rgb_data = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
                if rgb_data.size == 0:
                    rgb_data = np.zeros((VIEWER_CFG.resolution[1], VIEWER_CFG.resolution[0], 3), dtype=np.uint8)
                else:
                    rgb_data = rgb_data[:, :, :3]
                # write to video
                video_writer.append_data(rgb_data)
            _video_interval_counter += 1

        if args_cli.live_plot:
            if _plotter_counter % DECIMATION == 0:
                _, robot_pitch, robot_yaw = math_utils.euler_xyz_from_quat(motion_reference_frame.base_quat_w[:, 0])
                base_ang_vel = motion_reference_frame.base_ang_vel_w[:, 0]
                joint_vel = motion_reference_frame.joint_vel[:, 0, :12]
                joint_pos = motion_reference_frame.joint_pos[:, 0, 9]
                plotter.plot(
                    [
                        # robot_pitch[0].item(),
                        i
                        for i in joint_vel[0].cpu().numpy()
                        # joint_pos[0].item(),
                        # base_ang_vel[0, 2].item(),
                        # robot_yaw[0].item(),
                    ],
                    x=simulation_timestamp,
                )
            _plotter_counter += 1

        simulation_timestamp += sim_dt
        step_count += 1
        if step_count % 50 == 0:
            print(f"[INFO] step={step_count}/{fixed_steps if fixed_steps is not None else -1}", flush=True)

    if args_cli.video is not None:
        video_writer.close()
        print("[INFO] video saved to: ", args_cli.video)


def main():
    """Main function."""
    # Load kit helper
    # For video rendering / replicator graphs we need USD write access. Fabric disables USD read/write.
    use_fabric = False if args_cli.video is not None else True
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, use_fabric=use_fabric)
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Design scene
    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)
    # close sim app
    simulation_app.close()


if __name__ == "__main__":
    # run the main execution
    main()
