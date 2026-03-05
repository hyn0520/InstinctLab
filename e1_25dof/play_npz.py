import argparse

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Play a retargeted motion (.npz) on E1 in IsaacLab.")
parser.add_argument(
    "--npz",
    type=str,
    default="/home/user/hyn/Humanoid-Monument-Valley/e1_25dof/e1_walk_forward_retargeted.npz",
    help="Path to retargeted .npz motion file",
)
parser.add_argument("--loop", action="store_true", help="Loop the motion")
parser.add_argument(
    "--quat_order",
    type=str,
    default="wxyz",
    choices=["wxyz", "xyzw"],
    help="Quaternion order stored in npz base_quat_w",
)
parser.add_argument("--fps", type=float, default=None, help="Override motion framerate")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to spawn")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# rest imports after app launch
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import Timer, configclass

from instinctlab.assets.noetix_e1 import E1_25DOF_CFG

VIEWER_CFG = ViewerCfg()
VIEWER_CFG.resolution = (640, 360)


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    robot = E1_25DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _maybe_reorder_joints(joint_pos: np.ndarray, joint_names: list[str], robot_joint_names: list[str]) -> np.ndarray:
    if not joint_names:
        return joint_pos
    idx = [joint_names.index(name) for name in robot_joint_names]
    return joint_pos[:, idx]


def run_sim(sim: SimulationContext, scene: InteractiveScene):
    # load motion
    data = np.load(args_cli.npz, allow_pickle=True)
    framerate = float(data["framerate"]) if args_cli.fps is None else float(args_cli.fps)
    base_pos = data["base_pos_w"].astype(np.float32)
    base_quat = data["base_quat_w"].astype(np.float32)
    joint_pos = data["joint_pos"].astype(np.float32)
    joint_vel = data["joint_vel"].astype(np.float32) if "joint_vel" in data else None
    joint_names = data["joint_names"].tolist() if "joint_names" in data else []

    num_frames = base_pos.shape[0]

    # reorder joint positions to robot joint order if names are provided
    robot = scene["robot"]
    if joint_names:
        joint_pos = _maybe_reorder_joints(joint_pos, joint_names, robot.joint_names)
        if joint_vel is not None:
            joint_vel = _maybe_reorder_joints(joint_vel, joint_names, robot.joint_names)

    # fix quaternion order if needed
    if args_cli.quat_order == "xyzw":
        base_quat = base_quat[:, [3, 0, 1, 2]]

    device = torch.device(args_cli.device)
    base_pos_t = torch.tensor(base_pos, device=device)
    base_quat_t = torch.tensor(base_quat, device=device)
    joint_pos_t = torch.tensor(joint_pos, device=device)
    joint_vel_t = torch.tensor(joint_vel, device=device) if joint_vel is not None else None

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0

    while simulation_app.is_running():
        frame_idx = int(sim_time * framerate)
        if frame_idx >= num_frames:
            if args_cli.loop:
                frame_idx = frame_idx % num_frames
                sim_time = 0.0
            else:
                break

        # gather frame data
        pos = base_pos_t[frame_idx].unsqueeze(0).repeat(robot.num_instances, 1)
        quat = base_quat_t[frame_idx].unsqueeze(0).repeat(robot.num_instances, 1)
        qpos = joint_pos_t[frame_idx].unsqueeze(0).repeat(robot.num_instances, 1)
        if joint_vel_t is not None:
            qvel = joint_vel_t[frame_idx].unsqueeze(0).repeat(robot.num_instances, 1)
        else:
            qvel = torch.zeros_like(qpos)

        # write to sim
        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        robot.write_joint_state_to_sim(qpos, qvel)
        robot.write_root_velocity_to_sim(torch.zeros(robot.num_instances, 6, device=device))

        # step
        sim.render()
        scene.update(sim_dt)
        sim_time += sim_dt


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    scene_cfg = SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0, replicate_physics=False)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] Setup complete. Playing motion:", args_cli.npz)
    run_sim(sim, scene)
    simulation_app.close()


if __name__ == "__main__":
    main()
