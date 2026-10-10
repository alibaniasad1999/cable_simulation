"""The fitted cable held by a Franka in Isaac Sim (PhysX capsule chain), to test it there.

    # 1. in your Newton environment, once: a run of the fitted scene (writes meta.json, init_frames.csv)
    python 03_franka_holds_cable/ethernet_scene.py --config results/ethernet_cat6_fit/best_config.json

    # 2. with Isaac Sim's Python:
    ~/isaacsim/python.sh 03_franka_holds_cable/isaac_cable_scene.py --run results/ethernet_cat6_fit/best_run/scan/bend_x1

    # 3. compare Newton, Isaac and the scan in one table/plot (both runs sit in .../scan/):
    python 04_pointcloud_vs_sim/compare_to_scan.py --config configs/ethernet_cat6.json --runs results/ethernet_cat6_fit/best_run/scan

Everything comes from the Newton run folder: the cable's start pose (init_frames.csv,
twist included), the fitted EI / GJ / curl, mass, radius, segment length, the table
height and the Franka's base and joint angles (meta.json). Nothing is re-fitted.

The same cable in PhysX:
  * one rigid capsule per segment (total length = segment length), mass per metre from the JSON;
  * D6 joints, translations locked; angular drives as springs: twist about the joint's X
    axis = the capsule axis (GJ/h), bending about Y and Z (EI/h). USD wants N m per DEGREE;
  * the natural curl is built into the joint frames: with frames aligned the two
    segments sit at the rest angle (kappa*h about the curl axis), so no drive targets;
  * grip segments (and the plug, if it was held in Newton) are kinematic: held in place;
  * the Franka (Isaac's own franka.usd, a Panda: same kinematics as the FR3 used in Newton)
    at the same base pose and joint angles. Robot-cable collisions are filtered out: it is
    only visual, as in Newton.

Output: <run>/../isaac_physx/ in the same format as a Newton run (meta.json,
sim_centerline.csv, init_centerline.csv, sim_centerline.ply), so compare_to_scan.py reads it.

Written for the Isaac Sim 6.0 API used by legacy/methods/hang_physx_capsule.py.
Other versions may need small changes (asset path, import names): the error says where.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
ap.add_argument("--run", required=True, help="a Newton run folder of the fitted scene (has meta.json, init_frames.csv)")
ap.add_argument("--out", default=None, help="output folder (default <run>/../isaac_physx)")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--no-robot", action="store_true", help="leave the Franka out")
ap.add_argument("--physics-dt", type=float, default=1.0 / 480.0)
ap.add_argument("--pos-iters", type=int, default=32, help="PhysX position iterations per body")
ap.add_argument("--vel-iters", type=int, default=4)
ap.add_argument("--max-time", type=float, default=10.0, help="stop after this many simulated seconds")
ap.add_argument("--settle-mm", type=float, default=0.1, help="settled when nothing moved more than this in 0.5 s")
args = ap.parse_args()

run_dir = Path(args.run).resolve()
meta = json.loads((run_dir / "meta.json").read_text())
frames = np.loadtxt(run_dir / "init_frames.csv", delimiter=",", skiprows=1)  # px py pz qx qy qz qw
if frames.ndim != 2 or frames.shape[1] != 7:
    raise SystemExit("init_frames.csv has the wrong shape: rerun ethernet_scene.py with the current code")
out_dir = Path(args.out).resolve() if args.out else run_dir.parent / "isaac_physx"

# SimulationApp must exist before any other isaacsim / pxr import.
from isaacsim.simulation_app import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})
try:  # Isaac Sim 6 can default to Newton; this is a PhysX test
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.switch_physics_engine("physx")
except Exception as exc:  # pragma: no cover - depends on the Isaac build
    print(f"[warn] could not force the PhysX engine: {exc}")

from isaacsim.core.api.objects import DynamicCapsule  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics  # noqa: E402


# =============================================================================
# Small helpers (numpy only; quaternions (x, y, z, w) unless said otherwise)
# =============================================================================
def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qaxis(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([*(math.sin(angle / 2) * axis), math.cos(angle / 2)])


def qconj(q):
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def gf_quat(q):  # (x, y, z, w) -> Gf.Quatf(w, x, y, z)
    return Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def local_z(q_wxyz):
    w, x, y, z = (float(v) for v in q_wxyz)
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


# =============================================================================
# Scene
# =============================================================================
h = float(meta["segment_length_m"])
r = float(meta["radius_m"])
n = len(frames)
n_grip, n_plug = int(meta["grip_segments"]), int(meta["plug_segments"])
plug_fixed = bool(meta.get("plug_end_fixed", False))
EI, GJ = float(meta["EI_Nm2"]), float(meta["GJ_Nm2"])
kappa = float(meta.get("rest_curvature_per_m", 0.0))
phi = math.radians(float(meta.get("rest_curl_direction_deg", 0.0)))
mu_lin = float(meta.get("mass_per_length_kg_m", 0.045))
tau = float(meta.get("bend_damping_time_s", 0.002))
mu = float(meta.get("friction_mu", 0.5))
ground = float(meta["ground_z_m"])

DEG = math.pi / 180.0  # UsdPhysics angular drives: per degree
k_bend, k_twist = EI / h, GJ / h  # [N m / rad]
print(f"[isaac] {n} capsules of {1000 * h:.0f} mm, r {1000 * r:.2f} mm, EI {EI:.4g}, GJ {GJ:.4g}, "
      f"curl {kappa:.2f}/m @ {math.degrees(phi):.0f} deg, grip {n_grip}, plug {n_plug} "
      f"({'held' if plug_fixed else 'free'}), dt {args.physics_dt:.5f} s")

world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt, rendering_dt=1.0 / 60.0)
stage = world.stage
scene_prim = stage.GetPrimAtPath("/physicsScene")
if scene_prim and scene_prim.IsValid():
    px = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    px.CreateSolverTypeAttr().Set("TGS")
    px.CreateEnableCCDAttr().Set(True)
world.scene.add_default_ground_plane(z_position=ground, static_friction=mu, dynamic_friction=mu, restitution=0.0)

# --- cable --------------------------------------------------------------------
links = []
for i in range(n):
    p, q = frames[i, :3], frames[i, 3:7]
    m = mu_lin * h
    if not plug_fixed and n_plug and i >= n - n_plug:
        m += float(meta.get("plug_mass_kg", 0.0)) / n_plug
    path = f"/World/cable/seg_{i:03d}"
    links.append(world.scene.add(DynamicCapsule(
        prim_path=path, name=f"seg_{i:03d}", position=p, orientation=wxyz(q), radius=r,
        height=max(h - 2.0 * r, 1e-4), color=np.array([0.05, 0.25, 0.85]), mass=m)))
    prim = stage.GetPrimAtPath(path)
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSolverPositionIterationCountAttr().Set(args.pos_iters)
    rb.CreateSolverVelocityIterationCountAttr().Set(args.vel_iters)
    rb.CreateLinearDampingAttr().Set(0.05)
    rb.CreateAngularDampingAttr().Set(0.5)
    rb.CreateEnableCCDAttr().Set(True)
    if i < n_grip or (plug_fixed and i >= n - max(1, n_plug)):
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr().Set(True)

# Joint frames: X of the joint along the capsule axis (Z), so PhysX's twist axis is the cable axis.
q_x_to_z = qaxis([0.0, 1.0, 0.0], -math.pi / 2)
rest = qaxis([math.cos(phi), math.sin(phi), 0.0], kappa * h) if kappa > 0 else np.array([0.0, 0.0, 0.0, 1.0])
q_child = qmul(qconj(rest), q_x_to_z)  # aligned frames  <=>  segment i+1 = segment i turned by `rest`
for i in range(n - 1):
    j = UsdPhysics.Joint.Define(stage, f"/World/cable/joint_{i:03d}")
    j.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/cable/seg_{i:03d}")])
    j.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/cable/seg_{i + 1:03d}")])
    j.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.5 * h))
    j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -0.5 * h))
    j.CreateLocalRot0Attr().Set(gf_quat(q_x_to_z))
    j.CreateLocalRot1Attr().Set(gf_quat(q_child))
    j.CreateCollisionEnabledAttr().Set(False)
    j.CreateExcludeFromArticulationAttr().Set(True)
    prim = j.GetPrim()
    for axis in ("transX", "transY", "transZ"):
        lim = UsdPhysics.LimitAPI.Apply(prim, axis)
        lim.CreateLowAttr().Set(1.0)
        lim.CreateHighAttr().Set(-1.0)  # low > high: locked
    for axis, k in (("rotX", k_twist), ("rotY", k_bend), ("rotZ", k_bend)):
        d = UsdPhysics.DriveAPI.Apply(prim, axis)
        d.CreateTypeAttr().Set("force")
        d.CreateStiffnessAttr().Set(k * DEG)
        d.CreateDampingAttr().Set(tau * k * DEG)
        d.CreateMaxForceAttr().Set(1e6)
        d.CreateTargetPositionAttr().Set(0.0)

# Collision groups: the robot never touches the cable; the cable touches itself only if it did in Newton.
cable_group = UsdPhysics.CollisionGroup.Define(stage, "/World/collision_groups/cable")
cable_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(Sdf.Path("/World/cable"))
if not meta.get("self_collision", True):
    cable_group.CreateFilteredGroupsRel().AddTarget(Sdf.Path("/World/collision_groups/cable"))

# --- robot (visual) -----------------------------------------------------------
robot, q_robot = None, None
if not args.no_robot and meta.get("robot"):
    import importlib

    from isaacsim.core.utils.stage import add_reference_to_stage

    root = None
    for mod in ("isaacsim.storage.native", "isaacsim.core.utils.nucleus"):
        try:
            root = importlib.import_module(mod).get_assets_root_path()
            break
        except Exception:
            continue
    if root is None:
        raise SystemExit("could not find the Isaac assets root (Nucleus / local assets); use --no-robot")
    usd = None
    for rel in ("/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "/Isaac/Robots/Franka/franka.usd"):
        candidate = root + rel
        try:
            add_reference_to_stage(usd_path=candidate, prim_path="/World/franka")
            if stage.GetPrimAtPath("/World/franka").IsValid() and stage.GetPrimAtPath("/World/franka").GetChildren():
                usd = candidate
                break
        except Exception:
            continue
    if usd is None:
        raise SystemExit(f"Franka asset not found under {root}; use --no-robot")
    from isaacsim.core.api.robots import Robot

    base = np.asarray(meta["robot"]["base_xyz_m"], float)
    yaw = math.radians(meta["robot"]["base_yaw_deg"])
    robot = world.scene.add(Robot(prim_path="/World/franka", name="franka", position=base,
                                  orientation=np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])))
    q_robot = np.asarray(meta["robot"]["joint_q"], float)
    robot_group = UsdPhysics.CollisionGroup.Define(stage, "/World/collision_groups/robot")
    robot_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(Sdf.Path("/World/franka"))
    robot_group.CreateFilteredGroupsRel().AddTarget(Sdf.Path("/World/collision_groups/cable"))
    print(f"[isaac] Franka from {usd}")

world.reset()
if robot is not None:
    from isaacsim.core.utils.types import ArticulationAction

    nd = robot.num_dof
    q_set = np.zeros(nd)
    q_set[: min(nd, len(q_robot))] = q_robot[: min(nd, len(q_robot))]
    robot.set_joint_positions(q_set)
    hold = ArticulationAction(joint_positions=q_set)


# =============================================================================
# Run until it settles
# =============================================================================
def read_nodes():
    pos, axis = np.empty((n, 3)), np.empty((n, 3))
    for i, link in enumerate(links):
        p, q = link.get_world_pose()
        pos[i], axis[i] = np.asarray(p, float), local_z(q)
    nodes = np.empty((n + 1, 3))
    nodes[0] = pos[0] - 0.5 * h * axis[0]
    nodes[1:-1] = 0.5 * ((pos[:-1] + 0.5 * h * axis[:-1]) + (pos[1:] - 0.5 * h * axis[1:]))
    nodes[-1] = pos[-1] + 0.5 * h * axis[-1]
    return nodes


steps_per_check = max(1, round(0.5 / args.physics_dt))
t, settled, wall0 = 0.0, False, time.time()
snap = read_nodes()
try:
    while app.is_running() and t < args.max_time:
        for _ in range(steps_per_check):
            if robot is not None:
                robot.get_articulation_controller().apply_action(hold)
            world.step(render=not args.headless)
        t += steps_per_check * args.physics_dt
        nodes = read_nodes()
        if not np.all(np.isfinite(nodes)):
            raise SystemExit("PhysX cable blew up (NaN): try a smaller --physics-dt or more --pos-iters")
        drift = float(np.linalg.norm(nodes - snap, axis=1).max())
        snap = nodes
        print(f"  t = {t:4.1f} s   shape change over 0.5 s: {1000 * drift:7.3f} mm")
        if drift < args.settle_mm / 1000.0 and t >= 1.0:
            settled = True
            break

    # --- write a run folder that compare_to_scan.py understands --------------
    out_dir.mkdir(parents=True, exist_ok=True)
    init = np.genfromtxt(run_dir / "init_centerline.csv", delimiter=",", names=True)
    shutil.copy(run_dir / "init_centerline.csv", out_dir / "init_centerline.csv")
    np.savetxt(out_dir / "sim_centerline.csv", np.column_stack([init["s_m"], nodes, init["part"]]), delimiter=",",
               header="s_m,x_m,y_m,z_m,part", comments="", fmt=["%.6f"] * 4 + ["%d"])
    arc = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())
    meta_out = dict(meta)
    meta_out.update({"engine": "isaac_physx", "label": "Isaac Sim (PhysX)", "settled": settled,
                     "sim_time_s": round(t, 3), "wall_time_s": round(time.time() - wall0, 1),
                     "stretch_percent": 100.0 * (arc / float(meta["length_total_m"]) - 1.0),
                     "physics_dt": args.physics_dt, "pos_iters": args.pos_iters, "vel_iters": args.vel_iters,
                     "source_run": str(run_dir)})
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, default=float))
    summary = Path(meta["scan"]).parent / "summary.json"
    units = json.loads(summary.read_text()).get("ply_units", "m") if summary.exists() else "m"
    R = np.asarray(meta["scan_to_world"], float)
    pts = (nodes @ R) * (1000.0 if units == "mm" else 1.0)  # back to the scan's frame and units
    with open(out_dir / "sim_centerline.ply", "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n"
                "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for x, y, z in pts:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} 235 104 52\n")
    print(f"[isaac] {'settled' if settled else 'NOT settled'} at t = {t:.2f} s, stretch "
          f"{meta_out['stretch_percent']:+.3f} %  ->  {out_dir}")

    if not args.headless:  # keep the window until it is closed
        while app.is_running():
            if robot is not None:
                robot.get_articulation_controller().apply_action(hold)
            world.step(render=True)
finally:
    app.close()
