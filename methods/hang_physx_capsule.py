"""
Cable-hang benchmark -- PHYSX CAPSULE CHAIN method (pure Isaac Sim).

The classic articulated-cable model, and the one most Isaac Sim cable work uses:
rigid capsule links joined by D6 joints whose translations are locked and whose
rotX/rotY axes carry angular drives acting as bending springs. The drive
stiffness comes from beam theory,

    k_bend = E I / h    [N m / rad]  ->  converted to N m / deg for UsdPhysics,

with damping set from a target ratio zeta against the critical damping of a
link rotating about its end, c_crit = 2 sqrt(k I_rot), I_rot = m h^2 / 3.

Rest state is STRAIGHT: the joint rotations are identity, so the initial sag
curvature is genuinely resisted by the bending springs, as it must be for a
cable that is straight when unloaded. That is the honest configuration -- baking
the sagged shape into the rest state would hide the bending response entirely.

This is a PhysX method, so it explicitly forces the PhysX engine: Isaac Sim 6
selects Newton by default, and silently benchmarking "PhysX" on Newton would
make the whole comparison meaningless.

    python.sh hang_physx_capsule.py --points 2 --out results/x
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hang_common import method_parser, scenario_from_args  # noqa: E402

METHOD = "physx_capsule"
LABEL = "PhysX capsule chain (D6)"


def build_parser():
    p = method_parser("PhysX rigid capsule chain with D6 bending springs.")
    p.add_argument("--zeta", type=float, default=0.05,
                   help="joint damping ratio against critical damping")
    p.add_argument("--linear-damping", type=float, default=0.05)
    p.add_argument("--angular-damping", type=float, default=0.5)
    p.add_argument("--pos-iters", type=int, default=32,
                   help="PhysX solver position iterations per link")
    p.add_argument("--vel-iters", type=int, default=4)
    p.add_argument("--physics-dt", type=float, default=1.0 / 480.0)
    return p


args = build_parser().parse_args()
scenario = scenario_from_args(args)
headless = True if args.headless is None else args.headless

# SimulationApp must exist before any other isaacsim import.
from isaacsim.simulation_app import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": headless})

try:
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.switch_physics_engine("physx")
except Exception as exc:  # pragma: no cover - depends on the Isaac build
    print(f"[warn] could not force the PhysX engine: {exc}")

import numpy as np  # noqa: E402
from isaacsim.core.api.objects import DynamicCapsule  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdPhysics  # noqa: E402

from cable_config import with_length  # noqa: E402
from hang_common import (  # noqa: E402
    SettleMonitor,
    initial_polyline,
    write_outputs,
)


def quat_z_to(direction: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating +Z onto the unit vector `direction`."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, direction))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about X
    axis = np.cross(z, direction)
    axis /= np.linalg.norm(axis)
    half = 0.5 * math.acos(max(-1.0, min(1.0, c)))
    return np.array([math.cos(half), *(math.sin(half) * axis)])


def main() -> int:
    cable = with_length(scenario.length)
    print(scenario.describe())
    print(cable.summary())

    pts = initial_polyline(scenario)
    num_links = scenario.num_segments
    seg_vec = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg_vec, axis=1)
    seg_dir = seg_vec / seg_len[:, None]
    h = float(seg_len.mean())

    radius = cable.radius
    link_mass = cable.mass / num_links

    k_bend_rad = cable.bending_stiffness / h                 # [N m / rad]
    joint_stiffness = k_bend_rad * math.pi / 180.0           # UsdPhysics wants /deg
    link_rot_inertia = link_mass * h ** 2 / 3.0
    c_crit = 2.0 * math.sqrt(max(k_bend_rad * link_rot_inertia, 1e-30))
    joint_damping = args.zeta * c_crit * math.pi / 180.0

    world = World(stage_units_in_meters=1.0,
                  physics_dt=args.physics_dt,
                  rendering_dt=1.0 / 60.0)
    stage = world.stage

    scene_prim = stage.GetPrimAtPath("/physicsScene")
    if scene_prim and scene_prim.IsValid():
        physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
        physx_scene.CreateEnableCCDAttr().Set(True)
        physx_scene.CreateSolverTypeAttr().Set("TGS")

    def tip_offset(i: int) -> float:
        return max(seg_len[i] - 2.0 * radius, 1e-4) / 2.0 + radius

    # ---- links ----------------------------------------------------------
    links, link_quats = [], []
    for i in range(num_links):
        center = 0.5 * (pts[i] + pts[i + 1])
        q = quat_z_to(seg_dir[i])
        link_quats.append(q)
        cap = world.scene.add(DynamicCapsule(
            prim_path=f"/World/capsule_{i}",
            name=f"capsule_{i}",
            position=center,
            orientation=q,
            radius=radius,
            height=max(seg_len[i] - 2.0 * radius, 1e-4),
            color=np.array([0.05, 0.05, 0.05]),
            mass=link_mass,
        ))
        links.append(cap)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(f"/World/capsule_{i}"))
        rb.CreateSolverPositionIterationCountAttr().Set(args.pos_iters)
        rb.CreateSolverVelocityIterationCountAttr().Set(args.vel_iters)
        rb.CreateLinearDampingAttr().Set(args.linear_damping)
        rb.CreateAngularDampingAttr().Set(args.angular_damping)
        rb.CreateEnableCCDAttr().Set(True)
        rb.CreateSleepThresholdAttr().Set(1e-5)
        rb.CreateStabilizationThresholdAttr().Set(1e-6)

    # ---- D6 joints with bending springs ---------------------------------
    for i in range(num_links - 1):
        joint = UsdPhysics.Joint.Define(stage, f"/World/link_joint_{i}")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/World/capsule_{i}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{i + 1}")])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, +tip_offset(i)))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -tip_offset(i + 1)))
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        prim = joint.GetPrim()
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(prim, axis)
            lim.CreateLowAttr().Set(1.0)
            lim.CreateHighAttr().Set(-1.0)  # low > high => locked
        for axis in ("rotX", "rotY"):
            drive = UsdPhysics.DriveAPI.Apply(prim, axis)
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(joint_stiffness)
            drive.CreateDampingAttr().Set(joint_damping)
            drive.CreateMaxForceAttr().Set(1e6)
        drive = UsdPhysics.DriveAPI.Apply(prim, "rotZ")  # twist: damped, no spring
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(0.0)
        drive.CreateDampingAttr().Set(joint_damping)

    # ---- supports -------------------------------------------------------
    def weld_to_world(name: str, link_index: int):
        center = 0.5 * (pts[link_index] + pts[link_index + 1])
        q = link_quats[link_index]
        fj = UsdPhysics.FixedJoint.Define(stage, f"/World/{name}")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{link_index}")])
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in center]))
        fj.CreateLocalRot0Attr().Set(Gf.Quatf(*[float(v) for v in q]))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    weld_to_world("fix_end_1", 0)
    weld_to_world("fix_end_2", num_links - 1)

    if scenario.mid_node is not None:
        # Translation-locked, rotation-free joint to the world: the middle
        # support holds that material point but lets the cable pivot through it.
        mid = scenario.mid_node
        joint = UsdPhysics.Joint.Define(stage, "/World/mid_hold")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/World/capsule_{mid}")])
        joint.CreateLocalPos0Attr().Set(
            Gf.Vec3f(*[float(v) for v in scenario.supports[1]]))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(*[float(v) for v in link_quats[mid]]))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, -tip_offset(mid)))
        joint.CreateExcludeFromArticulationAttr().Set(True)
        for axis in ("transX", "transY", "transZ"):
            lim = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            lim.CreateLowAttr().Set(1.0)
            lim.CreateHighAttr().Set(-1.0)

    world.reset()
    for _ in range(10):  # warm-up
        world.step(render=False)

    info = {
        "engine": "physx",
        "num_links": num_links,
        "segment_length_m": h,
        "bend_stiffness_N_m_per_rad": k_bend_rad,
        "drive_stiffness_N_m_per_deg": joint_stiffness,
        "drive_damping": joint_damping,
        "zeta": args.zeta,
        "physics_dt": args.physics_dt,
        "pos_iters": args.pos_iters,
        "vel_iters": args.vel_iters,
    }
    print(f"[physx-capsule] {num_links} links, k_bend={k_bend_rad:.3e} N m/rad "
          f"({joint_stiffness:.3e} N m/deg), damping={joint_damping:.3e}, dt={args.physics_dt:.5f} s")

    def read_nodes() -> np.ndarray:
        nodes = np.empty((num_links + 1, 3))
        for i, cap in enumerate(links):
            p, q = cap.get_world_pose()
            p = np.asarray(p, dtype=float)
            w, x, y, z = (float(v) for v in q)
            axis = np.array([2.0 * (x * z + y * w),
                             2.0 * (y * z - x * w),
                             1.0 - 2.0 * (x * x + y * y)])
            nodes[i] = p - tip_offset(i) * axis
            if i == num_links - 1:
                nodes[i + 1] = p + tip_offset(i) * axis
        return nodes

    monitor = SettleMonitor(scenario)
    monitor.update(0.0, read_nodes())

    steps_per_sample = max(1, int(round(0.1 / args.physics_dt)))  # sample every 0.1 s
    sim_time = 0.0
    try:
        while simulation_app.is_running() and sim_time < scenario.max_time:
            for _ in range(steps_per_sample):
                world.step(render=False)
            sim_time += steps_per_sample * args.physics_dt
            nodes = read_nodes()
            settled = monitor.update(sim_time, nodes)
            print(monitor.report(sim_time, nodes))
            if settled:
                print("  settled.")
                break

        info["average_window_s"] = args.average_window
        write_outputs(args.out, scenario, monitor,
                      monitor.equilibrium_nodes(args.average_window),
                      METHOD, LABEL, extra=info)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
