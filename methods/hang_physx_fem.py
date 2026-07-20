"""
Cable-hang benchmark -- PHYSX FEM DEFORMABLE method (pure Isaac Sim).

A volumetric soft body: the cable is a triangulated cylinder, tetrahedralised
into a hexahedral simulation mesh and simulated as a PhysX deformable. It is the
only method here that models the cable as a 3-D continuum rather than a 1-D rod
or a chain, so it is the only one that can represent cross-section deformation.

TWO COMPROMISES ARE FORCED ON THIS METHOD, and both are worth understanding
before reading its error numbers.

1. THE ROD MUST BE FAT.  A FEM element cannot be arbitrarily slender before it
   becomes ill-conditioned, so a 2 mm-radius cable discretised end to end is not
   tractable: the simulation radius is inflated to ``--sim-radius`` (12 mm by
   default, 6x the real cable). To keep the fat rod behaving like the real thin
   one, the moduli are rescaled by the radius ratio q = r_real / r_sim:

       E_sim   = E   * q^4      (preserves bending stiffness E I, I ~ r^4)
       rho_sim = rho * q^2      (preserves mass per length, A ~ r^2)

   Bending stiffness and weight per length are therefore matched; the
   cross-section geometry is not. This is the "thinness floor" documented at
   length in the original IsaacLab work.

2. THE REST SHAPE MUST BE STRAIGHT.  PhysX auto-attachment fails on a curved
   rest mesh, so the cable cannot be built pre-sagged between the supports.
   It is built straight at full length L along +x, and the supports are then
   PULLED to their targets: each is a DYNAMIC cube overlapping the mesh (a
   kinematic body does not transmit motion through an attachment -- dynamic
   ones do), welded by a fixed joint to a KINEMATIC driver cube that is
   scripted smoothly to the target over ``--pull-seconds``. Once the drivers
   arrive the cable settles into its hanging shape.

   The consequence for the benchmark: unlike every other method, this one does
   NOT start on the analytic catenary. It starts straight and is dragged into
   place, so its settling transient is genuinely larger and its wall time is
   not comparable to the others' on equal terms.

The exported profile is the CENTRELINE of the fat rod -- simulation nodes binned
along x and averaged -- which is the fair thing to compare against the other
methods and the analytic curve.

    python.sh hang_physx_fem.py --points 2 --out results/x
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hang_common import method_parser, scenario_from_args  # noqa: E402

METHOD = "physx_fem"
LABEL = "PhysX FEM deformable"


def build_parser():
    p = method_parser("PhysX volumetric deformable cable hang.")
    p.add_argument("--sim-radius", type=float, default=12e-3,
                   help="inflated simulation radius [m]; moduli are rescaled to match")
    p.add_argument("--pos-iters", type=int, default=80)
    p.add_argument("--vertex-damping", type=float, default=0.5)
    p.add_argument("--elasticity-damping", type=float, default=0.3)
    p.add_argument("--mesh-segments", type=int, default=16, help="cross-section facets")
    p.add_argument("--mesh-stacks", type=int, default=80, help="lengthwise rings")
    p.add_argument("--fem-resolution", type=int, default=None,
                   help="hexahedral resolution (default: ~3 cells across the rod)")
    p.add_argument("--physics-dt", type=float, default=1.0 / 240.0)
    p.add_argument("--pull-start", type=float, default=0.5,
                   help="settle straight for this long before pulling [s]")
    p.add_argument("--pull-seconds", type=float, default=3.0,
                   help="time taken to drag the supports to their targets [s]")
    return p


args = build_parser().parse_args()
scenario = scenario_from_args(args)
headless = True if args.headless is None else args.headless

from isaacsim.simulation_app import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": headless})

try:
    from isaacsim.core.simulation_manager import SimulationManager

    SimulationManager.switch_physics_engine("physx")
except Exception as exc:  # pragma: no cover
    print(f"[warn] could not force the PhysX engine: {exc}")

import numpy as np  # noqa: E402
from isaacsim.core.api.objects import DynamicCuboid  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
# isaacsim.core.prims.DeformablePrim was removed in Isaac Sim 6 ("Omniverse
# PhysX removed the deprecated deformable body features"); the replacement
# lives in isaacsim.core.experimental and exposes get_nodal_positions()
# instead of get_simulation_mesh_nodal_positions().
from isaacsim.core.experimental.prims import DeformablePrim  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from cable_config import with_length  # noqa: E402
from hang_common import SettleMonitor, write_outputs  # noqa: E402
from isaac_env import isaac_version as _iv  # noqa: E402

isaac_version = _iv()

MESH_PATH = "/World/FemCable/source_mesh"
DEFORMABLE_ROOT = "/World/FemCable/body"
SIM_MESH_PATH = f"{DEFORMABLE_ROOT}/simulation_mesh"
DRIVER_Y = 0.12  # drivers sit beside the cable plane so they never touch it


def smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def centerline(nodes: np.ndarray, nbins: int) -> np.ndarray:
    """Centreline of the fat rod: nodes binned along x and averaged."""
    order = np.argsort(nodes[:, 0])
    nodes = nodes[order]
    edges = np.linspace(nodes[0, 0], nodes[-1, 0], nbins + 1)
    cents = []
    for k in range(nbins):
        sel = nodes[(nodes[:, 0] >= edges[k]) & (nodes[:, 0] <= edges[k + 1])]
        if len(sel):
            cents.append(sel.mean(axis=0))
    return np.asarray(cents)


def main() -> int:
    cable = with_length(scenario.length)
    print(scenario.describe())
    print(cable.summary())

    L = scenario.length
    R = args.sim_radius
    q = cable.radius / R
    e_sim = max(cable.youngs_modulus * q ** 4, 2.0e3)
    rho_sim = max(cable.density * q ** 2, 11.5)
    resolution = args.fem_resolution or int(
        min(max(math.ceil(3.0 * L / (2.0 * R)), 24), 130))

    world = World(stage_units_in_meters=1.0, physics_dt=args.physics_dt,
                  rendering_dt=1.0 / 60.0)
    stage = world.stage

    p1 = scenario.supports[0]
    x0, z0 = float(p1[0]), float(p1[2])

    # ---- straight cylinder mesh (curved rest meshes break auto-attachment) --
    nseg, nst = args.mesh_segments, args.mesh_stacks
    pts_, counts, idx = [], [], []
    for i in range(nst + 1):
        x = x0 + L * i / nst
        for j in range(nseg):
            th = 2.0 * math.pi * j / nseg
            pts_.append(Gf.Vec3f(x, R * math.cos(th), z0 + R * math.sin(th)))
    c0 = len(pts_); pts_.append(Gf.Vec3f(x0, 0.0, z0))
    c1 = len(pts_); pts_.append(Gf.Vec3f(x0 + L, 0.0, z0))

    def tri(a, b, c):
        counts.append(3)
        idx.extend((a, b, c))

    for i in range(nst):
        for j in range(nseg):
            jn = (j + 1) % nseg
            a, b = i * nseg + j, i * nseg + jn
            d, e = (i + 1) * nseg + j, (i + 1) * nseg + jn
            tri(a, b, e)
            tri(a, e, d)
    for j in range(nseg):
        jn = (j + 1) % nseg
        tri(c0, jn, j)
        tri(c1, nst * nseg + j, nst * nseg + jn)

    mesh = UsdGeom.Mesh.Define(stage, MESH_PATH)
    mesh.CreatePointsAttr(pts_)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.05, 0.05)])

    # Isaac Sim 6 replaced the old one-shot deformable API. The pre-6 calls
    # add_physx_deformable_body / add_deformable_body_material no longer exist;
    # a volume deformable is now an explicit HIERARCHY -- a root prim holding a
    # simulation tet mesh and a collision tet mesh, cooked from the source
    # triangle mesh.
    #
    # The root must already exist and be a UsdGeom.Imageable that is NOT a
    # Gprim -- an Xform. The helper only logs a carb warning and returns False
    # if it is missing, so creating it explicitly is required.
    UsdGeom.Xform.Define(stage, DEFORMABLE_ROOT)
    if not deformableUtils.create_auto_volume_deformable_hierarchy(
            stage,
            root_prim_path=DEFORMABLE_ROOT,
            simulation_tetmesh_path=f"{DEFORMABLE_ROOT}/simulation_mesh",
            collision_tetmesh_path=f"{DEFORMABLE_ROOT}/collision_mesh",
            cooking_src_mesh_path=MESH_PATH,
            simulation_hex_mesh_enabled=True,
            cooking_src_simplification_enabled=True):
        raise RuntimeError(
            "create_auto_volume_deformable_hierarchy failed -- try a lower "
            "--fem-resolution or a larger --sim-radius")

    mat_path = "/World/FemCable/material"
    deformableUtils.add_deformable_material(
        stage, mat_path, youngs_modulus=e_sim, poissons_ratio=cable.poisson_ratio,
        density=rho_sim, dynamic_friction=0.4)
    physicsUtils.add_physics_material_to_prim(
        stage, stage.GetPrimAtPath(DEFORMABLE_ROOT), mat_path)

    # ---- anchors and drivers --------------------------------------------
    def make_anchor(name, pos, fixed):
        path = f"/World/{name}"
        world.scene.add(DynamicCuboid(
            prim_path=path, name=name, position=pos,
            size=max(4.0 * R, 0.04), mass=0.01,
            color=np.array([0.2, 0.4, 0.8]) if fixed else np.array([0.95, 0.55, 0.05])))
        if fixed:
            fj = UsdPhysics.FixedJoint.Define(stage, f"/World/fix_{name}")
            fj.CreateBody1Rel().SetTargets([Sdf.Path(path)])
            fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in pos]))
            fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        else:
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(path))
            rb.CreateLinearDampingAttr().Set(1.0)
            rb.CreateAngularDampingAttr().Set(1.0)
        return path

    def make_driver(name, pos, anchor_path):
        path = f"/World/{name}"
        cube = world.scene.add(DynamicCuboid(
            prim_path=path, name=name, position=pos + np.array([0.0, DRIVER_Y, 0.0]),
            size=0.03, mass=0.05, color=np.array([0.4, 0.8, 0.4])))
        UsdPhysics.RigidBodyAPI.Apply(
            stage.GetPrimAtPath(path)).CreateKinematicEnabledAttr().Set(True)
        fj = UsdPhysics.FixedJoint.Define(stage, f"/World/weld_{name}")
        fj.CreateBody0Rel().SetTargets([Sdf.Path(path)])
        fj.CreateBody1Rel().SetTargets([Sdf.Path(anchor_path)])
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, -DRIVER_Y, 0.0))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        return cube

    def attach(anchor_path, name):
        # Isaac Sim 6: attachments are created through the deformable helper and
        # bind the deformable ROOT (not the source triangle mesh) to the rigid.
        if not deformableUtils.create_auto_deformable_attachment(
                stage, Sdf.Path(f"/World/{name}"),
                Sdf.Path(DEFORMABLE_ROOT), Sdf.Path(anchor_path)):
            raise RuntimeError(
                f"create_auto_deformable_attachment failed for {name} -- the "
                "anchor cube probably does not overlap the cable mesh")

    make_anchor("anchor_p1", p1.copy(), fixed=True)
    attach("/World/anchor_p1", "att_p1")

    straight_end = p1 + np.array([L, 0.0, 0.0])
    end_anchor = make_anchor("anchor_end", straight_end.copy(), fixed=False)
    attach(end_anchor, "att_end")
    pulls = [(make_driver("driver_end", straight_end.copy(), end_anchor),
              straight_end.copy(), scenario.supports[-1].copy())]

    if scenario.num_points == 3:
        # Equal-height supports split the cable in half, so the middle support
        # holds the material point at arc length L/2.
        straight_mid = p1 + np.array([0.5 * L, 0.0, 0.0])
        mid_anchor = make_anchor("anchor_mid", straight_mid.copy(), fixed=False)
        attach(mid_anchor, "att_mid")
        pulls.append((make_driver("driver_mid", straight_mid.copy(), mid_anchor),
                      straight_mid.copy(), scenario.supports[1].copy()))

    # Target the deformable ROOT (it carries DeformableBodyAPI). Pointing this
    # at the simulation tet mesh makes DeformablePrim try to re-apply the
    # schema to a prim that is already part of a body, which throws.
    view = DeformablePrim(DEFORMABLE_ROOT)
    world.reset()

    info = {
        "engine": "physx",
        "sim_radius_m": R,
        "radius_ratio": q,
        "youngs_modulus_sim_Pa": e_sim,
        "density_sim_kg_m3": rho_sim,
        "fem_resolution": resolution,
        "pos_iters": args.pos_iters,
        "physics_dt": args.physics_dt,
        "pull_start_s": args.pull_start,
        "pull_seconds_s": args.pull_seconds,
        "average_window_s": args.average_window,
        "note": "starts straight and is pulled into place, not initialised on the catenary",
    }
    print(f"[physx-fem] R_sim={R * 1e3:.0f} mm (q={q:.3f}), E_sim={e_sim:.3g} Pa, "
          f"rho_sim={rho_sim:.3g} kg/m^3, hex res={resolution}")

    def read_nodes():
        try:
            p = view.get_nodal_positions()
        except AttributeError as exc:
            raise RuntimeError(
                "PhysX deformable state cannot be read back on Isaac Sim "
                f"{isaac_version}.\n\n"
                f"  get_nodal_positions() failed: {exc}\n\n"
                "The deformable tensor view is created but its backend is None, so "
                "DeformablePrim._on_physics_ready dies reading "
                "num_nodes_per_element. The body itself cooks and simulates -- the "
                "hierarchy, material and attachments all succeed -- but there is no "
                "working route to its nodal positions, and without those there is "
                "no profile to export.\n\n"
                "This is the same class of gap as the Newton engine's tensors view "
                "(see hang_newton_engine.py). Reading the simulation TetMesh points "
                "straight from USD/Fabric each step is the likely workaround and is "
                "not implemented yet.") from exc
        if isinstance(p, (tuple, list)):
            p = p[0]
        if hasattr(p, "detach"):
            p = p.detach().cpu().numpy()
        p = np.asarray(p)
        return p[0] if p.ndim == 3 else p

    monitor = SettleMonitor(scenario)
    nbins = scenario.num_segments
    pull_end = args.pull_start + args.pull_seconds
    steps_per_sample = max(1, int(round(0.1 / args.physics_dt)))
    sim_time = 0.0
    try:
        while simulation_app.is_running() and sim_time < scenario.max_time:
            for _ in range(steps_per_sample):
                u = smoothstep((sim_time - args.pull_start) / args.pull_seconds)
                for driver, start, target in pulls:
                    driver.set_world_pose(
                        start + u * (target - start) + np.array([0.0, DRIVER_Y, 0.0]), None)
                world.step(render=False)
                sim_time += args.physics_dt

            nodes = read_nodes()
            if nodes is None or not np.isfinite(nodes).all():
                raise RuntimeError("FEM nodes invalid (cooking or attachment failure)")
            cl = centerline(nodes, nbins)
            # Do not let the settle test fire while the supports are still
            # being dragged: the cable is moving because it is being moved.
            settled = monitor.update(sim_time, cl, record=True)
            print(f"  pull={min(u, 1.0) * 100:3.0f}%  " + monitor.report(sim_time, cl).strip())
            if settled and sim_time > pull_end + 1.0:
                print("  settled.")
                break
            if sim_time <= pull_end + 1.0:
                monitor.settled = False  # provisional only

        write_outputs(args.out, scenario, monitor,
                      monitor.equilibrium_nodes(args.average_window),
                      METHOD, LABEL, extra=info)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
