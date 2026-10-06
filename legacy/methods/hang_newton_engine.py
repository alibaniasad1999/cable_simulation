"""
Cable-hang benchmark -- NEWTON ENGINE method (Isaac Sim 6 native integration).

Drives Newton the way Isaac Sim 6 exposes it to ordinary users: the scene is
plain UsdPhysics -- capsule rigid bodies linked by spherical joints along the
hang polyline, ends welded to the world -- and ``SimulationManager`` switches
the active engine to ``"newton"``, whose XPBD solver handles the chain in
maximal coordinates.

WHY THIS IS A DIFFERENT METHOD FROM ``hang_newton_cable.py``.  Both run Newton,
but through completely different front doors, and the models are NOT the same:

  * here the connection is a UsdPhysics SPHERICAL joint, which Newton parses as
    a BALL joint -- three free rotational degrees of freedom and no bending
    resistance whatsoever. The cable is a perfectly floppy chain, EI = 0.
  * ``hang_newton_cable.py`` uses ``ModelBuilder.add_rod``, which creates
    dedicated CABLE joints carrying a real bend stiffness E I / h.

So this method should track the analytic catenary MORE closely than the rod
does, because the catenary is itself the EI = 0 idealisation. Agreeing with the
catenary is therefore not evidence that this method is more accurate -- both are
neglecting the same physics. That distinction is the single most useful thing
this pair of methods demonstrates, and it is why both are worth running.

The USD/engine route also pays the full Kit start-up cost, whereas the rod route
runs Newton headless in about a second. Comparing their wall times measures the
simulator's overhead, not the solver's.

-----------------------------------------------------------------------------
BUILD-SPECIFIC BEHAVIOUR -- measured on Isaac Sim 6.0.0-rc.59, not guessed.
Each of these was verified with a single free-falling capsule, whose exact
solution z(t) = z0 - g t^2 / 2 makes any deviation unambiguous.

1. The XPBD solver cannot be used with state readback. Assigning
   ``cfg.solver_cfg = XPBDSolverConfig(...)`` makes ``world.reset()`` die with
   "Failed to create simulation view with backend 'newton'", because the
   omni.physics.tensors view supports only the default MuJoCo solver here.
   This is almost certainly why the earlier ``hang_newton.py`` produced empty
   result directories: it requested XPBD, so the run never got as far as
   writing a CSV. This method therefore uses the DEFAULT solver, and
   ``--solver xpbd`` is accepted only with an explicit warning.

2. ``isaacsim.physics.newton.tensors`` must be enabled explicitly. Without it,
   ``world.reset()`` fails the same way, and the older timeline-based route
   silently does not step physics at all -- a free-falling body stays exactly
   at its initial height. That failure is especially dangerous for this
   benchmark, because a frozen cable sits precisely on the analytic catenary
   it was initialised with and scores a perfect 0.0 mm RMSE. The run therefore
   asserts afterwards that the cable actually moved.

3. USD transforms are never written back. The prim xform attributes stay at
   their authored values for the whole run; only the RigidPrim tensor view
   reflects the simulation. All state is read through that view.

4. The step size ignores every timing knob. ``physics_dt``, ``rendering_dt``
   and ``cfg.physics_frequency`` all have no measurable effect: the engine
   advances a fixed 1/499.6 s per ``world.step()`` (measured identically at
   physics_frequency 300, 600 and 1200). The benchmark's time base therefore
   comes from ``--step-dt``, defaulting to that calibrated value, and
   ``--calibrate`` re-measures it on the current build before running.

5. ``cfg.armature`` defaults to 0.1, an artificial joint inertia. Against
   capsule links of ~0.5 g it dominates the rotational dynamics completely, so
   it is set to 0 here.
-----------------------------------------------------------------------------

    python.sh hang_newton_engine.py --points 2 --out results/x
"""

from __future__ import annotations

import contextlib
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hang_common import method_parser, scenario_from_args  # noqa: E402

METHOD = "newton_engine"
LABEL = "Newton engine (USD + XPBD)"


# Seconds of simulated time advanced by one world.step() on this build,
# measured by free fall (see the module docstring, point 4).
CALIBRATED_STEP_DT = 1.0 / 499.6


def build_parser():
    p = method_parser("Isaac Sim 6 native Newton engine via UsdPhysics.")
    p.add_argument("--solver", choices=["default", "xpbd"], default="default",
                   help="'default' is the MuJoCo solver, the only one whose state "
                        "can be read back on this build; 'xpbd' will fail at reset")
    p.add_argument("--iterations", type=int, default=100, help="solver iterations")
    p.add_argument("--substeps", type=int, default=1, help="engine substeps per step")
    p.add_argument("--armature", type=float, default=0.0,
                   help="artificial joint inertia; the engine default of 0.1 swamps "
                        "a light cable")
    p.add_argument("--step-dt", type=float, default=CALIBRATED_STEP_DT,
                   help="simulated seconds per world.step() on this build")
    p.add_argument("--calibrate", action="store_true",
                   help="re-measure --step-dt by free fall before running")
    return p


args = build_parser().parse_args()
scenario = scenario_from_args(args)
headless = True if args.headless is None else args.headless

from isaacsim.simulation_app import SimulationApp  # noqa: E402

simulation_app = SimulationApp({"headless": headless})

import numpy as np  # noqa: E402
from isaacsim.core.api.world import World  # noqa: E402
from isaacsim.core.experimental.prims import RigidPrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

# The Newton engine lives in isaacsim.physics.newton, which is not enabled by
# default. Enable it before importing from it, or the import fails with
# "No module named 'isaacsim.physics'".
enable_extension("isaacsim.physics.newton")
# Required for state readback -- see the module docstring, point 2.
enable_extension("isaacsim.physics.newton.tensors")
simulation_app.update()

from isaacsim.physics.newton import XPBDSolverConfig, acquire_stage  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from cable_config import with_length  # noqa: E402
from isaac_env import isaac_version as _isaac_version_fn  # noqa: E402

isaac_version = _isaac_version_fn()
from hang_common import (  # noqa: E402
    SettleMonitor,
    initial_polyline,
    write_outputs,
)


@contextlib.contextmanager
def quiet_stderr():
    """Silence fd 2 briefly.

    Activating the Newton engine makes Isaac initialise a physics view on the
    current (still cable-less) stage, which logs a harmless transient failure
    about creating a simulation view with backend 'newton' before recovering on
    the real play below. Swallow exactly that, and nothing longer-lived.
    """
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def calibrate_step_dt(world, steps: int = 400) -> float:
    """Measure the simulated seconds advanced by one ``world.step()``.

    Drops a free body and inverts z(t) = z0 - g t^2 / 2. Needed because this
    build honours none of the timing settings (docstring point 4), so the only
    reliable way to know the time base is to measure it on the running engine.
    """
    from isaacsim.core.api.objects import DynamicSphere

    probe = world.scene.add(DynamicSphere(
        prim_path="/World/_dt_probe", name="_dt_probe",
        position=np.array([0.0, 5.0, 50.0]), radius=0.05, mass=1.0))
    world.reset()
    z0 = float(np.asarray(probe.get_world_pose()[0])[2])
    for _ in range(steps):
        world.step(render=False)
    drop = z0 - float(np.asarray(probe.get_world_pose()[0])[2])
    if drop <= 0.0:
        raise RuntimeError("calibration probe did not fall -- physics is not stepping")
    total_t = math.sqrt(2.0 * drop / 9.81)
    world.scene.remove_object("_dt_probe")
    return total_t / steps


def quat_z_to(direction: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) rotating +Z onto the unit vector `direction`."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, direction))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
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
    radius = cable.radius
    link_mass = cable.mass / num_links

    SimulationManager.switch_physics_engine("newton")

    # Step through World rather than timeline.play() + simulation_app.update().
    # The latter ticks the app and renders but does not reliably advance
    # physics headlessly: the cable then sits frozen at its initial pose and
    # every error metric comes out at exactly zero -- a "perfect" score that
    # actually means nothing was simulated. World.step() drives the physics
    # scene explicitly, so a frozen result is impossible to mistake for a good
    # one. The run is validated against that failure mode after the loop.
    render_dt = 1.0 / 60.0
    world = World(stage_units_in_meters=1.0,
                  physics_dt=render_dt / args.substeps,
                  rendering_dt=render_dt)
    # Start from an empty stage. Whatever the app template left behind gets
    # parsed by Newton alongside the cable, and stray prims are enough to make
    # the simulation view fail to build.
    world.clear()
    stage = world.stage

    # Configure the physics scene World already owns; do NOT define a second
    # one. World creates /physicsScene, and authoring another UsdPhysics.Scene
    # at /World/PhysicsScene leaves two on the stage -- Newton then cannot
    # resolve which to simulate and the simulation view fails to build. That
    # duplicate was the actual cause of the "Failed to create simulation view
    # with backend 'newton'" error here, not the joint topology.
    ctx = world.get_physics_context()
    ctx.set_gravity(-abs(scenario.gravity))

    def half_z(i: int) -> float:
        return max(seg_len[i] - 2.0 * radius, 1.0e-4) / 2.0 + radius

    # ArticulationRootAPI is MANDATORY on the Newton engine path. Bisected on
    # this build: a stage of free rigid bodies creates its simulation view
    # fine, but adding ANY joint -- spherical or fixed -- makes world.reset()
    # fail with "Failed to create simulation view with backend 'newton'"
    # unless the bodies and joints sit under a prim carrying this API. Newton
    # parses articulations, not loose joints, which is the same reason its own
    # ModelBuilder.add_rod defaults to wrap_in_articulation=True.
    cable_root = UsdGeom.Xform.Define(stage, "/World/cable")
    UsdPhysics.ArticulationRootAPI.Apply(cable_root.GetPrim())

    seg_paths, seg_quats = [], []
    for i in range(num_links):
        center = 0.5 * (pts[i] + pts[i + 1])
        q = quat_z_to(seg_dir[i])
        seg_quats.append(q)
        path = f"/World/cable/seg_{i:03d}"
        capsule = UsdGeom.Capsule.Define(stage, path)
        capsule.CreateRadiusAttr().Set(radius)
        capsule.CreateHeightAttr().Set(max(seg_len[i] - 2.0 * radius, 1.0e-4))
        capsule.CreateAxisAttr().Set("Z")
        xf = UsdGeom.Xformable(capsule.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in center]))
        xf.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in q]))

        UsdPhysics.RigidBodyAPI.Apply(capsule.GetPrim())
        # No CollisionAPI (the cable hangs in air and the supports are joints),
        # so mass and inertia have to be stated explicitly.
        mass_api = UsdPhysics.MassAPI.Apply(capsule.GetPrim())
        mass_api.CreateMassAttr().Set(link_mass)
        rot_inertia = link_mass * seg_len[i] ** 2 / 12.0
        mass_api.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(rot_inertia, rot_inertia, 0.5 * link_mass * radius ** 2))
        seg_paths.append(path)

    def spherical_joint(path, body0, body1, pos0, pos1):
        j = UsdPhysics.SphericalJoint.Define(stage, path)
        if body0 is not None:
            j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(*pos0))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(*pos1))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
        return j

    for i in range(num_links - 1):
        spherical_joint(f"/World/cable/joint_{i:03d}",
                        seg_paths[i], seg_paths[i + 1],
                        (0.0, 0.0, +half_z(i)), (0.0, 0.0, -half_z(i + 1)))

    # Supports are FIXED JOINTS TO THE WORLD, and the support bodies must NOT
    # be marked kinematic.
    #
    # Bisected on an exported stage: setting physics:kinematicEnabled on bodies
    # inside the articulation is exactly what makes the simulation view fail to
    # build. Clearing that one attribute on an otherwise identical stage turns
    # the failure into a clean reset. Anchoring with world welds instead is
    # accepted at every chain length tested (3 to 60 links), provided the
    # articulation root is present and no kinematic flags are set.
    #
    # Note this differs from the boundary condition used by the Newton rod and
    # Warp methods (zeroed / infinite mass). It is the same constraint
    # physically -- the support holds that body's pose -- but imposed through a
    # joint because this engine path rejects the mass route.
    clamped = [0, num_links - 1]
    for name, idx in (("fix_end_1", 0), ("fix_end_2", num_links - 1)):
        center = 0.5 * (pts[idx] + pts[idx + 1])
        q = seg_quats[idx]
        fj = UsdPhysics.FixedJoint.Define(stage, f"/World/cable/{name}")
        fj.CreateBody1Rel().SetTargets([Sdf.Path(seg_paths[idx])])
        fj.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in center]))
        fj.CreateLocalRot0Attr().Set(Gf.Quatf(*[float(v) for v in q]))
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    if scenario.mid_node is not None:
        # Middle support: hold the node, let the cable pivot through it.
        mid = scenario.mid_node
        clamped.append(mid)
        spherical_joint("/World/cable/mid_hold", None, seg_paths[mid],
                        tuple(float(v) for v in scenario.supports[1]),
                        (0.0, 0.0, -half_z(mid)))
    clamped = sorted(set(clamped))

    # Configure the Newton stage AFTER the engine switch and the scene build,
    # but before reset() parses it.
    newton_stage = acquire_stage()
    newton_stage.cfg.num_substeps = args.substeps
    newton_stage.cfg.collapse_fixed_joints = False  # keep the world welds intact
    newton_stage.cfg.time_step_app = False          # do not tie physics to app time
    newton_stage.cfg.armature = args.armature       # engine default 0.1 swamps a light cable
    if args.solver == "xpbd":
        print("[warn] --solver xpbd: state readback is broken for XPBD on this "
              "build; world.reset() is expected to fail. See the module docstring.")
        newton_stage.cfg.solver_cfg = XPBDSolverConfig(iterations=args.iterations)

    if os.environ.get("DUMP_STAGE"):
        stage.Export(os.environ["DUMP_STAGE"])
        print(f"[debug] exported stage to {os.environ['DUMP_STAGE']}")
    try:
        with quiet_stderr():
            world.reset()
            for _ in range(5):  # warm-up
                world.step(render=False)
    except Exception as exc:
        raise RuntimeError(
            "Newton's USD/engine path cannot represent this cable on Isaac Sim "
            f"{isaac_version}.\n\n"
            f"  world.reset() failed: {exc}\n\n"
            "Diagnosis (bisected on an exported stage, see the module docstring):\n"
            "  * The bodies and joints MUST sit under a prim with "
            "UsdPhysics.ArticulationRootAPI -- loose joints are rejected outright.\n"
            "  * An articulation must be a TREE. A cable held at both ends is a "
            "closed loop: with one world weld the view builds, with two it fails.\n"
            "  * Marking the second anchor excludeFromArticulation lets the view "
            "build but the engine then IGNORES that constraint -- the cable hangs "
            "from one end only (min_z 0.18 m instead of 0.73 m).\n"
            "  * Anchoring by physics:kinematicEnabled instead also fails: that "
            "attribute on a body inside an articulation is itself what breaks the "
            "simulation view.\n\n"
            "There is therefore no working configuration for a both-ends-fixed "
            "cable through this route on this build. Use hang_newton_cable.py, "
            "which drives the SAME Newton solver through ModelBuilder.add_rod and "
            "supports both boundary conditions natively.") from exc

    active = SimulationManager.get_active_physics_engine()
    solver_name = type(newton_stage.cfg.solver_cfg).__name__
    print(f"[newton-engine] active engine: {active}, solver: {solver_name}, "
          f"substeps={args.substeps}, armature={args.armature}, {num_links} links")
    if active != "newton":
        print(f"[warn] expected the newton engine but got '{active}' -- "
              "results would describe the wrong solver")

    view = RigidPrim(seg_paths)

    def read_nodes() -> np.ndarray:
        positions, orientations = view.get_world_poses()
        p = positions.numpy().reshape(-1, 3)
        q = orientations.numpy().reshape(-1, 4)  # (w, x, y, z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        axis = np.column_stack([2.0 * (x * z + y * w),
                                2.0 * (y * z - x * w),
                                1.0 - 2.0 * (x * x + y * y)])
        half = np.array([half_z(i) for i in range(num_links)])
        nodes = np.empty((num_links + 1, 3))
        nodes[:-1] = p - half[:, None] * axis
        nodes[-1] = p[-1] + half[-1] * axis[-1]
        return nodes

    step_dt = args.step_dt
    if args.calibrate:
        step_dt = calibrate_step_dt(world)
        print(f"[newton-engine] calibrated step dt = 1/{1 / step_dt:.1f} s "
              f"(default was 1/{1 / CALIBRATED_STEP_DT:.1f})")

    info = {
        "engine": active,
        "solver": solver_name,
        "iterations": args.iterations,
        "substeps": args.substeps,
        "armature": args.armature,
        "num_links": num_links,
        "joint_type": "UsdPhysics.SphericalJoint (ball, EI=0)",
        "clamped_bodies": clamped,
        "step_dt_s": step_dt,
        "step_dt_calibrated": bool(args.calibrate),
        "average_window_s": args.average_window,
    }

    monitor = SettleMonitor(scenario)
    start_nodes = read_nodes()
    monitor.update(0.0, start_nodes)
    sim_time = 0.0
    steps_per_sample = max(1, int(round(0.1 / step_dt)))  # sample every ~0.1 s
    try:
        while simulation_app.is_running() and sim_time < scenario.max_time:
            for _ in range(steps_per_sample):
                world.step(render=False)
            sim_time += steps_per_sample * step_dt
            nodes = read_nodes()
            settled = monitor.update(sim_time, nodes)
            print(monitor.report(sim_time, nodes))
            if settled:
                print("  settled.")
                break

        # Guard against the silent-failure mode: if nothing ever moved, the
        # cable was never simulated and the "perfect" agreement with the
        # analytic catenary is an artefact of the initial condition, not a
        # result. Fail loudly rather than write a beautiful, meaningless CSV.
        moved = float(np.abs(read_nodes() - start_nodes).max())
        if moved < 1.0e-9:
            raise RuntimeError(
                f"the cable never moved (max displacement {moved:.2e} m over "
                f"{sim_time:.1f} s of sim time). Physics did not step, so this "
                "run reports the initial catenary rather than a simulation.")
        print(f"[newton-engine] max displacement from the initial pose: "
              f"{moved * 1e3:.2f} mm")

        write_outputs(args.out, scenario, monitor,
                      monitor.equilibrium_nodes(args.average_window),
                      METHOD, LABEL, extra=info)
    finally:
        simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
