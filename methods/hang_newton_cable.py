"""
Cable-hang benchmark -- NEWTON CABLE method (the headline).

Uses Newton's NATIVE cable/rod API -- ``ModelBuilder.add_rod`` -- which builds a
chain of capsule bodies joined by dedicated *cable joints*: one linear (stretch)
and one angular (bend/twist) degree of freedom per joint, i.e. a discrete
Cosserat rod. That is a genuinely different model from every other method here:

  * it carries real bend stiffness EI (a spherical-joint chain carries none),
  * it is a 1-D rod, so it does not suffer the FEM element aspect-ratio problem
    that forces the PhysX deformable to fatten the cable,
  * stiffness is set from the material moduli directly, via
    ``E A / L_seg`` and ``E I / L_seg`` -- the same formulas as
    ``newton._src.utils.cable.create_cable_stiffness_from_elastic_moduli``.

Integrated with ``SolverVBD`` (Vertex Block Descent): implicit, unconditionally
stable at large stiffness, which is what makes the stiff-cable regime tractable.

This runs Newton STANDALONE -- Isaac Sim's bundled Newton 1.2.0 and Warp are put
on the path by ``isaac_env.bootstrap_newton()``, but Kit is never booted. There
is no renderer, no USD stage and no window, so a run takes about a second. That
is the point of separating this from ``hang_newton_engine.py``, which drives the
SAME Newton physics through Isaac Sim's USD/engine layer and therefore pays the
full simulator start-up cost.

BOUNDARY CONDITIONS -- and a warning about the 3-support case.  Supports are
imposed by zeroing the mass and inertia of the capsule bodies at those nodes,
which CLAMPS them (position and orientation both held) rather than pinning them.
Because the cable is initialised exactly on the analytic catenary, the clamped
orientation is the catenary's own tangent, so the clamp is consistent with the
reference rather than fighting it.

For the two END supports that is uncontroversial, and every method here does the
equivalent. For the MIDDLE support it is not: the analytic solution has a slope
kink there, and clamping both adjacent capsules imposes that kink BY
CONSTRUCTION. This method then reproduces the catenary to ~2 mm in the
3-support case while the pinned methods (Warp rod, PhysX capsule chain) land at
26-38 mm -- a gap that measures the boundary condition, NOT solver quality.
Use ``--mid-support pin`` to compare the three on equal terms. See
``clamp_indices`` for the full argument.

    python.sh hang_newton_cable.py --length 1.0 --span 0.8 --points 2 --out results/x
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Optional Isaac Sim viewport.
#
# This method's physics is Newton, NOT Isaac Sim's engine -- Isaac Sim 6 cannot
# represent a both-ends-fixed cable through USD (see hang_newton_engine.py). The
# viewport is therefore render-only: Newton solves, Isaac Sim draws. That is how
# "Newton in the Isaac Sim GUI" is achieved here, and because the prims carry no
# physics, enabling it cannot change a benchmark number.
#
# SimulationApp must be constructed before any pxr/omni import AND before Warp,
# so that in GUI mode Warp comes from Isaac's bundle and shares its CUDA
# context. That has to happen before argparse runs, hence the raw argv peek --
# --gui itself is declared properly in method_parser().
# ---------------------------------------------------------------------------
_GUI = "--gui" in sys.argv
SIMULATION_APP = None
if _GUI:
    from isaacsim.simulation_app import SimulationApp

    SIMULATION_APP = SimulationApp({"headless": False})

import numpy as np

import isaac_env

WARP_VERSION, NEWTON_VERSION = isaac_env.bootstrap_newton()

import warp as wp  # noqa: E402  (must follow bootstrap)
import newton  # noqa: E402

from cable_config import CABLE, with_length, rod_stiffness  # noqa: E402
from cable_view import CableView  # noqa: E402
from hang_common import (  # noqa: E402
    SettleMonitor,
    initial_polyline,
    method_parser,
    scenario_from_args,
    tape_polyline,
    write_outputs,
)

METHOD = "newton_cable"
LABEL = "Newton cable (add_rod + VBD)"


def build_model(scenario, cable, args):
    """Assemble the Newton model: one rod, clamped at the support nodes."""
    # The 'tape' middle support starts the cable on a flat horizontal plateau so
    # the rod's bending stiffness rounds the top, instead of the point-pin's
    # imposed kink. tape_k is the number of plateau segments each side of the
    # middle node, used below to clamp exactly the flat bodies.
    tape_k = None
    if args.mid_support == "tape" and scenario.mid_node is not None:
        pts, tape_k = tape_polyline(scenario, args.tape_halfwidth)
    else:
        pts = initial_polyline(scenario)
    seg_len = float(np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
    stretch_k, bend_k = rod_stiffness(cable, seg_len)

    builder = newton.ModelBuilder()
    cfg = builder.default_shape_cfg.copy()
    # Solid cylinder of the given radius whose total mass matches the real
    # cable: Newton derives body mass from shape volume x density, so feeding
    # the true material density reproduces the true mass automatically.
    cfg.density = cable.mass / (cable.area * cable.length)
    cfg.mu = 0.0                  # nothing to rub against: the cable hangs in air
    cfg.collision_group = 0       # disable self-collision; a hanging cable never
                                  # touches itself, and enabling it costs a lot

    positions = [wp.vec3(float(p[0]), float(p[1]), float(p[2])) for p in pts]
    bodies, joints = builder.add_rod(
        positions=positions,
        radius=cable.radius,
        cfg=cfg,
        stretch_stiffness=stretch_k * args.stretch_scale,
        stretch_damping=args.stretch_damping,
        bend_stiffness=bend_k * args.bend_scale,
        bend_damping=args.bend_damping,
        label="cable",
    )

    if tape_k is not None:
        # Hold the whole flat plateau (the strip of tape) plus the two ends.
        m = scenario.mid_node
        clamped = sorted(set([0, len(bodies) - 1] + list(range(m - tape_k, m + tape_k))))
    else:
        clamped = clamp_indices(scenario, len(bodies), args.mid_support)
    for b in clamped:
        builder.body_mass[b] = 0.0
        builder.body_inv_mass[b] = 0.0
        builder.body_inertia[b] = wp.mat33(0.0)
        builder.body_inv_inertia[b] = wp.mat33(0.0)

    builder.color(balance_colors=False)
    model = builder.finalize(device=wp.get_device(args.device) if args.device else None)
    model.set_gravity((0.0, 0.0, -scenario.gravity))

    info = {
        "newton_version": NEWTON_VERSION,
        "warp_version": WARP_VERSION,
        "device": str(model.device),
        "segment_length_m": seg_len,
        "stretch_stiffness_N_per_m": stretch_k * args.stretch_scale,
        "bend_stiffness_N_m": bend_k * args.bend_scale,
        "stretch_damping": args.stretch_damping,
        "bend_damping": args.bend_damping,
        "num_bodies": len(bodies),
        "num_joints": len(joints),
        "clamped_bodies": clamped,
        "mid_support": args.mid_support,
        "tape_halfwidth_m": args.tape_halfwidth if tape_k is not None else None,
        "tape_segments_each_side": tape_k,
        "vbd_iterations": args.iterations,
        "substeps": args.substeps,
        "density_kg_m3": cfg.density,
    }
    return model, bodies, pts, info


def clamp_indices(scenario, num_bodies: int, mid_support: str = "clamp") -> list[int]:
    """Body indices held fixed at the supports.

    Body ``i`` is the capsule spanning node ``i`` to node ``i+1``, so the first
    and last bodies carry the end supports.

    The middle support (3-support case) admits two different physical models,
    and the choice DOMINATES the 3-support error -- far more than solver
    quality does, so it must be stated rather than assumed:

      ``clamp``  hold BOTH capsules meeting at the mid node. Two equal-height
                 catenaries meet there with opposite slopes, so the analytic
                 solution has a kink; clamping both sides imposes that kink by
                 construction. This reproduces the catenary almost exactly --
                 but partly because it is being told the answer. Physically it
                 is a cable gripped in a fixture.

      ``pin``    hold only ONE of the two capsules, leaving the other free to
                 pivot about the shared joint.

    MEASURED: the two modes give the SAME answer -- 3-support, L=1 m, S=0.8 m,
    both produce RMSE 2.1 mm and arc drift +1.44 %. Dropping the second clamped
    body changes nothing, because holding ONE capsule's orientation already
    pins the tangent on that side and thereby imposes the kink.

    The conclusion is an API limitation worth stating plainly: ``add_rod``
    exposes capsule BODIES, not nodes, and every way of holding a body also
    holds its orientation. There is therefore no way to express a TRUE pinned
    interior support -- position held, both tangents free -- without adding a
    loop-closing joint to the world. The Warp rod can (a node with inverse mass
    0 constrains position only), and it lands at 39 mm rather than 2 mm on the
    same problem.

    So the large 3-support gap between this method and the pinned ones measures
    the boundary condition, not solver quality, and this method cannot currently
    be brought onto equal terms with them. Compare 3-support numbers across
    methods only with that in mind; the 2-support case has no such ambiguity.
    """
    clamped = [0, num_bodies - 1]
    mid = scenario.mid_node
    if mid is not None:
        clamped += [mid - 1, mid] if mid_support == "clamp" else [mid]
    return sorted(set(int(b) for b in clamped))


def read_nodes(state, bodies, half_len) -> np.ndarray:
    """Recover the N+1 centreline nodes from the N capsule body transforms.

    Each capsule's local +Z runs along the segment, so its two endpoints are
    ``centre -/+ half_len * z_axis``. Node i is body i's start; the final node
    is the last body's end.
    """
    q = state.body_q.numpy()[bodies]
    pos = q[:, 0:3]
    qx, qy, qz, qw = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    zx = 2.0 * (qx * qz + qy * qw)
    zy = 2.0 * (qy * qz - qx * qw)
    zz = 1.0 - 2.0 * (qx * qx + qy * qy)
    axis = np.column_stack([zx, zy, zz])

    nodes = np.empty((len(bodies) + 1, 3))
    nodes[:-1] = pos - half_len[:, None] * axis
    nodes[-1] = pos[-1] + half_len[-1] * axis[-1]
    return nodes


def main() -> int:
    p = method_parser(__doc__.strip().splitlines()[1])
    # VBD is iterative, so the iteration count directly controls how nearly
    # inextensible the cable is. Measured on the 2-point case (L=1, S=0.8):
    #   20 iters -> arc +1.85%, RMSE 9.9 mm
    #  100 iters -> arc +0.40%, RMSE  2.4 mm
    #  200 iters -> arc +0.22%, RMSE  1.5 mm   <- default: the knee
    #  400 iters -> arc +0.12%, RMSE  2.0 mm   (2.6x the cost, no better)
    # Anything below ~100 reports a cable that visibly stretches under its own
    # weight, which is a solver artefact and not a property of the model.
    p.add_argument("--iterations", type=int, default=200,
                   help="VBD solver iterations per substep")
    p.add_argument("--substeps", type=int, default=8, help="substeps per 1/60 s frame")
    p.add_argument("--bend-scale", type=float, default=1.0,
                   help="multiplier on the beam-theory bend stiffness E*I/L")
    p.add_argument("--stretch-scale", type=float, default=1.0,
                   help="multiplier on the beam-theory stretch stiffness E*A/L")
    p.add_argument("--bend-damping", type=float, default=1.0e-4)
    p.add_argument("--stretch-damping", type=float, default=1.0e-2)
    p.add_argument("--mid-support", choices=["clamp", "pin", "tape"], default="clamp",
                   help="3-support case: 'clamp' holds both capsules at the middle "
                        "node (imposes the catenary kink); 'pin' holds one, letting "
                        "the cable pivot -- matches the other methods' boundary "
                        "condition; 'tape' holds a short flat plateau, so the cable "
                        "leaves the support horizontally and bending rounds the top "
                        "instead of cornering -- this is what a strip of tape does")
    p.add_argument("--tape-halfwidth", type=float, default=0.012,
                   help="'tape' mid-support only: half-width [m] of the flat "
                        "plateau at the middle support (default 12 mm, i.e. a "
                        "~24 mm strip of tape)")
    p.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu")
    p.add_argument("--view-style", choices=["tube", "capsules"], default="tube",
                   help="--gui rendering: 'tube' draws one continuous cable (how "
                        "Newton's own viewer shows it); 'capsules' draws each rod "
                        "segment separately, which makes the discretisation visible")
    args = p.parse_args()

    scenario = scenario_from_args(args)
    cable = with_length(scenario.length)
    print(scenario.describe())
    print(cable.summary())

    model, bodies, pts, info = build_model(scenario, cable, args)
    print(f"[newton-cable] Newton {NEWTON_VERSION}, Warp {WARP_VERSION}, "
          f"device {info['device']}")
    print(f"[newton-cable] {info['num_bodies']} capsules, "
          f"stretch {info['stretch_stiffness_N_per_m']:.3e} N/m, "
          f"bend {info['bend_stiffness_N_m']:.3e} N*m, "
          f"clamped bodies {info['clamped_bodies']}")

    solver = newton.solvers.SolverVBD(model, iterations=args.iterations)
    state_0, state_1 = model.state(), model.state()
    control, contacts = model.control(), model.contacts()

    half_len = 0.5 * np.linalg.norm(np.diff(pts, axis=0), axis=1)

    fps = 60.0
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / args.substeps
    monitor = SettleMonitor(scenario)

    def simulate_frame():
        nonlocal state_0, state_1
        for _ in range(args.substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    # Render-only mirror of the cable in the Isaac Sim stage (no-op without --gui).
    view = CableView(
        SIMULATION_APP,
        len(bodies) + 1,
        cable.radius,
        label="cable_newton",
        supports=scenario.supports,
        style=args.view_style,
    )

    sim_time = 0.0
    monitor.update(0.0, read_nodes(state_0, bodies, half_len))
    view.sync(read_nodes(state_0, bodies, half_len))
    while sim_time < scenario.max_time:
        for _ in range(int(round(fps * 0.1))):  # sample every 0.1 s
            simulate_frame()
            sim_time += frame_dt
            if view.enabled:  # animate every frame, not every sample
                view.sync(read_nodes(state_0, bodies, half_len))
        nodes = read_nodes(state_0, bodies, half_len)
        settled = monitor.update(sim_time, nodes)
        print(monitor.report(sim_time, nodes))
        if settled:
            print("  settled.")
            break

    nodes = monitor.equilibrium_nodes(args.average_window)
    info["average_window_s"] = args.average_window
    # For 3 supports the middle boundary condition is the whole point of the
    # comparison, so it goes in the label; the two Newton runs sit side by side.
    label = LABEL
    if scenario.num_points == 3:
        label = ("Newton cable - tape (rounded)" if args.mid_support == "tape"
                 else "Newton cable - point pin (kink)")
    write_outputs(args.out, scenario, monitor, nodes, METHOD, label, extra=info)

    if view.enabled:
        # Show the reported equilibrium shape, then wait: without this the window
        # would vanish the instant the cable settles, which reads as a crash.
        view.sync(nodes)
        view.hold_open(nodes, "\n[newton-cable] settled -- close the Isaac Sim window to exit.")
        SIMULATION_APP.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
