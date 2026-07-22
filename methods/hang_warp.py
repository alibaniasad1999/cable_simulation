"""
Cable-hang benchmark -- WARP XPBD ROD method (solver-independent reference).

A position-based (XPBD) elastic rod written directly in NVIDIA Warp: particles
on the centreline, a distance constraint per segment (stretch) and a
midpoint-deviation constraint per interior node (bend). The kernels are the ones
from the original ``cable_warp.py``, carried over intact.

Its role in the benchmark is to be the method that owes NOTHING to Isaac Sim or
Newton: no engine, no USD, no ModelBuilder, no Kit. If the Isaac-side methods
and this one agree, the agreement is not an artefact of a shared code path. It
also runs on CPU, so it still works on a machine with no usable GPU.

COMPLIANCE FROM PHYSICS, NOT FROM FITTING.  XPBD compliance is the inverse of a
stiffness, so both are derived from the cable's own moduli rather than tuned:

    stretch:  k = E A / h          ->  alpha = h / (E A)          [m/N]
    bend:     k = 8 E I / h^3      ->  alpha = h^3 / (8 E I)      [m/N]

The bend figure comes from equating this constraint's energy to Euler-Bernoulli
bending energy. The midpoint-deviation constraint measures the sagitta C of the
arc through three consecutive nodes; for a circular arc of curvature kappa over
a chord 2h the sagitta is C ~ kappa h^2 / 2, so kappa ~ 2C/h^2. Substituting
into U = (1/2) EI kappa^2 (2h) gives U = 4 EI C^2 / h^3, and matching that to
the constraint form U = (1/2) k C^2 yields k = 8 EI / h^3.

The original script instead used a hand-tuned bend compliance (10.0) chosen to
make the profile sit on the analytic catenary. That is circular for a benchmark
whose whole purpose is measuring the gap to the catenary: fitting the parameter
to the reference guarantees a good score and measures nothing. The physical
value is used here, and ``--bend-compliance`` is left exposed for deliberate
sensitivity studies.

    python hang_warp.py --length 1.0 --span 0.8 --points 2 --out results/x
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

try:
    import warp as wp
except ImportError:  # fall back to Isaac Sim's bundled Warp
    import isaac_env

    isaac_env.bootstrap_newton()
    import warp as wp

from cable_config import CABLE, with_length  # noqa: E402
from hang_common import (  # noqa: E402
    SettleMonitor,
    initial_polyline,
    method_parser,
    scenario_from_args,
    tape_polyline,
    write_outputs,
)

METHOD = "warp_rod"
LABEL = "Warp XPBD rod"


# ---------------------------------------------------------------------------
# Kernels (from the original cable_warp.py)
# ---------------------------------------------------------------------------
@wp.kernel
def predict(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invm: wp.array(dtype=float),
    g: float,
    dt: float,
):
    i = wp.tid()
    x_prev[i] = x[i]
    if invm[i] > 0.0:
        vel = v[i] + wp.vec3(0.0, 0.0, g) * dt
        v[i] = vel
        x[i] = x[i] + vel * dt


@wp.kernel
def solve_rod(
    x: wp.array(dtype=wp.vec3),
    invm: wp.array(dtype=float),
    rest: wp.array(dtype=float),
    n: int,
    iters: int,
    a_stretch: float,
    a_bend: float,
):
    # Gauss-Seidel over the whole rod: sequential by construction, so it runs
    # in a single thread. The rod is short enough that this is not the
    # bottleneck, and it keeps the constraint ordering identical to the
    # original implementation.
    if wp.tid() != 0:
        return
    for _it in range(iters):
        for s in range(n - 1):
            xa = x[s]
            xb = x[s + 1]
            wa = invm[s]
            wb = invm[s + 1]
            d = xa - xb
            ln = wp.length(d)
            if ln > 1.0e-9 and (wa + wb) > 0.0:
                nrm = d / ln
                C = ln - rest[s]
                dl = -C / (wa + wb + a_stretch)
                x[s] = xa + wa * dl * nrm
                x[s + 1] = xb - wb * dl * nrm
        for i in range(1, n - 1):
            xa = x[i - 1]
            xb = x[i]
            xc = x[i + 1]
            wa = invm[i - 1]
            wb = invm[i]
            wc = invm[i + 1]
            mid = 0.5 * (xa + xc)
            d = xb - mid
            ln = wp.length(d)
            denom = wb + 0.25 * wa + 0.25 * wc + a_bend
            if ln > 1.0e-9 and denom > 0.0:
                nrm = d / ln
                dl = -ln / denom
                x[i - 1] = xa - 0.5 * wa * dl * nrm
                x[i] = xb + wb * dl * nrm
                x[i + 1] = xc - 0.5 * wc * dl * nrm


@wp.kernel
def finalize(
    x: wp.array(dtype=wp.vec3),
    x_prev: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    invm: wp.array(dtype=float),
    dt: float,
    damp: float,
):
    i = wp.tid()
    if invm[i] > 0.0:
        v[i] = (x[i] - x_prev[i]) / dt * (1.0 - damp)


def main() -> int:
    p = method_parser("Warp XPBD elastic rod cable hang.")
    p.add_argument("--substeps", type=int, default=16, help="substeps per 1/60 s frame")
    p.add_argument("--iterations", type=int, default=12,
                   help="Gauss-Seidel constraint iterations per substep")
    p.add_argument("--damping", type=float, default=0.02,
                   help="velocity damping per substep [0..1]")
    p.add_argument("--bend-compliance", type=float, default=None,
                   help="override the derived bend compliance h^3/(8 E I) [m/N]")
    p.add_argument("--bend-length", type=float, default=None,
                   help="set bending stiffness from the elasto-gravitational length "
                        "l=(EI/w)^(1/3) [m] instead of the material EI: EI=w*l^3. "
                        "The placeholder material gives l~0.15 m, at which a slack "
                        "rod's draped state is unstable and buckles; the real cable "
                        "drapes with rounding confined to ~2 cm, i.e. l~0.02 m.")
    p.add_argument("--stretch-compliance", type=float, default=None,
                   help="override the derived stretch compliance h/(E A) [m/N]")
    p.add_argument("--device", default=None, help="warp device, e.g. cuda:0 or cpu")
    p.add_argument("--init", choices=["catenary", "tape"], default="catenary",
                   help="initial shape (3-support only): 'catenary' starts on the "
                        "analytic solution, which KINKS at the middle support and can "
                        "seed an upward buckle; 'tape' starts on the smooth flat-tape "
                        "shape, draped and kink-free")
    args = p.parse_args()

    scenario = scenario_from_args(args)
    cable = with_length(scenario.length)
    print(scenario.describe())
    print(cable.summary())

    wp.init()
    device = args.device or ("cuda:0" if wp.is_cuda_available() else "cpu")

    if args.init == "tape" and scenario.mid_node is not None:
        pts, _ = tape_polyline(scenario, 0.012)
    else:
        pts = initial_polyline(scenario)
    n = len(pts)
    rest = np.linalg.norm(np.diff(pts, axis=0), axis=1).astype(np.float32)
    h = float(rest.mean())

    alpha_s = (args.stretch_compliance
               if args.stretch_compliance is not None
               else h / cable.axial_stiffness)
    # Bending compliance, in order of precedence: explicit compliance, then a
    # physical bending length l (EI = w l^3), then the material EI. The bending
    # length is the honest control here: it is read from the cable's visible
    # rounding scale, not tuned to sit on the catenary.
    if args.bend_compliance is not None:
        alpha_b = args.bend_compliance
    elif args.bend_length is not None:
        w = (cable.mass / cable.length) * scenario.gravity          # weight/length [N/m]
        ei_eff = w * args.bend_length ** 3                          # [N m^2]
        alpha_b = h ** 3 / (8.0 * ei_eff)
    else:
        alpha_b = h ** 3 / (8.0 * cable.bending_stiffness)

    # Held nodes get infinite mass (inverse mass 0).
    m_node = cable.mass / n
    invm = np.full(n, 1.0 / m_node, dtype=np.float32)
    invm[0] = 0.0
    invm[-1] = 0.0
    if scenario.mid_node is not None:
        invm[scenario.mid_node] = 0.0

    x_wp = wp.array(pts.astype(np.float32), dtype=wp.vec3, device=device)
    xprev_wp = wp.zeros(n, dtype=wp.vec3, device=device)
    v_wp = wp.zeros(n, dtype=wp.vec3, device=device)
    invm_wp = wp.array(invm, dtype=float, device=device)
    rest_wp = wp.array(rest, dtype=float, device=device)

    frame_dt = 1.0 / 60.0
    dt = frame_dt / args.substeps
    a_s = alpha_s / (dt * dt)
    a_b = alpha_b / (dt * dt)

    info = {
        "warp_version": wp.config.version,
        "device": device,
        "num_nodes": n,
        "segment_length_m": h,
        "stretch_compliance_m_per_N": alpha_s,
        "bend_compliance_m_per_N": alpha_b,
        "bend_length_m": args.bend_length,
        "substeps": args.substeps,
        "iterations": args.iterations,
        "damping": args.damping,
    }
    print(f"[warp] {n} nodes on {device}, h={h * 1e3:.2f} mm, "
          f"alpha_stretch={alpha_s:.3e}, alpha_bend={alpha_b:.3e} m/N")

    monitor = SettleMonitor(scenario)
    monitor.update(0.0, pts)

    sim_time = 0.0
    sample_every = 6  # frames -> 0.1 s
    frame = 0
    while sim_time < scenario.max_time:
        for _ in range(args.substeps):
            wp.launch(predict, dim=n,
                      inputs=[x_wp, xprev_wp, v_wp, invm_wp, -scenario.gravity, dt],
                      device=device)
            wp.launch(solve_rod, dim=1,
                      inputs=[x_wp, invm_wp, rest_wp, n, args.iterations, a_s, a_b],
                      device=device)
            wp.launch(finalize, dim=n,
                      inputs=[x_wp, xprev_wp, v_wp, invm_wp, dt, args.damping],
                      device=device)
        sim_time += frame_dt
        frame += 1
        if frame % sample_every == 0:
            nodes = x_wp.numpy().astype(float)
            settled = monitor.update(sim_time, nodes)
            print(monitor.report(sim_time, nodes))
            if settled:
                print("  settled.")
                break

    nodes = monitor.equilibrium_nodes(args.average_window)
    info["average_window_s"] = args.average_window
    write_outputs(args.out, scenario, monitor, nodes, METHOD, LABEL, extra=info)
    return 0


if __name__ == "__main__":
    sys.exit(main())
