"""Does the simulated cable bend like its EI? A cantilever test with the JSON's cable settings.

    python 03_franka_holds_cable/check_stiffness.py --config configs/ethernet_cat6.json
    python 03_franka_holds_cable/check_stiffness.py --config configs/ethernet_cat6.json --bend-scale 0.2 1 4

A straight piece of the cable is clamped horizontally (like holding it out over a
table edge) and sags under its own weight. Beam theory gives the tip drop:

    bending   d_b = w L^4 / (8 EI)          w = mass per metre x g
    shear     d_s = w L^2 / (2 kGA)         (small; the model allows a little shear)

The simulation uses exactly the JSON's segment length, rigidities, mass, damping,
iterations and substeps, so this tells you whether the cable in ethernet_scene.py
behaves like the EI you typed. Run it whenever you change those numbers.

Why this exists: with stretch/shear rigidities far above the bending one
(EA = 2e4 N, kGA = 7e3 N) VBD made the cable bend ~10x too easily, and more
iterations did not help. Keeping EA and kGA moderate fixes it.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin" and os.environ.pop("DYLD_LIBRARY_PATH", None):
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

REPO = Path(__file__).resolve().parents[1]
G = 9.81


def load_config(path):
    with open(path) as f:
        cfg = json.load(f)

    def clean(d):
        return {k: clean(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")}
    return clean(cfg)


def cable_radius(cfg):
    r = cfg["cable"].get("radius_m")
    if r is not None:
        return float(r)
    summary = cfg["scan"].get("summary_json")
    path = Path(summary) if summary and Path(summary).is_absolute() else REPO / (summary or "")
    if summary and path.exists():
        return json.loads(path.read_text())["radius_mm"] / 1000.0
    print("radius: cable.radius_m is null and summary.json not found, using 3.6 mm")
    return 0.0036


def cantilever(cfg, r, EI, GJ, L_free, device):
    """Simulated tip drop [m] of a horizontal cantilever of free length L_free."""
    cab, sim = cfg["cable"], cfg["sim"]
    h = float(sim["segment_length_m"])
    n_clamp = 2
    n_seg = n_clamp + max(2, round(L_free / h))
    L_free = (n_seg - n_clamp) * h

    b = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    SolverVBD.register_custom_attributes(b)
    rod = newton.Rod.create_straight(
        start=wp.vec3(0.0, 0.0, 1.0), direction=wp.vec3(1.0, 0.0, 0.0), length=n_seg * h, segment_count=n_seg,
        radius=r, stretch_rigidity=cab["stretch_rigidity_EA_N"], shear_rigidity=cab["shear_rigidity_kGA_N"],
        bend_rigidity=EI, twist_rigidity=GJ)
    V = np.pi * r * r * h + 4.0 / 3.0 * np.pi * r ** 3  # capsule: cylinder + two half spheres
    tau = cab["bend_damping_time_s"]
    bodies, _ = b.add_rod(rod=rod, cfg=newton.ModelBuilder.ShapeConfig(density=cab["mass_per_length_kg_m"] * h / V),
                          body_frame_origin="com", bend_damping=tau * EI / h, twist_damping=tau * GJ / h,
                          stretch_damping=0.0)
    for i in bodies[:n_clamp]:
        b.body_mass[i] = 0.0
        b.body_inv_mass[i] = 0.0
        b.body_inertia[i] = wp.mat33(0.0)
        b.body_inv_inertia[i] = wp.mat33(0.0)
    b.color()
    m = b.finalize(device=device)

    s0, s1, control = m.state(), m.state(), m.control()
    solver = SolverVBD(m, iterations=int(sim["iterations"]), rigid_compliant_alm=True, rigid_contact_history=False)
    fps, substeps = sim["fps"], int(sim["substeps"])
    dt = 1.0 / fps / substeps

    def tip_z():
        q = s0.body_q.numpy()[bodies[-1]]
        qx, qy, qz, qw = q[3:7]
        return q[2] + 0.5 * h * 2.0 * (qy * qz - qx * qw)  # centre + half a segment along local +Z

    prev = tip_z()
    for frame in range(int(10 * fps)):
        for _ in range(substeps):
            s0.clear_forces()
            solver.step(s0, s1, control, None, dt)
            s0, s1 = s1, s0
        if (frame + 1) % (fps // 2) == 0:  # every 0.5 s
            z = tip_z()
            if not np.isfinite(z):
                raise SystemExit("cantilever blew up (NaN)")
            if abs(z - prev) < 1e-5:
                break
            prev = z
    return 1.0 - tip_z(), L_free


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--length", type=float, default=0.2, help="free length sticking out [m] (keep the drop small)")
    ap.add_argument("--bend-scale", type=float, nargs="+", default=[1.0], help="multipliers on EI (and GJ)")
    args = ap.parse_args()

    wp.config.quiet = True
    cfg = load_config(args.config)
    cab = cfg["cable"]
    r = cable_radius(cfg)
    w = cab["mass_per_length_kg_m"] * G
    print(f"segment {1000 * cfg['sim']['segment_length_m']:.0f} mm, EA {cab['stretch_rigidity_EA_N']:g} N, "
          f"kGA {cab['shear_rigidity_kGA_N']:g} N, {cfg['sim']['iterations']} iterations x {cfg['sim']['substeps']} "
          f"substeps, free length {1000 * args.length:.0f} mm")
    for k in args.bend_scale:
        EI, GJ = cab["bend_rigidity_EI_Nm2"] * k, cab["twist_rigidity_GJ_Nm2"] * k
        t0 = time.time()
        drop, L = cantilever(cfg, r, EI, GJ, args.length, cfg["sim"].get("device"))
        d_b = w * L ** 4 / (8.0 * EI)
        d_s = w * L ** 2 / (2.0 * cab["shear_rigidity_kGA_N"])
        ratio = drop / (d_b + d_s)
        verdict = "OK" if abs(ratio - 1.0) < 0.15 else "WRONG: the cable does not bend like its EI"
        print(f"EI {EI:.3g} N m^2: simulated drop {1000 * drop:6.1f} mm, theory {1000 * (d_b + d_s):6.1f} mm "
              f"(bending {1000 * d_b:.1f} + shear {1000 * d_s:.1f})  ->  sim/theory {ratio:4.2f}  {verdict}"
              f"   [{time.time() - t0:.0f} s]")
        if d_b > 0.15 * L:
            print(f"   note: drop is {100 * d_b / L:.0f} % of the length; beam theory is only exact for small drops, "
                  "use a shorter --length")


if __name__ == "__main__":
    main()
