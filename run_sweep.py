#!/usr/bin/env python3
"""
Hyperparameter sweep for the Newton cable solver.

WHY THIS EXISTS
---------------
``run_benchmark.py`` answers "which METHOD is most accurate?". This script
answers the question underneath it: "is this method being run with sensible
SOLVER SETTINGS?" -- which has to be settled first, because a badly configured
solver loses to a well configured one for reasons that have nothing to do with
the physics being compared.

Two specific discrepancies against upstream Newton motivate it:

1. SOLVER BUDGET. Every cable example shipped with Newton
   (example_cable_pile / _twist / _y_junction / _bundle_hysteresis) uses
   ``sim_substeps = 10`` and ``sim_iterations = 5`` -- 50 iteration-steps per
   frame. This repository's default is 8 substeps x 200 iterations = 1600, a
   32x larger budget. The existing default was tuned by varying iterations
   ALONE at fixed substeps, so it never tested whether spending that budget on
   substeps instead would be cheaper, more accurate, or both. For an implicit
   solver like VBD those two axes are not interchangeable, so the question is
   empirical.

2. DAMPING. Newton's examples run bend_damping/bend_stiffness ratios of
   0.05 to 1.0. With this cable's beam-theory stiffness (EI/L = 6.03e-2 N m)
   the repository default of bend_damping = 1e-4 is a ratio of 1.7e-3 --
   between 30x and 600x less damped than any Newton example. That is the
   prime suspect for runs that finish with ``"settled": false`` after a swing
   that is still visibly decaying at t = 30 s.

WHAT IT MEASURES
----------------
Accuracy is scored against the ANALYTIC CATENARY, not against another solver,
so a configuration cannot look good merely by agreeing with its neighbours.
Cost is wall-clock. Settling is reported too, because a configuration that is
accurate only after 30 s of simulated swinging is not equivalent to one that
is accurate at 3 s -- that difference is invisible in a static-shape-only
comparison.

    # both sweeps, the full picture (this is the one for the report)
    python3 run_sweep.py --sweep both --length 1.0 --span 0.8 --points 2

    # just the budget question, 3-support case
    python3 run_sweep.py --sweep budget --points 3

    # see the grid without running it
    python3 run_sweep.py --sweep both --dry-run

Results land in ``results/sweep_<timestamp>/``:
    sweep.csv        one row per configuration, every parameter and metric
    budget.png       accuracy vs cost, with Newton's and this repo's defaults marked
    damping.png      accuracy and settling vs damping ratio
    sweep_report.tex the write-up
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "methods"))

from hang_common import Scenario, add_scenario_args  # noqa: E402
import isaac_env  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

# Upstream Newton's setting, identical across all four of its cable examples.
NEWTON_SUBSTEPS, NEWTON_ITERATIONS = 10, 5
# This repository's current default (methods/hang_newton_cable.py).
REPO_SUBSTEPS, REPO_ITERATIONS = 8, 200

# Budget grid. Spans Newton's 50 iteration-steps/frame up to well past this
# repo's 1600, so the accuracy-vs-cost knee is bracketed on both sides rather
# than extrapolated.
BUDGET_SUBSTEPS = [4, 8, 10, 16]
BUDGET_ITERATIONS = [5, 10, 25, 50, 100, 200]

# Damping grids.
#
# A first pass swept bend_damping over 1e-4 .. 6e-2 (a 600x range, spanning this
# repo's default up to Newton's ratio) and the result did not move: RMSE varied
# by 0.006% and arc drift by 0.001%. That is a real finding, not a broken run --
# the values were verified to reach the solver -- and it has a physical reason.
# This cable's EI is 1.0e-3 N m^2, small enough that the ZERO-bending analytic
# catenary matches the simulation to about a millimetre, so the bending mode
# carries almost no energy and damping it changes almost nothing.
#
# What the same pass did show is arc drift of +5.37% at Newton's 50-step budget.
# The cable should stretch about 0.05% under its own weight (T/EA with
# EA = 1005 N), so a hundredfold more than that is an UNDER-CONVERGED STRETCH
# CONSTRAINT, not physical extension. Stretch is where the energy is, so that
# is what the damping arm now sweeps.
#
# bend_damping is kept as a two-point NEGATIVE CONTROL. Reporting a knob that
# provably does not matter is worth the two runs: it stops the next person
# tuning it, and it makes the stretch result harder to dismiss as luck.
STRETCH_DAMPING_VALUES = [0.0, 1.0e-4, 1.0e-2, 1.0e-1]  # 0.0 = Newton's default
BEND_DAMPING_CONTROL = [1.0e-4, 6.0e-2]  # this repo's default, and 600x it


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_args(p)
    p.add_argument("--sweep", choices=["budget", "damping", "both"], default="both",
                   help="which sweep to run")
    p.add_argument("--out", default=None,
                   help="output directory (default: results/sweep_<timestamp>)")
    p.add_argument("--isaac-python", default=None,
                   help="path to Isaac Sim's python.sh (auto-detected)")
    p.add_argument("--timeout", type=int, default=1200,
                   help="wall-clock cap per configuration [s]")
    p.add_argument("--dry-run", action="store_true",
                   help="list the configurations without running them")
    p.add_argument("--no-report", action="store_true",
                   help="write sweep.csv but skip plots and the LaTeX report")
    return p


def configurations(args) -> list[dict]:
    """Build the list of runs. Each is a dict of the parameters that vary."""
    runs: list[dict] = []

    if args.sweep in ("budget", "both"):
        for substeps in BUDGET_SUBSTEPS:
            for iterations in BUDGET_ITERATIONS:
                runs.append({
                    "sweep": "budget",
                    "substeps": substeps,
                    "iterations": iterations,
                    "bend_damping": 1.0e-4,   # repo defaults, held fixed here
                    "stretch_damping": 1.0e-2,
                })

    if args.sweep in ("damping", "both"):
        # All damping runs sit at Newton's budget, so a damping effect cannot be
        # confounded with a solver-effort effect.
        for stretch_damping in STRETCH_DAMPING_VALUES:
            runs.append({
                "sweep": "stretch_damping",
                "substeps": NEWTON_SUBSTEPS,
                "iterations": NEWTON_ITERATIONS,
                "bend_damping": 1.0e-4,
                "stretch_damping": stretch_damping,
            })
        for bend_damping in BEND_DAMPING_CONTROL:
            runs.append({
                "sweep": "bend_damping_control",
                "substeps": NEWTON_SUBSTEPS,
                "iterations": NEWTON_ITERATIONS,
                "bend_damping": bend_damping,
                "stretch_damping": 1.0e-2,
            })

    # Deduplicate: the arms can generate the same parameter tuple twice. Keep
    # the first, recording every sweep it belongs to, so a shared point runs
    # ONCE but is still plotted on every chart it belongs to.
    unique: dict[tuple, dict] = {}
    for run in runs:
        key = (run["substeps"], run["iterations"], run["bend_damping"],
               run["stretch_damping"])
        if key in unique:
            unique[key]["sweep"] += "+" + run["sweep"]
        else:
            unique[key] = dict(run)
    return list(unique.values())


def tag_for(run: dict) -> str:
    # Two mantissa digits, not zero: 1.5e-2 is stored as 0.01499..., so a
    # ".0e" format renders it "1e-02" -- both misleading as a directory name
    # and liable to collide with a genuine 1.0e-2 entry.
    return (f"s{run['substeps']}_i{run['iterations']}"
            f"_bd{run['bend_damping']:.2e}_sd{run['stretch_damping']:.2e}")


def run_one(run: dict, scenario_path: str, out_root: str, isaac_py: str,
            timeout: int) -> dict:
    """Run one configuration and harvest its metrics. Never raises."""
    tag = tag_for(run)
    out_dir = os.path.join(out_root, tag)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        isaac_py, os.path.join(ROOT, "methods", "hang_newton_cable.py"),
        "--scenario", scenario_path,
        "--out", out_dir,
        "--substeps", str(run["substeps"]),
        "--iterations", str(run["iterations"]),
        "--bend-damping", str(run["bend_damping"]),
        "--stretch-damping", str(run["stretch_damping"]),
    ]

    t0 = time.perf_counter()
    log_path = os.path.join(out_dir, "run.log")
    with open(log_path, "w") as log:
        try:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=timeout, cwd=ROOT)
            status = "ok" if rc == 0 else f"exit {rc}"
        except subprocess.TimeoutExpired:
            status = f"timeout>{timeout}s"
    wall = time.perf_counter() - t0

    row = dict(run)
    row.update({"tag": tag, "status": status, "wall_s": round(wall, 2),
                "budget": run["substeps"] * run["iterations"]})

    meta_path = os.path.join(out_dir, "meta.json")
    if not os.path.isfile(meta_path):
        row.update({"rmse_mm": None, "arc_drift_pct": None, "sag_error_mm": None,
                    "max_abs_error_mm": None, "settled": None,
                    "settled_time_s": None, "final_max_speed_mps": None,
                    "damping_ratio": None})
        return row

    with open(meta_path) as fh:
        meta = json.load(fh)
    bend_k = meta["solver"]["bend_stiffness_N_m"]
    row.update({
        "rmse_mm": meta["rmse_mm"],
        "arc_drift_pct": meta["arc_drift_pct"],
        "sag_error_mm": meta["sag_error_mm"],
        "max_abs_error_mm": meta["max_abs_error_mm"],
        "settled": meta["settled"],
        "settled_time_s": meta["settled_time_s"],
        "final_max_speed_mps": meta["final_max_speed_mps"],
        "bend_stiffness_N_m": bend_k,
        # The dimensionless number that is actually comparable to Newton's
        # examples -- bend_damping alone means nothing without its stiffness.
        "damping_ratio": run["bend_damping"] / bend_k if bend_k else None,
    })
    return row


FIELDS = ["tag", "sweep", "substeps", "iterations", "budget", "bend_damping",
          "stretch_damping", "damping_ratio", "bend_stiffness_N_m", "rmse_mm",
          "max_abs_error_mm", "sag_error_mm", "arc_drift_pct", "settled",
          "settled_time_s", "final_max_speed_mps", "wall_s", "status"]


def main() -> int:
    args = build_parser().parse_args()
    runs = configurations(args)

    scenario = Scenario(
        length=args.length, span=args.span, height=args.height,
        num_points=args.num_points, num_segments=args.num_segments,
        max_time=args.max_time, settle_vel=args.settle_vel)

    out_root = args.out or os.path.join(
        ROOT, "results", f"sweep_{time.strftime('%Y%m%d_%H%M%S')}_{scenario.tag}")
    os.makedirs(out_root, exist_ok=True)
    scenario_path = os.path.join(out_root, "run.json")
    scenario.save(scenario_path)

    print("=" * 78)
    print(scenario.describe())
    print(f"{len(runs)} configurations | sweep: {args.sweep}")
    print(f"reference points: newton {NEWTON_SUBSTEPS}x{NEWTON_ITERATIONS}="
          f"{NEWTON_SUBSTEPS * NEWTON_ITERATIONS}, "
          f"this repo {REPO_SUBSTEPS}x{REPO_ITERATIONS}={REPO_SUBSTEPS * REPO_ITERATIONS}")
    print(f"output: {out_root}")
    print("=" * 78)

    if args.dry_run:
        for run in runs:
            print(f"  {tag_for(run):<28} sweep={run['sweep']:<14} "
                  f"budget={run['substeps'] * run['iterations']}")
        return 0

    isaac_py = args.isaac_python or os.path.join(isaac_env.find_isaac_sim(), "python.sh")
    if not os.path.isfile(isaac_py):
        print(f"!! Isaac Sim python not found at {isaac_py}")
        return 2

    csv_path = os.path.join(out_root, "sweep.csv")
    rows: list[dict] = []
    for i, run in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {tag_for(run)}", flush=True)
        row = run_one(run, scenario_path, out_root, isaac_py, args.timeout)
        rows.append(row)

        if row["rmse_mm"] is None:
            print(f"    {row['status']}  ({row['wall_s']:.1f} s)  NO RESULT")
        else:
            settled = "settled" if row["settled"] else "unsettled"
            print(f"    RMSE {row['rmse_mm']:6.2f} mm | arc {row['arc_drift_pct']:+6.2f}% "
                  f"| {settled:9s} | {row['wall_s']:6.1f} s")

        # Rewrite after every run: a sweep interrupted at config 17 of 30 still
        # leaves a usable CSV rather than nothing.
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

    print(f"\n[out] {csv_path}")

    ok = [r for r in rows if r["rmse_mm"] is not None]
    if ok:
        best = min(ok, key=lambda r: r["rmse_mm"])
        cheap = min((r for r in ok if r["rmse_mm"] < 2.0 * best["rmse_mm"]),
                    key=lambda r: r["wall_s"], default=None)
        print(f"\n  most accurate : {best['tag']}  "
              f"RMSE {best['rmse_mm']:.2f} mm in {best['wall_s']:.1f} s")
        if cheap:
            print(f"  best value    : {cheap['tag']}  "
                  f"RMSE {cheap['rmse_mm']:.2f} mm in {cheap['wall_s']:.1f} s "
                  f"(within 2x of best RMSE, cheapest)")

    if args.no_report:
        return 0

    print(f"\n>>> report")
    rc = subprocess.call([isaac_py, os.path.join(ROOT, "analysis", "sweep_report.py"),
                          "--sweep-dir", out_root], cwd=ROOT)
    if rc != 0:
        print("[warn] report generation failed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
