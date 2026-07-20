#!/usr/bin/env python3
"""
Run one experiment end to end, from a config file.

    python3 run_experiment.py --config experiments/apple_cable_3pt.yaml

This is the top-level entry point. It produces the three-way comparison the
project exists for:

    MATH        the governing ODE, integrated numerically (methods/cable_ode.py)
    REAL        the cable in the photograph (image_utils/measure_length.py)
    SIMULATION  each requested solver (methods/hang_*.py)

WHY ALL THREE
-------------
Any two of them can agree for uninteresting reasons. Simulation matching the
analytic curve only shows the solver reproduces a model, not that the model
describes the cable on the door. Simulation matching the photograph, with no
analytic reference, cannot separate "the solver is right" from "two errors
cancelled". With the third leg present, a disagreement can be attributed:
against MATH it is numerical error, against REAL it is modelling error, and the
two are different problems with different fixes.

WHAT IT WRITES  (into results/<experiment name>_<timestamp>/)
    config.resolved.yaml   the experiment as actually run, measurements filled in
    provenance.json        which numbers were measured, which were assumed
    run.json               the Scenario handed to every solver
    real_profile.csv       the cable extracted from the photograph
    detection.png          the detection figure -- check this before believing anything
    <method>/              one directory per solver, as run_benchmark.py produces
    metrics.csv, overlay.png, report.tex   from analysis/compare.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "methods"))

from experiment import Experiment, resolve_from_image  # noqa: E402
import isaac_env  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="experiment YAML/JSON file")
    p.add_argument("--out", default=None,
                   help="output directory (default: results/<name>_<timestamp>)")
    p.add_argument("--isaac-python", default=None, help="path to Isaac Sim's python.sh")
    p.add_argument("--timeout", type=int, default=1800, help="per-method wall-clock cap [s]")
    p.add_argument("--skip-sim", action="store_true",
                   help="measure and compute the math model, but run no solvers")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve the config and print the plan without running anything")
    return p


def main() -> int:
    args = build_parser().parse_args()
    exp = Experiment.load(args.config)

    print("=" * 78)
    print(exp.describe())
    print("=" * 78)

    isaac_py = args.isaac_python or os.path.join(isaac_env.find_isaac_sim(), "python.sh")
    if not os.path.isfile(isaac_py):
        print(f"!! Isaac Sim python not found at {isaac_py}", file=sys.stderr)
        return 2

    out_dir = args.out or os.path.join(
        ROOT, "results", f"{exp.name}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 1. REAL: measure the photograph ---------------------------------
    print("\n>>> 1/4  measuring the photograph")
    exp, prov = resolve_from_image(exp)
    with open(os.path.join(out_dir, "provenance.json"), "w") as fh:
        json.dump(prov, fh, indent=2)
    for k, v in prov.items():
        extra = ""
        if k == "supports.mid_fraction" and v["source"] == "measured":
            extra = (f"  (chord-proportional would be {v['chord_proportional']:.4f}, "
                     f"{v['difference_pp']:+.1f} pp)")
        print(f"    {k:24s} {v['source']:9s} = {v['value']}{extra}")

    try:
        import yaml
        with open(os.path.join(out_dir, "config.resolved.yaml"), "w") as fh:
            yaml.safe_dump(exp.to_dict(), fh, sort_keys=False)
    except Exception as exc:
        print(f"    [warn] could not write resolved config: {exc}")

    if exp.image.path:
        real_csv = os.path.join(out_dir, "real_profile.csv")
        cmd = [isaac_py, os.path.join(ROOT, "image_utils", "measure_length.py"),
               "--image", exp.image.path,
               "--supports", *[str(x) for x in exp.supports.x_m],
               "--height", str(exp.supports.height_m),
               "--sat-max", str(exp.image.sat_max), "--val-min", str(exp.image.val_min),
               "--step", str(exp.image.step_px), "--halfwidth", str(exp.image.halfwidth_px),
               "--profile-csv", real_csv,
               "--figure", os.path.join(out_dir, "detection.png")]
        if args.dry_run:
            print("    " + " ".join(cmd))
        else:
            subprocess.call(cmd, cwd=ROOT, stdout=subprocess.DEVNULL)
            print(f"    [out] {real_csv}")
            print(f"    [out] {os.path.join(out_dir, 'detection.png')}")
    else:
        real_csv = None

    # ---- 2. MATH: integrate the governing ODE ----------------------------
    print("\n>>> 2/4  mathematical model (ODE, integrated numerically)")
    cmd = [isaac_py, os.path.join(ROOT, "methods", "cable_ode.py"),
           "--span", str(exp.span_m), "--length", str(exp.cable.length_m),
           "--height", str(exp.supports.height_m), "--compare"]
    if args.dry_run:
        print("    " + " ".join(cmd))
    else:
        with open(os.path.join(out_dir, "math_model.txt"), "w") as fh:
            subprocess.call(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.DEVNULL)
        with open(os.path.join(out_dir, "math_model.txt")) as fh:
            for line in fh:
                if any(k in line for k in ("solved a", "sag", "arc-length check", "AGREE", "DISAGREE")):
                    print("    " + line.rstrip())

    # ---- 3. SIMULATION ---------------------------------------------------
    from hang_common import Scenario

    scenario = Scenario(
        length=exp.cable.length_m, span=exp.span_m, height=exp.supports.height_m,
        num_points=exp.num_points, num_segments=exp.simulation.segments,
        max_time=exp.simulation.max_time, settle_vel=exp.simulation.settle_vel,
        mid_x=exp.mid_x)
    scenario_path = os.path.join(out_dir, "run.json")
    scenario.save(scenario_path)
    print(f"\n>>> 3/4  simulation  ({len(exp.simulation.methods)} methods)")
    print("    " + scenario.describe().replace("\n", "\n    "))

    method_scripts = {
        "newton_cable": "hang_newton_cable.py",
        "newton_engine": "hang_newton_engine.py",
        "physx_capsule": "hang_physx_capsule.py",
        "physx_fem": "hang_physx_fem.py",
        "warp_rod": "hang_warp.py",
    }
    produced = 0
    for method in exp.simulation.methods:
        if method not in method_scripts:
            print(f"    [skip] unknown method {method!r}")
            continue
        mdir = os.path.join(out_dir, method)
        os.makedirs(mdir, exist_ok=True)
        cmd = [isaac_py, os.path.join(ROOT, "methods", method_scripts[method]),
               "--scenario", scenario_path, "--out", mdir]
        # Solver knobs are only accepted by the Newton cable method; the others
        # derive their own from the shared scenario.
        if method == "newton_cable":
            cmd += ["--substeps", str(exp.simulation.substeps),
                    "--iterations", str(exp.simulation.iterations),
                    "--stretch-damping", str(exp.simulation.stretch_damping),
                    "--bend-damping", str(exp.simulation.bend_damping),
                    "--mid-support", exp.simulation.mid_support]
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue
        if args.skip_sim:
            continue
        print(f"    >>> {method} ...", flush=True)
        t0 = time.perf_counter()
        with open(os.path.join(mdir, "run.log"), "w") as log:
            try:
                subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                timeout=args.timeout)
            except subprocess.TimeoutExpired:
                pass
        dt = time.perf_counter() - t0
        meta = os.path.join(mdir, "meta.json")
        if os.path.isfile(meta):
            with open(meta) as fh:
                m = json.load(fh)
            produced += 1
            print(f"        RMSE {m['rmse_mm']:.2f} mm | arc {m['arc_drift_pct']:+.2f}% | "
                  f"{'settled' if m['settled'] else 'unsettled'} | {dt:.1f} s")
        else:
            print(f"        NO RESULT after {dt:.1f} s -- see {mdir}/run.log")

    if args.dry_run:
        print("\n(dry run: nothing executed)")
        return 0

    # ---- 4. COMPARE ------------------------------------------------------
    if args.skip_sim or produced == 0:
        print("\n>>> 4/4  comparison skipped (no simulation results)")
        print(f"\nDONE.  {out_dir}")
        return 0

    print("\n>>> 4/4  comparison")
    cmd = [isaac_py, os.path.join(ROOT, "analysis", "compare.py"), "--run", out_dir]
    if real_csv and os.path.isfile(real_csv):
        cmd += ["--real", real_csv]
    subprocess.call(cmd, cwd=ROOT)

    print(f"\nDONE.  {out_dir}")
    print(f"  detection figure : {os.path.join(out_dir, 'detection.png')}")
    print(f"  report           : {os.path.join(out_dir, 'report.tex')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
