#!/usr/bin/env python3
"""
Cable-hang benchmark driver.

Runs every simulation method on ONE shared scenario, then compares them all
against the analytic catenary and against each other.

    # 2 supports, 1 m cable over an 0.8 m span
    python3 run_benchmark.py --length 1.0 --span 0.8 --points 2

    # 3 supports, and only the two Newton methods
    python3 run_benchmark.py --points 3 --methods newton_cable newton_engine

    # everything, both scenarios
    python3 run_benchmark.py --all

The scenario is written ONCE to ``results/<run>/run.json`` and handed to every
method with ``--scenario``. No method re-derives the geometry, so they provably
solve the same problem; a method that disagrees is disagreeing about physics,
not about its inputs.

Each method runs as a separate subprocess under Isaac Sim's ``python.sh``. That
is not optional bookkeeping: the PhysX methods force the PhysX engine and the
Newton ones force Newton, and Isaac Sim cannot switch engines cleanly inside a
single process. Separate processes also mean a method that crashes or hangs
costs only its own result -- the remaining methods still run, and the failure is
recorded in the metrics table rather than taking down the benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "methods"))

from hang_common import Scenario, add_scenario_args  # noqa: E402
import isaac_env  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))

# method -> (script, default per-job wall-clock cap in seconds)
METHODS = {
    "newton_cable": ("hang_newton_cable.py", 1800),
    "newton_engine": ("hang_newton_engine.py", 1800),
    "physx_capsule": ("hang_physx_capsule.py", 1800),
    "physx_fem": ("hang_physx_fem.py", 3600),
    "warp_rod": ("hang_warp.py", 900),
}

# Presentation order: Newton family, then pure Isaac Sim / PhysX, then the
# solver-independent reference.
ORDER = ["newton_cable", "newton_engine", "physx_capsule", "physx_fem", "warp_rod"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_scenario_args(p)
    p.add_argument("--methods", nargs="+", default=ORDER, choices=ORDER,
                   help="which methods to run")
    p.add_argument("--all", action="store_true",
                   help="run BOTH the 2-support and 3-support scenarios")
    p.add_argument("--out", default=None,
                   help="output directory (default: results/<timestamp>_<tag>)")
    p.add_argument("--isaac-python", default=None,
                   help="path to Isaac Sim's python.sh (auto-detected)")
    p.add_argument("--timeout", type=int, default=None,
                   help="override the per-method wall-clock cap [s]")
    p.add_argument("--no-analysis", action="store_true",
                   help="skip the comparison/report step")
    p.add_argument("--dry-run", action="store_true",
                   help="print the commands without running them")
    return p


def find_isaac_python(explicit: str | None) -> str:
    if explicit:
        return explicit
    return os.path.join(isaac_env.find_isaac_sim(), "python.sh")


def run_scenario(scenario: Scenario, args, isaac_py: str) -> str:
    """Run every requested method on one scenario. Returns the output dir."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join(ROOT, "results", f"{stamp}_{scenario.tag}")
    if args.all and args.out:  # keep the two scenarios apart under one --out
        out_dir = os.path.join(args.out, scenario.tag)
    os.makedirs(out_dir, exist_ok=True)

    scenario_path = os.path.join(out_dir, "run.json")
    scenario.save(scenario_path)

    sol = scenario.catenary()
    print("=" * 78)
    print(scenario.describe())
    print(sol.summary())
    print(f"output: {out_dir}")
    print("=" * 78)

    summary = {}
    for method in args.methods:
        script, default_timeout = METHODS[method]
        timeout = args.timeout or default_timeout
        method_dir = os.path.join(out_dir, method)
        os.makedirs(method_dir, exist_ok=True)
        cmd = [isaac_py, os.path.join(ROOT, "methods", script),
               "--scenario", scenario_path, "--out", method_dir]

        print(f"\n>>> {method}  ({script}, cap {timeout} s)")
        if args.dry_run:
            print("    " + " ".join(cmd))
            continue

        log_path = os.path.join(method_dir, "run.log")
        t0 = time.perf_counter()
        with open(log_path, "w") as log:
            try:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout, cwd=ROOT)
                status = "ok" if rc == 0 else f"exit {rc}"
            except subprocess.TimeoutExpired:
                rc, status = -1, f"TIMEOUT after {timeout} s"
        wall = time.perf_counter() - t0

        produced = os.path.isfile(os.path.join(method_dir, "profile.csv"))
        if produced:
            with open(os.path.join(method_dir, "meta.json")) as fh:
                meta = json.load(fh)
            print(f"    {status} in {wall:.1f} s -> RMSE {meta['rmse_mm']:.1f} mm, "
                  f"arc {meta['arc_drift_pct']:+.2f}%, "
                  f"{'settled' if meta['settled'] else 'not settled'}")
        else:
            print(f"    {status} in {wall:.1f} s -> NO RESULT (see {log_path})")
            _tail(log_path)
        summary[method] = {"status": status, "wall_s": round(wall, 1),
                           "produced": produced}

    with open(os.path.join(out_dir, "run_summary.json"), "w") as fh:
        json.dump({"scenario": asdict(scenario), "methods": summary}, fh, indent=2)
    return out_dir


def _tail(path: str, n: int = 12) -> None:
    """Show the end of a failed run's log, so failures are visible immediately."""
    try:
        with open(path, errors="replace") as fh:
            lines = [l.rstrip() for l in fh if l.strip()]
    except OSError:
        return
    for line in lines[-n:]:
        print(f"      | {line[:160]}")


def main() -> int:
    args = build_parser().parse_args()
    isaac_py = find_isaac_python(args.isaac_python)
    if not os.path.isfile(isaac_py):
        print(f"!! Isaac Sim python not found at {isaac_py}")
        print("   Set --isaac-python /path/to/isaacsim/python.sh "
              "or ISAAC_SIM_PATH=/path/to/isaacsim")
        return 2
    print(f"Isaac Sim: {isaac_env.find_isaac_sim()} "
          f"({isaac_env.isaac_version()})")

    point_counts = [2, 3] if args.all else [args.num_points]
    out_dirs = []
    for n in point_counts:
        scenario = Scenario(
            length=args.length, span=args.span, height=args.height,
            num_points=n, num_segments=args.num_segments,
            max_time=args.max_time, settle_vel=args.settle_vel)
        out_dirs.append(run_scenario(scenario, args, isaac_py))

    if args.dry_run or args.no_analysis:
        return 0

    # Run the analysis under Isaac Sim's interpreter, not the system one. The
    # system python may pair a numpy 2.x with a matplotlib built against 1.x,
    # which imports far enough to look fine and then dies with "_ARRAY_API not
    # found" the moment a figure is drawn -- so the tables appear but the plots
    # silently do not. Isaac's bundled pair is known-consistent.
    for out_dir in out_dirs:
        print(f"\n>>> analysis: {out_dir}")
        subprocess.call([isaac_py, os.path.join(ROOT, "analysis", "compare.py"),
                         "--run", out_dir], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
