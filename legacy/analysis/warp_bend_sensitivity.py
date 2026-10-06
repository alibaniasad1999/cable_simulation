#!/usr/bin/env python3
"""
Warp rod: accuracy versus assumed bending stiffness.

    <isaac python> analysis/warp_bend_sensitivity.py --scenario <run.json> \
        --out results/<run>/warp_bend.png

WHY THIS EXISTS
---------------
The Warp XPBD rod is the one method here that imposes a TRUE position-only pin at
each support (the others weld the orientation too). Given that freedom, its
settled shape depends on the bending stiffness, which is the only input this
project cannot measure: the material moduli are a stand-in.

Parametrised by the elasto-gravitational length l = (EI/w)^(1/3), and swept at
two mesh resolutions, this shows two things at once:

  * accuracy degrades as the assumed stiffness rises. The stand-in material sits
    at l ~ 0.15 m and is the worst; the cable's visible ~2 cm rounding scale puts
    it at l ~ 0.02 m, where the rod drapes far closer to the catenary. This is
    read from the rounding, NOT tuned to the catenary (it lands well off it).

  * the coarse mesh additionally BUCKLES at high stiffness -- the slack rod's
    draped state goes unstable and it arches above the supports. Refining the
    mesh removes the buckle, leaving a milder under-sag: so the dramatic buckle
    was partly a discretisation artefact, while the stiffness sensitivity is not.

Either way the remedy is the same: the floppy, drape-consistent stiffness at an
adequate resolution.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Bending lengths to probe [m]. Spans the placeholder (0.15) down to nearly rigid
# catenary agreement (0.005).
LENGTHS = [0.005, 0.008, 0.012, 0.02, 0.03, 0.05, 0.08, 0.12, 0.15]
# Two mesh resolutions, so the figure shows both the stiffness sensitivity and
# the resolution-dependence of the buckle.
RESOLUTIONS = [60, 150]
PLACEHOLDER_L = 0.15   # the stand-in material EI
DRAPE_L = 0.02         # implied by the cable's ~2 cm rounding scale


def run_one(isaac_py, scenario, ell, tmp):
    out = os.path.join(tmp, f"l_{ell:.4f}")
    os.makedirs(out, exist_ok=True)
    cmd = [isaac_py, os.path.join(ROOT, "methods", "hang_warp.py"),
           "--scenario", scenario, "--out", out, "--bend-length", str(ell)]
    subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    meta = os.path.join(out, "meta.json")
    prof = os.path.join(out, "profile.csv")
    if not os.path.isfile(meta):
        return None
    with open(meta) as fh:
        m = json.load(fh)
    # Buckled = the rod rises meaningfully above the supports somewhere.
    import csv
    zmax = -1e9
    height = m["scenario"]["height"]
    with open(prof) as fh:
        for r in csv.DictReader(fh):
            zmax = max(zmax, float(r["z_m"]))
    buckled = zmax > height + 0.01
    return {"l": ell, "rmse_mm": m["rmse_mm"], "sag_mm": m["sag_m"] * 1e3,
            "buckled": buckled, "zmax_above_mm": (zmax - height) * 1e3}


def scenario_at(scenario, segments, tmp):
    """Write a copy of the scenario with a given segment count."""
    with open(scenario) as fh:
        s = json.load(fh)
    s["num_segments"] = segments
    path = os.path.join(tmp, f"run_{segments}.json")
    with open(path, "w") as fh:
        json.dump(s, fh)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--tmp", default="/tmp/warp_bend_sweep")
    args = ap.parse_args()

    isaac_py = sys.executable
    os.makedirs(args.tmp, exist_ok=True)

    series = {}   # segments -> list of rows
    for seg in RESOLUTIONS:
        scen = scenario_at(args.scenario, seg, args.tmp)
        rows = []
        print(f"resolution {seg} segments:")
        for ell in LENGTHS:
            r = run_one(isaac_py, scen, ell, os.path.join(args.tmp, str(seg)))
            if r:
                rows.append(r)
                print(f"  l={ell*1e3:6.1f} mm  RMSE={r['rmse_mm']:7.1f} mm  "
                      f"{'BUCKLED' if r['buckled'] else 'draped'} "
                      f"(peak {r['zmax_above_mm']:+.0f} mm vs supports)")
        series[seg] = rows
    if not any(series.values()):
        print("no runs succeeded", file=sys.stderr)
        return 1

    with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
        json.dump({str(k): v for k, v in series.items()}, fh, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    line_colors = {60: "0.55", 150: "#0072B2"}
    for seg, rows in series.items():
        if not rows:
            continue
        lc = line_colors.get(seg, "0.4")
        ax.plot([r["l"] * 1e3 for r in rows], [r["rmse_mm"] for r in rows],
                "-", color=lc, lw=1.6, zorder=3, label=f"{seg} segments")
        # Buckled points get a distinct orange ring so the regime is visible.
        for r in rows:
            if r["buckled"]:
                ax.plot(r["l"] * 1e3, r["rmse_mm"], "o", ms=11, zorder=6,
                        mfc="#D55E00", mec="#D55E00")
            else:
                ax.plot(r["l"] * 1e3, r["rmse_mm"], "o", ms=7, zorder=5,
                        mfc=lc, mec=lc)

    ax.axvline(DRAPE_L * 1e3, color="#009E73", ls="--", lw=1.4)
    ax.axvline(PLACEHOLDER_L * 1e3, color="#D55E00", ls="--", lw=1.4)
    top = ax.get_ylim()[1]
    ax.text(DRAPE_L * 1e3, top * 0.93, " drape scale\n $\\ell\\approx$20 mm",
            color="#009E73", fontsize=8, va="top")
    ax.text(PLACEHOLDER_L * 1e3, top * 0.55, "stand-in\nmaterial\n$\\ell\\approx$150 mm ",
            color="#D55E00", fontsize=8, va="top", ha="right")

    handles = [Line2D([0], [0], color=line_colors.get(s, "0.4"), marker="o",
                      label=f"{s} segments") for s in series if series[s]]
    handles.append(Line2D([0], [0], marker="o", ls="", mfc="#D55E00",
                          mec="#D55E00", label="buckled (arch above supports)"))
    ax.legend(handles=handles, fontsize=9, loc="upper left")

    ax.set_xscale("log")
    ax.set_xlabel("assumed bending length  $\\ell = (EI/w)^{1/3}$  [mm]")
    ax.set_ylabel("RMSE vs analytic catenary  [mm]")
    ax.set_title("Warp rod: bending-stiffness sensitivity, at two mesh resolutions")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"[out] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
