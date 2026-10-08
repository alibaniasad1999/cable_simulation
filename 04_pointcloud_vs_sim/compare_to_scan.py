"""Compare simulated Ethernet-cable centrelines with the scanned one.

    # from the repository root, after 03_franka_holds_cable/ethernet_scene.py
    python 04_pointcloud_vs_sim/compare_to_scan.py --config configs/ethernet_cat6.json

Reads every run under <output_dir>/<init>/ (one per bend_scale), compares each
with the tube-fit centreline, and writes into <output_dir>/<init>/compare/:

    metrics.csv     one row per run (all distances in mm)
    summary.md      the same as a readable table, best run marked
    overlay.png     scan vs simulation: top view and two side views
    errors.png      distance to the scan along the cable
    sweep.png       error vs bending stiffness (only with 2+ runs)

Needs only numpy and matplotlib (no Newton), so it runs anywhere.

Distances (all on the scanned stretch, s = 0 .. scan length, skipping nodes the
tube fit marked unsupported):

    shape       distance from each scan node to the nearest point of the sim
                curve. "Is the sim cable where the real one is?" Insensitive
                to the cable sliding along itself.
    arc         distance between points at the SAME arc length s. Also sees
                sliding and length errors.
    hanging /   the shape error split at the table: nodes more than 3 mm above
    lying       "lying on the table" height count as hanging.
    plug end    arc distance at the last scanned node.
    moved       how far the sim cable moved from its starting shape (for
                init = 'scan' this is how far the scan is from an equilibrium).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# Reference palette (dataviz skill): ink, gridline, surface, categorical, blue ramp.
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # light -> dark, validated ordinal


# =============================================================================
# Loading
# =============================================================================
def load_config(path):
    with open(path) as f:
        cfg = json.load(f)

    def clean(d):
        return {k: clean(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")}
    return clean(cfg)


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def arclength(X):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])


def load_scan(csv_path, gripper_end, scan_to_world):
    """Scan centreline in the simulation's world frame (up = +z), gripper end first."""
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    X = np.column_stack([d["x_m"], d["y_m"], d["z_m"]]) @ np.asarray(scan_to_world, float).T
    sup = d["supported"].astype(bool) if "supported" in d.dtype.names else np.ones(len(X), bool)
    if gripper_end == "end":  # same orientation the simulation used
        X, sup = X[::-1].copy(), sup[::-1].copy()
    return X, arclength(X), sup


def load_run(run_dir):
    with open(run_dir / "meta.json") as f:
        meta = json.load(f)
    sim = np.genfromtxt(run_dir / "sim_centerline.csv", delimiter=",", names=True)
    init = np.genfromtxt(run_dir / "init_centerline.csv", delimiter=",", names=True)
    xyz = lambda a: np.column_stack([a["x_m"], a["y_m"], a["z_m"]])
    return meta, sim["s_m"], xyz(sim), xyz(init), sim["part"].astype(int)


# =============================================================================
# Geometry
# =============================================================================
def point_to_polyline(P, Q):
    """Distance from each point in P to the polyline Q (exact, segment by segment)."""
    A, B = Q[:-1], Q[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-18)
    best = np.full(len(P), np.inf)
    for i in range(0, len(P), 256):  # chunks keep memory small
        p = P[i:i + 256, None, :]
        t = np.clip(((p - A) * AB).sum(-1) / L2, 0.0, 1.0)
        d = np.linalg.norm(p - (A + t[..., None] * AB), axis=-1)
        best[i:i + 256] = d.min(1)
    return best


def interp_at(s_query, s, X):
    return np.column_stack([np.interp(s_query, s, X[:, i]) for i in range(3)])


def stats(d):
    d = np.asarray(d) * 1000.0  # mm
    if len(d) == 0:
        return {"mean": np.nan, "rms": np.nan, "p95": np.nan, "max": np.nan}
    return {"mean": d.mean(), "rms": np.sqrt((d ** 2).mean()), "p95": np.percentile(d, 95), "max": d.max()}


def compare_one(scan_X, scan_s, scan_sup, meta, sim_s, sim_X, init_X):
    r, ground = meta["radius_m"], meta["ground_z_m"]
    L = scan_s[-1]
    on_scan = (sim_s >= -1e-9) & (sim_s <= L + 1e-9)
    sim_part = sim_X[on_scan]

    P = scan_X[scan_sup]
    s_P = scan_s[scan_sup]
    shape = point_to_polyline(P, sim_X)
    arc = np.linalg.norm(interp_at(s_P, sim_s, sim_X) - P, axis=1)
    lying = P[:, 2] < ground + r + 0.003
    back = point_to_polyline(sim_part, scan_X)  # sim -> scan, for the symmetric Hausdorff

    row = {"bend_scale": meta["bend_scale"], "EI_Nm2": meta["EI_Nm2"], "settled": meta["settled"],
           "sim_time_s": meta["sim_time_s"], "stretch_percent": meta["stretch_percent"]}
    for name, d in [("shape", shape), ("arc", arc), ("shape_hanging", shape[~lying]), ("shape_lying", shape[lying])]:
        for k, v in stats(d).items():
            row[f"{name}_{k}_mm"] = v
    row["plug_end_mm"] = 1000.0 * float(np.linalg.norm(interp_at([L], sim_s, sim_X)[0] - scan_X[-1]))
    row["hausdorff_mm"] = 1000.0 * float(max(shape.max(), back.max()))
    row["moved_max_mm"] = 1000.0 * float(np.linalg.norm(sim_X - init_X, axis=1).max())
    row["touchdown_scan_s_m"] = first_on_table(scan_X, scan_s, ground, r)
    row["touchdown_sim_s_m"] = first_on_table(sim_part, sim_s[on_scan], ground, r)
    per_node = {"s": s_P, "shape": shape, "arc": arc, "lying": lying}
    return row, per_node


def first_on_table(X, s, ground, r, tol=0.003):
    idx = np.flatnonzero(X[:, 2] < ground + r + tol)
    return float(s[idx[0]]) if len(idx) else float("nan")


# =============================================================================
# Plots
# =============================================================================
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def run_colors(n):
    if n == 1:
        return [CATEGORICAL[0]]
    idx = np.round(np.linspace(0, len(BLUE_RAMP) - 1, n)).astype(int)
    return [BLUE_RAMP[i] for i in idx]


def plot_overlay(path, scan_X, runs, ground):
    import matplotlib.pyplot as plt
    views = [("top view", 0, 1, "x [mm]", "y [mm]"), ("side view", 0, 2, "x [mm]", "z [mm]"),
             ("side view", 1, 2, "y [mm]", "z [mm]")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=SURFACE)
    colors = run_colors(len(runs))
    for ax, (title, i, j, xl, yl) in zip(axes, views):
        style_axes(ax)
        if j == 2:
            ax.axhline(1000 * ground, color=MUTED, linewidth=1.0)
        for (label, sim_s, _, sim_X, init_X), c in zip(runs, colors):
            out = sim_s >= 0.0  # leave out the stub clamped inside the fingers
            ax.plot(1000 * sim_X[out, i], 1000 * sim_X[out, j], color=c, linewidth=2, label=label)
        if len(runs) == 1:
            out = runs[0][1] >= 0.0
            ax.plot(1000 * runs[0][4][out, i], 1000 * runs[0][4][out, j], color=MUTED, linewidth=1,
                    linestyle="--", label="sim start")
        ax.plot(1000 * scan_X[:, i], 1000 * scan_X[:, j], color=INK, linewidth=2.5, label="scan")
        ax.plot(1000 * scan_X[0, i], 1000 * scan_X[0, j], "o", color=INK, markersize=8)
        ax.annotate("gripper", (1000 * scan_X[0, i], 1000 * scan_X[0, j]), textcoords="offset points",
                    xytext=(6, 6), color=INK_2, fontsize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Ethernet cable: scan (black) vs simulation", color=INK, x=0.01, ha="left", fontsize=12)
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right",
               ncol=min(len(labels), 7))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_errors(path, runs, per_node_list):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    style_axes(ax)
    lying = per_node_list[0]["lying"]
    s = 1000 * per_node_list[0]["s"]
    if lying.any():  # shade the stretch lying on the table
        ax.fill_between(s, 0, 1, where=lying, transform=ax.get_xaxis_transform(), color=GRID, alpha=0.6,
                        linewidth=0, label="lying on the table")
    for (label, *_), pn, c in zip(runs, per_node_list, run_colors(len(runs))):
        ax.plot(1000 * pn["s"], 1000 * pn["shape"], color=c, linewidth=2, label=label)
    ax.set_xlabel("arc length from the gripper, s [mm]")
    ax.set_ylabel("distance to the sim cable [mm]")
    ax.set_ylim(bottom=0)
    ax.set_title("Shape error along the cable (scan node to nearest sim point)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep(path, rows):
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: r["EI_Nm2"])
    EI = np.array([r["EI_Nm2"] for r in rows])
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    style_axes(ax)
    series = [("all", "shape_rms_mm", "o"), ("hanging", "shape_hanging_rms_mm", "s"), ("lying", "shape_lying_rms_mm", "^")]
    ends = []
    for (name, key, marker), c in zip(series, CATEGORICAL):
        y = np.array([r[key] for r in rows], float)
        if np.all(np.isnan(y)):
            continue
        ax.plot(EI, y, color=c, linewidth=2, marker=marker, markersize=8, label=name)
        ends.append((y[-1], name))
    # Direct labels at the right end, staggered so close values do not collide.
    for rank, (y_end, name) in enumerate(sorted(ends)):
        ax.annotate(name, (EI[-1], y_end), textcoords="offset points", xytext=(10, 11 * (rank - (len(ends) - 1) / 2)),
                    va="center", color=INK_2, fontsize=9)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)  # headroom for the end labels
    best = rows[int(np.nanargmin([r["shape_rms_mm"] for r in rows]))]
    ax.axvline(best["EI_Nm2"], color=MUTED, linewidth=1, linestyle=":")
    ax.annotate(f"best EI = {best['EI_Nm2']:.3g}", (best["EI_Nm2"], ax.get_ylim()[1]), textcoords="offset points",
                xytext=(4, -12), color=INK_2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(EI, [f"{v:.2g}" for v in EI])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(EI[0] / 1.3, EI[-1] * 1.6)
    ax.set_xlabel("bending rigidity EI [N m²]")
    ax.set_ylabel("RMS shape error [mm]")
    ax.set_title("Which stiffness matches the scan?", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
# Output tables
# =============================================================================
COLUMNS = ["run", "bend_scale", "EI_Nm2", "settled", "sim_time_s", "stretch_percent",
           "shape_mean_mm", "shape_rms_mm", "shape_p95_mm", "shape_max_mm",
           "shape_hanging_rms_mm", "shape_lying_rms_mm", "arc_rms_mm", "arc_max_mm",
           "plug_end_mm", "hausdorff_mm", "moved_max_mm", "touchdown_scan_s_m", "touchdown_sim_s_m"]


def write_metrics(out, rows):
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    best = int(np.nanargmin([r["shape_rms_mm"] for r in rows]))
    lines = [
        "# Scan vs simulation",
        "",
        "Distances in mm on the scanned stretch. *shape* = scan node to nearest sim point; "
        "*arc* = same arc length. Best run (lowest shape RMS) in bold.",
        "",
        "| run | EI [N m²] | settled | shape RMS | hanging RMS | lying RMS | shape max | arc RMS | plug end | moved |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(rows):
        b = "**" if i == best else ""
        lines.append(
            f"| {b}{r['run']}{b} | {r['EI_Nm2']:.3g} | {'yes' if r['settled'] else 'NO'} | "
            f"{b}{r['shape_rms_mm']:.1f}{b} | {r['shape_hanging_rms_mm']:.1f} | {r['shape_lying_rms_mm']:.1f} | "
            f"{r['shape_max_mm']:.1f} | {r['arc_rms_mm']:.1f} | {r['plug_end_mm']:.1f} | {r['moved_max_mm']:.1f} |")
    if not all(r["settled"] for r in rows):
        lines += ["", "Runs marked NO had not come to rest by `sim.max_time_s`: raise it and re-run before trusting them."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs", default=None, help="folder holding the run folders (default <output_dir>/<sim.init>)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(args.runs) if args.runs else repo_path(cfg["output_dir"]) / cfg["sim"]["init"]
    run_dirs = sorted([d for d in root.iterdir() if (d / "meta.json").exists()],
                      key=lambda d: json.loads((d / "meta.json").read_text())["bend_scale"]) if root.exists() else []
    if not run_dirs:
        raise SystemExit(f"no runs in {root}: run 03_franka_holds_cable/ethernet_scene.py first")

    rows, runs, per_node_list = [], [], []
    scan_X = None
    for d in run_dirs:
        meta, sim_s, sim_X, init_X, _ = load_run(d)
        if scan_X is None:
            scan_X, scan_s, scan_sup = load_scan(meta["scan"], meta["gripper_end"],
                                                 meta.get("scan_to_world", np.eye(3).tolist()))
            ground = meta["ground_z_m"]
        row, pn = compare_one(scan_X, scan_s, scan_sup, meta, sim_s, sim_X, init_X)
        row["run"] = d.name
        rows.append(row)
        per_node_list.append(pn)
        runs.append((f"EI {meta['EI_Nm2']:.2g} N m²", sim_s, None, sim_X, init_X))

    out = root / "compare"
    out.mkdir(exist_ok=True)
    print(write_metrics(out, rows))
    plot_overlay(out / "overlay.png", scan_X, runs, ground)
    plot_errors(out / "errors.png", runs, per_node_list)
    if len(rows) > 1:
        plot_sweep(out / "sweep.png", rows)
    print(f"\nwrote {out}/ (metrics.csv, summary.md, overlay.png, errors.png{', sweep.png' if len(rows) > 1 else ''})")


if __name__ == "__main__":
    main()
