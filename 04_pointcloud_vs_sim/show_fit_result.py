"""Show a finished fit: scan vs simulation as plots and as point clouds, plus a report.

    # from the repository root, after 04_pointcloud_vs_sim/fit_to_scan.py
    python 04_pointcloud_vs_sim/show_fit_result.py --config configs/ethernet_cat6.json
    python 04_pointcloud_vs_sim/show_fit_result.py --config configs/ethernet_cat6.json --show     # 3-D window (Open3D)

No simulation is run: it reads what the fit saved (best.json, evaluations.csv,
runs/<best>/) and the tube fit's cable points (cable_points.ply).

Writes into <fit folder>/result/:

    report.md                  the fitted values, what each one means, how well it matches
    compare.png                three views + error along the cable + distance histogram
    scan_points_by_error.ply   the real scan's cable points, coloured by their distance to
                               the simulated cable (light = close, dark = far)
    sim_cable.ply              the simulated cable as a tube of points (orange)
    compare_cloud.ply          both together

The .ply files are in the scan's own frame and units: open them in CloudCompare
together with the original scan.

Needs numpy and matplotlib; Open3D only for --show.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
G = 9.81

# Reference palette (dataviz skill), same as compare_to_scan.py.
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # light -> dark


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = load_module("compare_to_scan", REPO / "04_pointcloud_vs_sim" / "compare_to_scan.py")


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


# =============================================================================
# PLY in / out (no extra packages)
# =============================================================================
PLY_TYPES = {"char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1", "short": "i2", "int16": "i2",
             "ushort": "u2", "uint16": "u2", "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
             "float": "f4", "float32": "f4", "double": "f8", "float64": "f8"}


def read_ply_points(path):
    """x, y, z of the vertices of a PLY file (ascii or binary), as an (N, 3) float array."""
    with open(path, "rb") as f:
        fmt, element, n, props = None, None, 0, []
        while True:
            parts = f.readline().decode("ascii", "replace").split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                element = parts[1]
                if element == "vertex":
                    n = int(parts[2])
                elif n == 0:
                    raise SystemExit(f"{path}: '{element}' comes before the vertices; not supported")
            elif parts[0] == "property" and element == "vertex":
                if parts[1] == "list":
                    raise SystemExit(f"{path}: list property on vertices; not supported")
                props.append((parts[2], PLY_TYPES[parts[1]]))
            elif parts[0] == "end_header":
                break
        names = [name for name, _ in props]
        if fmt == "ascii":
            data = np.loadtxt(f, max_rows=n, usecols=range(len(props)), ndmin=2)
            return data[:, [names.index("x"), names.index("y"), names.index("z")]].astype(float)
        endian = "<" if fmt == "binary_little_endian" else ">"
        dtype = np.dtype([(name, endian + t) for name, t in props])
        data = np.frombuffer(f.read(dtype.itemsize * n), dtype=dtype, count=n)
    return np.column_stack([data["x"], data["y"], data["z"]]).astype(float)


def write_ply(path, points, colours):
    """Binary PLY with uchar colours (fast to write and to open)."""
    points = np.asarray(points, np.float32)
    colours = np.asarray(colours, np.uint8)
    rec = np.empty(len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                       ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = points.T
    rec["red"], rec["green"], rec["blue"] = colours.T
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {len(points)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        f.write(rec.tobytes())


def hex_rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def ramp_colours(values, vmax):
    """Sequential blue ramp: 0 -> lightest, vmax and above -> darkest (linear between steps)."""
    stops = np.array([hex_rgb(h) for h in BLUE_RAMP], float)
    t = np.clip(np.asarray(values) / vmax, 0.0, 1.0) * (len(stops) - 1)
    i = np.minimum(t.astype(int), len(stops) - 2)
    f = (t - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)


def tube_points(nodes, r, spacing=0.002, ring=16):
    """Points on the surface of a tube of radius r around the polyline `nodes`."""
    out = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        t = d / L
        u = np.cross(t, [0.0, 0.0, 1.0] if abs(t[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(t, u)
        ang = np.linspace(0.0, 2 * np.pi, ring, endpoint=False)
        circle = r * (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * v)
        for s in np.arange(0.0, L, spacing):
            out.append(a + s * t + circle)
    return np.vstack(out)


# =============================================================================
# Loading the fit
# =============================================================================
def load_fit(fit_dir):
    best = json.loads((fit_dir / "best.json").read_text())
    rows = list(csv.DictReader(open(fit_dir / "evaluations.csv")))
    run_dir = fit_dir / "runs" / best["run"]
    if not (run_dir / "meta.json").exists():
        raise SystemExit(f"best run folder not found: {run_dir}")
    cfg_path = fit_dir / "best_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return best, rows, run_dir, cfg


def find_cloud(cfg_scan_csv, given):
    if given:
        return repo_path(given)
    p = repo_path(cfg_scan_csv).parent / "cable_points.ply"
    return p if p.exists() else None


# =============================================================================
# Plot
# =============================================================================
def style(ax):
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


def make_figure(path, scan_X, sim_X, cloud_W, ground, per_node, surf_mm, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 9.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.25)
    views = [("top view", 0, 1, "x [mm]", "y [mm]"), ("side view", 0, 2, "x [mm]", "z [mm]"),
             ("side view", 1, 2, "y [mm]", "z [mm]")]
    rng = np.random.default_rng(0)
    pts = cloud_W[rng.choice(len(cloud_W), min(len(cloud_W), 6000), replace=False)] if cloud_W is not None else None
    for k, (name, i, j, xl, yl) in enumerate(views):
        ax = fig.add_subplot(gs[0, k])
        style(ax)
        if j == 2:
            ax.axhline(1000 * ground, color=MUTED, linewidth=1.0)
        if pts is not None:
            ax.scatter(1000 * pts[:, i], 1000 * pts[:, j], s=1.5, color=GRID, linewidths=0, label="scan points")
        ax.plot(1000 * scan_X[:, i], 1000 * scan_X[:, j], color=INK, linewidth=2.2, label="scan centreline")
        ax.plot(1000 * sim_X[:, i], 1000 * sim_X[:, j], color=BLUE, linewidth=2.0, label="simulation (fitted)")
        ax.plot(1000 * scan_X[0, i], 1000 * scan_X[0, j], "o", color=INK, markersize=8)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(name, color=INK, fontsize=10, loc="left")
        if k == 0:
            handles, labels = ax.get_legend_handles_labels()

    ax = fig.add_subplot(gs[1, :2])
    style(ax)
    s = 1000 * per_node["s"]
    if per_node["lying"].any():
        ax.fill_between(s, 0, 1, where=per_node["lying"], transform=ax.get_xaxis_transform(), color=GRID,
                        alpha=0.6, linewidth=0, label="lying on the table")
    ax.plot(s, 1000 * per_node["shape"], color=BLUE, linewidth=2, label="scan centreline to simulation")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("arc length from the gripper [mm]")
    ax.set_ylabel("distance [mm]")
    ax.set_title("Error along the cable", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)

    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    if surf_mm is not None:
        hi = max(np.percentile(surf_mm, 99), 1.0)
        ax.hist(np.clip(surf_mm, 0, hi), bins=40, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(np.median(surf_mm), color=INK, linewidth=1.2, linestyle=":")
        ax.annotate(f"median {np.median(surf_mm):.1f} mm", (np.median(surf_mm), ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(4, -12), color=INK_2, fontsize=8)
        ax.set_xlabel("scan point to simulated cable surface [mm]")
        ax.set_ylabel("points")
        ax.set_title("Real scan points: how far off", color=INK, fontsize=10, loc="left")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "no cable_points.ply found\n(--cloud PATH)", ha="center", va="center", color=INK_2)

    fig.suptitle(title, color=INK, x=0.01, ha="left", fontsize=12)
    fig.legend(handles, labels, frameon=False, fontsize=8, labelcolor=INK_2, loc="upper right", ncol=3, markerscale=6)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Report
# =============================================================================
def fmt(v, nd=3):
    if v is None or v == "":
        return "—"
    v = float(v)
    return f"{v:.{nd}g}" if (abs(v) < 1e-2 or abs(v) >= 1e4) and v != 0 else f"{v:.{nd}f}".rstrip("0").rstrip(".")


def write_report(path, best, rows, meta, cfg, row, surf_mm, n_cloud, figure_name):
    cab = cfg.get("cable", {})
    con = cfg.get("contact", {})
    sim = cfg.get("sim", {})
    EI = float(best["EI_Nm2"])
    w = float(meta.get("mass_per_length_kg_m", cab.get("mass_per_length_kg_m", 0.045))) * G
    gj_ratio = meta["GJ_Nm2"] / meta["EI_Nm2"] if meta.get("EI_Nm2") else 0.77
    kappa = best.get("rest_curvature_per_m")
    phi = best.get("rest_curl_direction_deg")
    ell = (EI / w) ** (1.0 / 3.0)
    L = []
    L += [f"# Fit result: {cfg.get('name', 'cable')}", ""]
    L += [f"Fitted to `{meta.get('scan', '?')}`, scored on **{best.get('part_scored', 'all')}** of the cable, "
          f"{best.get('runs', len(rows))} simulations, {best.get('wall_min', '?')} min. Figure: `{figure_name}`.", ""]

    L += ["## Found by the optimisation", "",
          "| variable | JSON key | value | unit | what it means |", "|---|---|---|---|---|",
          f"| bending stiffness EI | `cable.bend_rigidity_EI_Nm2` | **{EI:.4g}** | N m² | how hard the cable is to bend "
          "(torque per unit curvature). Higher = straighter, sticks out further from the gripper before it droops. |"]
    if kappa is not None:
        coil = f"coil radius {1000 / kappa:.0f} mm" if kappa > 0 else "straight"
        L += [f"| curl strength κ | `cable.rest_curvature_per_m` | **{float(kappa):.3g}** | 1/m | the cable's memory of the "
              f"coil it came in: the curvature it has when nothing pushes on it ({coil}). 0 = a cable that wants to be "
              "straight. |",
              f"| curl direction φ | `cable.rest_curl_direction_deg` | **{float(phi):.0f}** | deg | to which side the cable "
              "curls, around its own axis, measured from the cable's frame at the gripper (0° = that frame's x axis). "
              "It depends on how the cable sits in the fingers: not a material property, only meaningful with this grip. |"]
    L += [""]

    L += ["## Follows from those (computed, not fitted)", "",
          "| quantity | value | what it means |", "|---|---|---|",
          f"| twist stiffness GJ | {EI * gj_ratio:.3g} N m² | resistance to twisting; kept at {gj_ratio:.2f} × EI (not fitted) |",
          f"| EI ÷ weight per metre | {EI / w:.3g} m³ | the only stiffness number a still cable can reveal: double EI and "
          "the mass together and the shape doesn't change. So EI is only right for the mass below. |",
          f"| gravity-bending length ℓ = (EI/w)^⅓ | {1000 * ell:.0f} mm | pieces much shorter than ℓ stick out almost straight, "
          "much longer ones droop like a rope |"]
    if kappa:
        L += [f"| coil radius 1/κ | {1000 / float(kappa):.0f} mm | the loop the cable would form lying loose on a table "
              "(check it by eye: lay the cable down) |"]
    L += [""]

    L += ["## Kept fixed (from the JSON, not fitted)", "",
          "| input | value | note |", "|---|---|---|",
          f"| mass per metre | {1000 * w / G:.1f} g/m | weigh 1 m of your cable: EI scales with it |",
          f"| radius | {1000 * meta['radius_m']:.2f} mm | from the tube fit |",
          f"| stretch / shear rigidity EA, kGA | {fmt(cab.get('stretch_rigidity_EA_N'))} N, {fmt(cab.get('shear_rigidity_kGA_N'))} N "
          "| solver choice, not the real values: much stiffer makes VBD bend too easily (`check_stiffness.py`) |",
          f"| segment length | {1000 * meta['segment_length_m']:.0f} mm ({meta['segments']} segments) | |",
          f"| grip in the fingers / plug | {fmt(cab.get('grip_length_m'))} m / {fmt(meta.get('plug_length_m'))} m, plug "
          f"{'held at its scanned place' if meta.get('plug_end_fixed') else 'free, ' + fmt(meta.get('plug_mass_kg')) + ' kg'} | |",
          f"| friction on the table | μ = {fmt(meta.get('friction_mu', con.get('mu')))} | |",
          f"| solver | {sim.get('substeps', '?')} substeps × {sim.get('iterations', '?')} iterations, "
          f"friction_epsilon {fmt(sim.get('friction_epsilon'))} | |", ""]

    L += ["## How well it matches", "",
          "| measure | value |", "|---|---|",
          f"| centreline RMS, whole cable | **{row['shape_rms_mm']:.2f} mm** |",
          f"| centreline RMS, hanging part / lying part | {row['shape_hanging_rms_mm']:.2f} / {row['shape_lying_rms_mm']:.2f} mm |",
          f"| centreline worst point | {row['shape_max_mm']:.1f} mm |",
          f"| how far the cable moved from the scanned start | {row['moved_max_mm']:.1f} mm |"]
    if surf_mm is not None:
        L += [f"| real scan points ({n_cloud}) to the simulated surface: median / mean / 95 % / max | "
              f"{np.median(surf_mm):.1f} / {surf_mm.mean():.1f} / {np.percentile(surf_mm, 95):.1f} / {surf_mm.max():.1f} mm |"]
    L += ["", "Differences below ~2–3 mm are within the pipeline's own noise (re-meshing, settling).", ""]

    good = sorted([r for r in rows if r.get("score_mm") not in ("", None) and np.isfinite(float(r["score_mm"]))],
                  key=lambda r: float(r["score_mm"]))[:10]
    L += ["## The 10 best candidates tried", "",
          "| EI [N m²] | curl [1/m] | direction [deg] | score [mm] |", "|---|---|---|---|"]
    for r in good:
        L += [f"| {float(r['EI_Nm2']):.4g} | {fmt(r.get('rest_curvature_per_m'))} | {fmt(r.get('rest_curl_direction_deg'))} | "
              f"{float(r['score_mm']):.2f} |"]
    L += ["", "If several quite different rows score almost the same, the scan can't tell those values apart: "
          "they are equally good explanations of this one shape.", ""]

    L += ["## What this does and doesn't tell you", "",
          "- These are **effective values**: the ones that make this simulation reproduce this scan. The part on the "
          "table and the cable's twist also depend on how the cable was put down, which a scan can't show.",
          "- **EI is only as right as the mass** you gave it (above).",
          "- **To test the fit**, scan the cable in a second pose, copy `best_config.json`, change only "
          "`scan.centerline_csv`, run `ethernet_scene.py` and `compare_to_scan.py` without refitting, and compare "
          "that error with the one here.", ""]
    path.write_text("\n".join(L) + "\n")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True, help="the JSON the fit was run with")
    ap.add_argument("--fit", default=None, help="fit folder (default <output_dir>_fit)")
    ap.add_argument("--cloud", default=None, help="scan cable points .ply (default: cable_points.ply next to the centreline)")
    ap.add_argument("--max-mm", type=float, default=10.0, help="distance shown darkest in scan_points_by_error.ply")
    ap.add_argument("--show", action="store_true", help="open a 3-D window (needs open3d)")
    args = ap.parse_args()

    cfg_in = cs.load_config(args.config)
    fit_dir = Path(args.fit) if args.fit else Path(str(repo_path(cfg_in["output_dir"])) + "_fit")
    best, rows, run_dir, cfg = load_fit(fit_dir)
    cfg = cfg or cfg_in
    out = fit_dir / "result"
    out.mkdir(exist_ok=True)

    # Scan and simulation, both in the simulation's world frame (up = +z).
    meta, sim_s, sim_X, init_X, _ = cs.load_run(run_dir)
    R = np.asarray(meta["scan_to_world"], float)
    scan_X, scan_s, scan_sup = cs.load_scan(meta["scan"], meta["gripper_end"], R)
    row, per_node = cs.compare_one(scan_X, scan_s, scan_sup, meta, sim_s, sim_X, init_X)
    sim_free = sim_X[sim_s >= 0.0]  # leave out the stub inside the fingers
    r = float(meta["radius_m"])

    # Real scan points.
    summary_path = Path(meta["scan"]).parent / "summary.json"
    units = json.loads(summary_path.read_text()).get("ply_units", "m") if summary_path.exists() else "m"
    scale = 0.001 if units == "mm" else 1.0  # PLY units -> metres
    cloud_path = find_cloud(meta["scan"], args.cloud)
    cloud_W, surf_mm = None, None
    if cloud_path is not None and cloud_path.exists():
        cloud_S = read_ply_points(cloud_path)
        cloud_W = (cloud_S * scale) @ R.T
        surf_mm = 1000.0 * np.abs(cs.point_to_polyline(cloud_W, sim_free) - r)
        print(f"scan points: {len(cloud_W)} from {cloud_path}")
    else:
        print("no cable_points.ply found: point-cloud comparison skipped (give it with --cloud)")

    title = (f"Fitted simulation vs scan: EI {best['EI_Nm2']:.3g} N m²"
             + (f", curl {best['rest_curvature_per_m']:.2f}/m @ {best['rest_curl_direction_deg']:.0f}°"
                if best.get("rest_curvature_per_m") is not None else "")
             + f"  —  RMS {row['shape_rms_mm']:.1f} mm")
    make_figure(out / "compare.png", scan_X, sim_free, cloud_W, meta["ground_z_m"], per_node, surf_mm, title)

    # Point clouds, in the scan's own frame and units (overlay the original scan).
    to_scan = lambda P: (P @ R) / scale  # noqa: E731  world [m] -> scan frame, PLY units
    tube = tube_points(sim_free, r)
    tube_rgb = np.tile(hex_rgb(ORANGE), (len(tube), 1))
    write_ply(out / "sim_cable.ply", to_scan(tube), tube_rgb)
    if cloud_W is not None:
        err_rgb = ramp_colours(surf_mm, args.max_mm)
        write_ply(out / "scan_points_by_error.ply", cloud_S, err_rgb)
        write_ply(out / "compare_cloud.ply", np.vstack([cloud_S, to_scan(tube)]), np.vstack([err_rgb, tube_rgb]))

    write_report(out / "report.md", best, rows, meta, cfg, row, surf_mm, 0 if cloud_W is None else len(cloud_W),
                 "compare.png")
    print(f"centreline RMS {row['shape_rms_mm']:.2f} mm (hanging {row['shape_hanging_rms_mm']:.2f}, lying "
          f"{row['shape_lying_rms_mm']:.2f})" + (f"; scan points to simulated surface: median {np.median(surf_mm):.1f} mm"
                                                 if surf_mm is not None else ""))
    print(f"wrote {out}/ (report.md, compare.png, sim_cable.ply"
          + (", scan_points_by_error.ply, compare_cloud.ply" if cloud_W is not None else "") + ")")

    if args.show:
        try:
            import open3d as o3d
        except ImportError:
            raise SystemExit("--show needs open3d (pip install open3d); the .ply files can be opened in CloudCompare")
        geoms = []
        if cloud_W is not None:
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cloud_W))
            pc.colors = o3d.utility.Vector3dVector(ramp_colours(surf_mm, args.max_mm) / 255.0)
            geoms.append(pc)
        tc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tube))
        tc.paint_uniform_color(hex_rgb(ORANGE) / 255.0)
        geoms.append(tc)
        line = o3d.geometry.LineSet(o3d.utility.Vector3dVector(scan_X),
                                    o3d.utility.Vector2iVector(np.c_[np.arange(len(scan_X) - 1), np.arange(1, len(scan_X))]))
        line.paint_uniform_color([0.05, 0.05, 0.05])
        geoms.append(line)
        o3d.visualization.draw_geometries(geoms, window_name="scan (blue = distance) vs simulation (orange)")


if __name__ == "__main__":
    main()
