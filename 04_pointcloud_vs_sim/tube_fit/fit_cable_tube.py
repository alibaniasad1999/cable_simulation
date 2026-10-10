"""Fit a tube (smooth 3-D curve + constant radius) to a cable in a point cloud.

    python fit_cable_tube.py ../../data/Ethernet.ply

Steps: local tube detection from the normals -> radius -> march along the cable
-> B-spline tube fit on the raw surface points -> trim -> write results.
Everything is computed in millimetres in the scanner frame; the CSV is in metres.
See README.md for the method and the outputs.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy.interpolate import BSpline, make_lsq_spline
from scipy.sparse import csgraph, coo_matrix
from scipy.spatial import cKDTree


# --------------------------------------------------------------------------- io
def load_cloud(path, units="auto"):
    """Return points [mm], unit normals (or None), colours (or None), mm-per-file-unit."""
    pcd = o3d.io.read_point_cloud(path)
    P = np.asarray(pcd.points, dtype=float)
    if len(P) == 0:
        raise SystemExit(f"no points in {path}")
    if units == "auto":
        units = "mm" if np.median(np.ptp(P, axis=0)) > 10.0 else "m"
    scale = 1.0 if units == "mm" else 1000.0
    N = np.asarray(pcd.normals, dtype=float) if pcd.has_normals() else None
    C = np.asarray(pcd.colors, dtype=float) if pcd.has_colors() else None
    return P * scale, N, C, scale


def estimate_normals(P, radius):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=60))
    pcd.orient_normals_consistent_tangent_plane(20)
    return np.asarray(pcd.normals)


def voxel_downsample(P, N, voxel):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pcd.normals = o3d.utility.Vector3dVector(N)
    ds = pcd.voxel_down_sample(voxel)
    n = np.asarray(ds.normals)
    return np.asarray(ds.points), n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


# ------------------------------------------------------------- 1. tube detection
def tube_features(P, N, rho):
    """Per point: local axis, local radius, collapse scatter, normal-spread eigenvalues.

    On a cylinder the normals span a plane; the direction they do NOT span is the
    axis. Projected on the plane across the axis, p - r n is the same point for
    every neighbour, which gives r in closed form.
    """
    n_pts = len(P)
    axis = np.zeros((n_pts, 3))
    r_loc = np.full(n_pts, np.nan)
    scatter = np.full(n_pts, np.nan)
    mu = np.full((n_pts, 3), np.nan)
    tree = cKDTree(P)
    for i, idx in enumerate(tree.query_ball_point(P, rho)):
        if len(idx) < 12:
            continue
        n = N[idx]
        w, v = np.linalg.eigh(n.T @ n / len(idx))
        a = v[:, 0]
        p = P[idx]
        u = p - np.outer(p @ a, a)
        m = n - np.outer(n @ a, a)
        uc, mc = u - u.mean(0), m - m.mean(0)
        den = float((mc * mc).sum())
        if den < 1e-9:
            continue
        r = float((uc * mc).sum()) / den
        axis[i], r_loc[i], mu[i] = a, r, w
        scatter[i] = np.sqrt(((uc - r * mc) ** 2).sum(1).mean())
    return axis, r_loc, scatter, mu


def tube_candidates(r_loc, scatter, mu):
    with np.errstate(invalid="ignore"):
        return (mu[:, 0] < 0.05) & (mu[:, 1] > 0.04) & (scatter < 0.35 * np.abs(r_loc))


def radius_mode(r, lo=0.3, hi=40.0):
    """Most common tube radius [mm]: the cable is the longest thin tube in the scene."""
    r = r[(r > lo) & (r < hi)]
    if len(r) < 50:
        raise SystemExit("no tube-like structure found; pass --radius-mm")
    hist, edges = np.histogram(np.log(r), bins=60, range=(np.log(lo), np.log(hi)))
    hist = np.convolve(hist, [1, 2, 1], mode="same")
    k = int(np.argmax(hist))
    centre = np.exp(0.5 * (edges[k] + edges[k + 1]))
    return float(np.median(r[(r > 0.75 * centre) & (r < 1.33 * centre)]))


# ------------------------------------------------------------------- 2. marching
def largest_component_seed(Q, link):
    pairs = cKDTree(Q).query_pairs(link, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Q), len(Q)))
    _, lab = csgraph.connected_components(g, directed=False)
    comp = np.flatnonzero(lab == np.argmax(np.bincount(lab)))
    q = Q[comp]
    proj = (q - q.mean(0)) @ np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
    return int(comp[np.argsort(proj)[len(proj) // 2]])


def local_tube(P, N, tree, x, d, r, tol):
    """Surface points of the tube section at x (axis d): right distance, normal pointing out."""
    idx = np.asarray(tree.query_ball_point(x, 1.5 * r + tol), dtype=int)
    if len(idx) == 0:
        return idx
    v = P[idx] - x
    along = v @ d
    perp = v - np.outer(along, d)
    dist = np.linalg.norm(perp, axis=1)
    out = (perp * N[idx]).sum(1) / np.maximum(dist, 1e-9)
    return idx[(np.abs(along) < r) & (np.abs(dist - r) < tol) & (out > 0.5)]


def march(P, N, tree, x0, d0, r, max_gap):
    """Walk along the cable from x0 in direction d0, re-centring on the raw surface points.

    The normal test rejects a second strand lying against this one, so the walk
    follows the cable through a self-crossing.
    """
    step = r
    x, d = x0.copy(), d0 / np.linalg.norm(d0)
    nodes, gap = [], 0.0
    for _ in range(50000):
        x = x + step * d
        found = False
        for _ in range(2):
            idx = local_tube(P, N, tree, x, d, r, 0.35 * r + 0.1 * gap)
            if len(idx) < 6:
                break
            found = True
            off = (P[idx] - r * N[idx]).mean(0) - x
            x = x + off - (off @ d) * d
            a = np.linalg.eigh(N[idx].T @ N[idx])[1][:, 0]
            d = d + 0.5 * a * np.sign(a @ d)
            d /= np.linalg.norm(d)
        gap = 0.0 if found else gap + step
        nodes.append((x.copy(), found))
        if gap > max_gap:
            break
    while nodes and not nodes[-1][1]:
        nodes.pop()
    return [n[0] for n in nodes]


def trace_centreline(P, N, x0, d0, r, max_gap):
    tree = cKDTree(P)
    fwd = march(P, N, tree, x0, d0, r, max_gap)
    bwd = march(P, N, tree, x0, -d0, r, max_gap)
    return np.array(bwd[::-1] + [x0] + fwd)


# ------------------------------------------------------------------ 3. tube fit
def clamped_knots(n_ctrl, k=3):
    inner = np.linspace(0.0, 1.0, n_ctrl - k + 1)
    return np.r_[[0.0] * k, inner, [1.0] * k]


def initial_spline(nodes, ctrl_spacing):
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    n_ctrl = int(np.clip(s[-1] / ctrl_spacing + 3, 6, max(6, len(nodes) // 2)))
    t = clamped_knots(n_ctrl)
    return make_lsq_spline(s / s[-1], nodes, t, k=3)


def sample_curve(spl, ds=0.25):
    u = np.linspace(0.0, 1.0, 400)
    length = np.linalg.norm(np.diff(spl(u), axis=0), axis=1).sum()
    u = np.linspace(0.0, 1.0, max(400, int(length / ds) + 1))
    S = spl(u)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(S, axis=0), axis=1))]
    return u, S, s


def assign(P, N, spl, r, band):
    """Surface points belonging to the tube: near r from the curve, normal pointing out."""
    u, S, s = sample_curve(spl)
    d, j = cKDTree(S).query(P, distance_upper_bound=r + band)
    ok = np.isfinite(d) & (j > 0) & (j < len(S) - 1)
    idx = np.flatnonzero(ok)
    e = (P[idx] - S[j[idx]]) / np.maximum(d[idx], 1e-9)[:, None]
    keep = (np.abs(d[idx] - r) < band) & ((e * N[idx]).sum(1) > 0.3)
    idx = idx[keep]
    return idx, u[j[idx]], s[j[idx]], d[idx], e[keep], s[-1]


def fit_tube(P, N, spl, r, iters=12, smooth=1e-5, fix_radius=False):
    """Robust Gauss-Newton on control points and radius; residual = distance - r."""
    t, k = spl.t, spl.k
    C = spl.c.copy()
    nc = len(C)
    D2 = np.diff(np.eye(nc), 2, axis=0)
    K = D2.T @ D2
    sigma = 0.15 * r
    for _ in range(iters):
        spl = BSpline(t, C, k)
        band = max(4.0 * sigma, 0.08 * r)
        idx, u, _, d, e, _ = assign(P, N, spl, r, band)
        if len(idx) < 4 * nc:
            raise SystemExit("too few cable points support the fit")
        res = d - r
        sigma = max(1.4826 * np.median(np.abs(res - np.median(res))), 0.01)
        w = np.clip(1.0 - (res / (4.685 * sigma)) ** 2, 0.0, None) ** 2
        B = BSpline.design_matrix(u, t, k).toarray()
        J = np.hstack([-(e[:, [0]] * B), -(e[:, [1]] * B), -(e[:, [2]] * B), -np.ones((len(idx), 1))])
        JW = J * w[:, None]
        H = JW.T @ J
        g = -JW.T @ res
        lam = smooth * w.sum() / nc
        for a in range(3):
            sl = slice(a * nc, (a + 1) * nc)
            H[sl, sl] += lam * K
            g[sl] -= lam * K @ C[:, a]
        H[np.diag_indices_from(H)] += 1e-9 * np.trace(H) / len(H)
        if fix_radius:
            H[-1, :], H[:, -1], H[-1, -1], g[-1] = 0.0, 0.0, 1.0, 0.0
        dx = np.linalg.solve(H, g)
        C = C + dx[:-1].reshape(3, nc).T
        r = r + dx[-1]
    return BSpline(t, C, k), float(r), float(sigma)


# ---------------------------------------------------------------------- 4. trim
def supported_range(s_pts, length, bin_mm=2.0):
    n_bins = max(1, int(np.ceil(length / bin_mm)))
    bins = np.minimum((s_pts / bin_mm).astype(int), n_bins - 1)
    sup = np.bincount(bins, minlength=n_bins) >= 3
    first, last = np.flatnonzero(sup)[[0, -1]]
    s_ok = s_pts[sup[bins]]
    gaps, start = [], None
    for b in range(first, last + 1):
        if not sup[b] and start is None:
            start = b
        if sup[b] and start is not None:
            if (b - start) * bin_mm >= 4.0:
                gaps.append([start * bin_mm, b * bin_mm])
            start = None
    return float(s_ok.min()), float(s_ok.max()), gaps


def trim_ends(spl, s_pts, r, min_bend_radii, bin_mm=2.0):
    """Cut back each end until the curve is a plausible cable: well supported, not kinked.

    Plugs and gripper fingers are not tubes; the fit wanders there and bends
    tighter than a cable can.
    """
    u, S, s = sample_curve(spl)
    t = np.gradient(S, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    kappa = np.linalg.norm(np.gradient(t, axis=0), axis=1) / np.gradient(s)
    n_bins = int(np.ceil(s[-1] / bin_mm))
    count = np.bincount(np.minimum((s_pts / bin_mm).astype(int), n_bins - 1), minlength=n_bins)
    kmax = np.zeros(n_bins)
    np.maximum.at(kmax, np.minimum((s / bin_mm).astype(int), n_bins - 1), kappa)
    good = (count >= 0.5 * np.median(count[count > 0])) & (kmax < 1.0 / (min_bend_radii * r))
    need = max(2, int(np.ceil(4.0 * r / bin_mm)))        # a clean run this long ends the trim
    run = np.convolve(good, np.ones(need), mode="valid") == need
    ok = np.flatnonzero(run)
    if len(ok) == 0:
        return 0.0, float(s[-1])
    return ok[0] * bin_mm, min((ok[-1] + need) * bin_mm, float(s[-1]))


def resample(spl, s_a, s_b, n_nodes):
    u, S, s = sample_curve(spl, ds=0.1)
    s_new = np.linspace(s_a, s_b, n_nodes)
    X = np.column_stack([np.interp(s_new, s, S[:, a]) for a in range(3)])
    T = spl.derivative()(np.interp(s_new, s, u))
    return s_new - s_a, X, T / np.linalg.norm(T, axis=1, keepdims=True)


# -------------------------------------------------------------------- 5. output
def render_preview(path, cloud, cable, curve, size=900):
    """Three orthographic views: cloud grey, cable points green, centreline red."""
    c0 = curve.mean(0)
    axes = np.linalg.svd(curve - c0, full_matrices=False)[2]
    panels = []
    for a, b, d in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        layers = [(cloud, (190, 190, 190), 1), (cable, (40, 170, 70), 1), (curve, (220, 30, 30), 2)]
        xy = [((p - c0) @ axes[[a, b]].T, (p - c0) @ axes[d]) for p, _, _ in layers]
        lo = np.min([q.min(0) for q, _ in xy], axis=0)
        hi = np.max([q.max(0) for q, _ in xy], axis=0)
        sc = (size - 40) / max(hi - lo)
        img = np.full((int((hi[1] - lo[1]) * sc) + 40, int((hi[0] - lo[0]) * sc) + 40, 3), 255, np.uint8)
        for (q, depth), (_, col, px) in zip(xy, layers):
            o = np.argsort(depth)
            col_i = ((q[o, 0] - lo[0]) * sc + 20).astype(int)
            row_i = img.shape[0] - 1 - ((q[o, 1] - lo[1]) * sc + 20).astype(int)
            for dr in range(px):
                for dc in range(px):
                    img[np.clip(row_i + dr, 0, img.shape[0] - 1), np.clip(col_i + dc, 0, img.shape[1] - 1)] = col
        panels.append(img)
    h = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, h - p.shape[0]), (0, 10), (0, 0)), constant_values=255) for p in panels]
    o3d.io.write_image(path, o3d.geometry.Image(np.ascontiguousarray(np.hstack(panels))))


def write_ply(path, points, colours=None):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if colours is not None:
        pcd.colors = o3d.utility.Vector3dVector(colours)
    o3d.io.write_point_cloud(path, pcd)


def pick_seed(P):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("shift+click ONE point on the cable, then press Q")
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    picked = vis.get_picked_points()
    if not picked:
        raise SystemExit("no point picked")
    return P[picked[0]]


# ------------------------------------------------------------------------ main
def fit_cable(P, N, radius_mm=None, seed_mm=None, voxel=0.8, max_gap=20.0,
              ctrl_spacing=8.0, smooth=1e-5, min_bend_radii=4.0, log=print):
    """Full pipeline on points [mm] with normals. Returns a result dict."""
    Pd, Nd = voxel_downsample(P, N, voxel)
    rho = 2.0 * radius_mm if radius_mm else 5.0
    axis, r_loc, scatter, mu = tube_features(Pd, Nd, rho)
    cand = tube_candidates(r_loc, scatter, mu)
    if np.median(np.sign(r_loc[cand])) < 0:           # normals point inward
        N, Nd, r_loc = -N, -Nd, -r_loc
    r = radius_mode(r_loc[cand]) if radius_mm is None else float(radius_mm)
    log(f"radius from local fits: {r:.3f} mm  (feature radius {rho:.1f} mm)")
    if radius_mm is None and abs(rho - 2.0 * r) > 0.5 * r:
        axis, r_loc, scatter, mu = tube_features(Pd, Nd, 2.0 * r)
        cand = tube_candidates(r_loc, scatter, mu)
        r = radius_mode(r_loc[cand])
        log(f"radius, second pass:    {r:.3f} mm")

    sel = cand & (np.abs(r_loc - r) < 0.4 * r)
    Q, A = Pd[sel] - r * Nd[sel], axis[sel]
    log(f"tube-like points: {sel.sum()} of {len(Pd)} (downsampled)")
    seed = (int(cKDTree(Q).query(seed_mm)[1]) if seed_mm is not None
            else largest_component_seed(Q, 0.6 * r))
    near = cKDTree(Q).query_ball_point(Q[seed], 1.5 * r)
    d0 = (A[near] * np.sign(A[near] @ A[seed])[:, None]).mean(0)
    nodes = trace_centreline(Pd, Nd, Q[near].mean(0), d0, r, max_gap)
    log(f"marched {len(nodes)} nodes, about {np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum():.0f} mm")

    spl = initial_spline(nodes, ctrl_spacing)
    fixed = radius_mm is not None
    spl, r, sigma = fit_tube(P, N, spl, r, smooth=smooth, fix_radius=fixed)
    band = max(4.0 * sigma, 0.08 * r)
    idx, _, s_pts, d, _, length = assign(P, N, spl, r, band)
    t_a, t_b = trim_ends(spl, s_pts, r, min_bend_radii)
    if t_a > 0.0 or t_b < length:
        log(f"trimmed {t_a:.0f} mm and {length - t_b:.0f} mm of non-cable ends; refitting")
        _, core, _ = resample(spl, t_a, t_b, int((t_b - t_a) / r) + 1)
        spl, r, sigma = fit_tube(P, N, initial_spline(core, ctrl_spacing), r, iters=8,
                                 smooth=smooth, fix_radius=fixed)
        band = max(4.0 * sigma, 0.08 * r)
        idx, _, s_pts, d, _, length = assign(P, N, spl, r, band)
    s_a, s_b, gaps = supported_range(s_pts, length)
    inside = (s_pts >= s_a) & (s_pts <= s_b)
    idx, res = idx[inside], d[inside] - r
    log(f"tube fit: r = {r:.3f} mm, length = {s_b - s_a:.1f} mm, "
        f"{len(idx)} cable points, residual RMS = {np.sqrt((res ** 2).mean()):.3f} mm")
    return dict(spline=spl, radius=r, s_range=(s_a, s_b), gaps=[[g0 - s_a, g1 - s_a] for g0, g1 in gaps],
                inliers=idx, residuals=res, sigma=sigma, s_inliers=s_pts[inside] - s_a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ply")
    ap.add_argument("--out", default=None, help="output folder (default: results/<name>_tube_fit)")
    ap.add_argument("--units", choices=["auto", "mm", "m"], default="auto", help="units of the PLY")
    ap.add_argument("--radius-mm", type=float, default=None, help="cable radius if known (else estimated)")
    ap.add_argument("--seed", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="a point on the cable, in PLY units (else the longest tube is used)")
    ap.add_argument("--pick", action="store_true", help="click the seed point in a window")
    ap.add_argument("--nodes", type=int, default=151, help="centreline nodes to export")
    ap.add_argument("--max-gap-mm", type=float, default=20.0, help="longest hidden stretch to bridge")
    ap.add_argument("--ctrl-spacing-mm", type=float, default=8.0, help="B-spline control point spacing")
    ap.add_argument("--min-bend-radii", type=float, default=4.0,
                    help="ends bending tighter than this many cable radii are cut off")
    ap.add_argument("--show", action="store_true", help="open a 3-D view of the result")
    args = ap.parse_args()

    P, N, C, scale = load_cloud(args.ply, args.units)
    print(f"{args.ply}: {len(P)} points, units {'mm' if scale == 1.0 else 'm'}, "
          f"extent {np.ptp(P, axis=0).round(0)} mm")
    if N is None:
        print("no normals in the file: estimating them")
        N = estimate_normals(P, 2.0)
    seed = None
    if args.pick:
        seed = pick_seed(P)
    elif args.seed is not None:
        seed = np.asarray(args.seed) * scale

    res = fit_cable(P, N, args.radius_mm, seed, max_gap=args.max_gap_mm,
                    ctrl_spacing=args.ctrl_spacing_mm, min_bend_radii=args.min_bend_radii)
    s_a, s_b = res["s_range"]
    s, X, T = resample(res["spline"], s_a, s_b, args.nodes)
    _, dense, _ = resample(res["spline"], s_a, s_b, int((s_b - s_a) / 0.5) + 1)
    r, resid, idx = res["radius"], res["residuals"], res["inliers"]
    supported = np.ones(len(s), bool)
    for g0, g1 in res["gaps"]:
        supported &= ~((s > g0) & (s < g1))

    name = os.path.splitext(os.path.basename(args.ply))[0]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", f"{name}_tube_fit")
    os.makedirs(out, exist_ok=True)
    np.savetxt(os.path.join(out, "centerline.csv"),
               np.column_stack([s / 1000.0, X / 1000.0, T, supported]),
               delimiter=",", header="s_m,x_m,y_m,z_m,tx,ty,tz,supported", comments="",
               fmt=["%.6f"] * 7 + ["%d"])
    write_ply(os.path.join(out, "cable_points.ply"), P[idx] / scale, None if C is None else C[idx])
    write_ply(os.path.join(out, "centerline.ply"), dense / scale, np.tile([1.0, 0.0, 0.0], (len(dense), 1)))
    render_preview(os.path.join(out, "preview.png"), P[:: max(1, len(P) // 200000)], P[idx], dense)
    summary = {
        "source": os.path.abspath(args.ply),
        "frame": "scanner frame of the PLY (not aligned to gravity or the robot)",
        "ply_units": "mm" if scale == 1.0 else "m",
        "radius_mm": round(r, 4),
        "diameter_mm": round(2.0 * r, 4),
        "length_mm": round(s_b - s_a, 2),
        "n_points_total": int(len(P)),
        "n_cable_points": int(len(idx)),
        "residual_mm": {"rms": round(float(np.sqrt((resid ** 2).mean())), 4),
                        "median_abs": round(float(np.median(np.abs(resid))), 4),
                        "p95_abs": round(float(np.percentile(np.abs(resid), 95)), 4)},
        "hidden_stretches_mm": [[round(a, 1), round(b, 1)] for a, b in res["gaps"]],
        "end_start_m": (X[0] / 1000.0).round(6).tolist(),
        "end_finish_m": (X[-1] / 1000.0).round(6).tolist(),
        "nodes": int(args.nodes),
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {os.path.normpath(out)}/  (centerline.csv, cable_points.ply, centerline.ply, preview.png, summary.json)")

    if args.show:
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        cloud.paint_uniform_color([0.75, 0.75, 0.75])
        cable = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[idx]))
        cable.paint_uniform_color([0.1, 0.7, 0.3])
        line = o3d.geometry.LineSet(o3d.utility.Vector3dVector(dense),
                                    o3d.utility.Vector2iVector(np.c_[np.arange(len(dense) - 1), np.arange(1, len(dense))]))
        line.paint_uniform_color([1.0, 0.0, 0.0])
        o3d.visualization.draw_geometries([cloud, cable, line])


if __name__ == "__main__":
    main()
