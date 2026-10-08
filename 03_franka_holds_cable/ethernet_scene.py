"""Franka holds the scanned Ethernet cable: build the scene from a JSON config,
simulate it in Newton (VBD), and export the simulated centreline for comparison.

    # from the repository root
    python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json
    python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --viewer gl
    python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --sweep

Then compare with the scan:

    python 04_pointcloud_vs_sim/compare_to_scan.py --config configs/ethernet_cat6.json

What is simulated (see README.md in this folder for the reasoning):

  * scan    the tube-fit centreline (04_pointcloud_vs_sim/tube_fit), metres, in
            the scanner's own frame. 'Up' is measured from the cable (scan.up)
            and everything is turned so up is +z. Node 0 is the gripper end.
  * cable   a Newton rod built from section rigidities (EI, EA, GJ, kGA) in the
            JSON, so Newton divides by the segment length itself. Its REST shape
            is straight; only its starting pose follows the scan.
  * grip    the first `grip_length_m` of cable (inside the fingers) is clamped
            (zero mass = kinematic) at the scanned position and tangent of the
            gripper end. That is the boundary condition.
  * plug    `plug_length_m` of extra cable past the last scanned node, with the
            plug mass spread over it.
  * table   a ground plane at the height where the cable lies on it.
  * far end with cable.fix_plug_end, the plug segments are held still at their
            scanned place too; otherwise they are free and carry the plug mass.
  * Franka  kinematic, placed with IK so its TCP sits on the grip (robot.grasp
            says how the fingers hold the cable), keeping the arm above the
            table. It does not touch the cable and does not change the result.
  * settle  until the shape stops changing (sim.settle_tol_m over settle_window_s).
"""

from __future__ import annotations

import os
import sys

# macOS workaround (same as 01_franka_in_newton/franka_example.py): with
# DYLD_LIBRARY_PATH pointing at Homebrew, the viewer can die decoding a PNG.
if sys.platform == "darwin" and os.environ.pop("DYLD_LIBRARY_PATH", None):
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton.solvers import SolverVBD

REPO = Path(__file__).resolve().parents[1]
UP = np.array([0.0, 0.0, 1.0])


# =============================================================================
# 1. Config and scan
# =============================================================================
def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    # Strip the "_comment" keys so they never reach the code below.
    def clean(d):
        return {k: clean(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")}
    return clean(cfg)


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


AXES = {"x": (1, 0, 0), "-x": (-1, 0, 0), "y": (0, 1, 0), "-y": (0, -1, 0), "z": (0, 0, 1), "-z": (0, 0, -1)}


def cable_rise(X, dirs):
    """How much the cable climbs again, walking from its higher end, for each direction [m].

    A cable hanging from a gripper and lying on a table only goes DOWN from the
    gripper (plus a few mm where it crosses itself). With a wrong 'up' it climbs a
    lot: a loop lying on the table, seen sideways, is a tall vertical loop.
    X: (N, 3) nodes in order along the cable; dirs: (M, 3) unit vectors.
    Note: up and -up score the same (a cable "hanging upward" is monotone too).
    """
    H = X @ np.atleast_2d(dirs).T  # (N, M) heights
    rev = H[-1] > H[0]  # walk from the higher end
    H[:, rev] = H[::-1, rev]
    return np.clip(np.diff(H, axis=0), 0.0, None).sum(axis=0)


def sphere_directions(n):
    """n roughly evenly spread unit vectors (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * i / n)
    azim = np.pi * (1.0 + 5 ** 0.5) * i
    return np.column_stack([np.cos(azim) * np.sin(polar), np.sin(azim) * np.sin(polar), np.cos(polar)])


def directions_near(axis, max_deg, step_deg):
    """Unit vectors within max_deg of `axis`, on a grid of about step_deg."""
    axis = unit(np.asarray(axis, float))
    a = unit(np.cross(axis, [1.0, 0.0, 0.0] if abs(axis[0]) < 0.9 else [0.0, 1.0, 0.0]))
    b = np.cross(axis, a)
    out = [axis]
    for tilt in np.radians(np.arange(step_deg, max_deg + 1e-9, step_deg)):
        n = max(6, int(round(2 * np.pi * np.sin(tilt) / np.radians(step_deg))))
        for f in np.linspace(0.0, 2 * np.pi, n, endpoint=False):
            out.append(np.cos(tilt) * axis + np.sin(tilt) * (np.cos(f) * a + np.sin(f) * b))
    return np.array(out)


def find_up(X, radius, setting):
    """Direction of 'up' (against gravity) in the scan's frame, and how it was found.

    setting: 'auto', an axis name ('z', '-y', ...) or a vector [x, y, z].
    'auto' MEASURES the scan's orientation from the cable, the way the tube fit
    measures the radius; it never looks at the simulation:
      1. the table: the direction for which the MOST nodes sit at the same, lowest
         height (the stretch resting on the table), with the cable hanging at
         least 5 cm above it and its highest point at one end (the gripper).
         Every direction on the sphere (~1.4 deg apart) is tried. With 'up' even
         slightly wrong, only a few nodes stay at the bottom. Ties: the direction
         along which the cable climbs least (cable_rise);
      2. refine within 6 deg (0.1 deg grid) counting only nodes within 0.5 mm of
         the bottom: nodes really resting on a table are at the same height;
      3. polish: a plane through the nodes resting on the table (refit on those
         within 1 mm), if they spread in two directions.
    Only the lowest stretch is used, so a stiff cable whose loop arches up off
    the table, or a strand lying on another, does not disturb it.
    """
    if isinstance(setting, (list, tuple)):
        return unit(np.asarray(setting, float)), {"method": "given vector"}
    if setting in AXES:
        return np.asarray(AXES[setting], float), {"method": f"given axis {setting}"}
    if setting != "auto":
        raise SystemExit(f"scan.up must be 'auto', an axis like 'z' or '-y', or a vector, got {setting!r}")

    tol = 0.003  # nodes resting on the table: within 3 mm of the lowest one
    dirs = sphere_directions(20000)
    H = X @ dirs.T  # (nodes, directions)
    low, high = H.min(axis=0), H.max(axis=0)
    flat = (H < low + tol).sum(axis=0)
    end_on_top = np.maximum(H[0], H[-1]) > high - 0.02  # the gripper (an end) is the highest point
    ok = (high - low > 0.05) & end_on_top
    if not ok.any():
        ok = high - low > 0.05
    flat_ok = np.where(ok, flat, -1)
    best = flat_ok >= 0.9 * flat_ok.max()  # (nearly) the most nodes on the table ...
    climb = cable_rise(X, dirs)
    up = dirs[int(np.argmin(np.where(best, climb, np.inf)))]  # ... and of those, climbing least

    # Refine: nodes really resting on a table are at the SAME height, so count
    # those within 0.5 mm, on a 0.1 deg grid within 6 deg. (3 mm is too loose for
    # a stiff cable whose loop arches: a tilted plane can graze more of the arch.)
    fine_tol = 0.0005
    near = directions_near(up, 6.0, 0.1)
    Hn = X @ near.T
    flat_fine = (Hn < Hn.min(axis=0) + fine_tol).sum(axis=0)
    best = flat_fine == flat_fine.max()
    up = near[int(np.argmin(np.where(best, cable_rise(X, near), np.inf)))]

    h = X @ up
    lying = X[h < h.min() + fine_tol + 0.0005]
    info = {"method": "auto", "nodes_lying_flat": int(len(lying))}
    if len(lying) >= 3:  # do the resting nodes span an area, or only a line?
        sv0 = np.linalg.svd(lying - lying.mean(0), compute_uv=False)
        info["contact_is_a_line"] = bool(sv0[1] < 0.2 * sv0[0])
    for _ in range(3):  # polish: plane through the resting nodes, then only those within 1 mm of it
        if len(lying) < 10:
            break
        centre = lying.mean(0)
        sv, vt = np.linalg.svd(lying - centre, full_matrices=False)[1:]
        n = vt[2] if vt[2] @ up > 0 else -vt[2]
        if sv[1] < 0.2 * sv[0] or np.degrees(np.arccos(np.clip(n @ up, -1, 1))) > 5.0:
            break  # a line, not a plane, or too far from the estimate: keep the estimate
        up = n
        info["polished_by_table_plane"] = True
        lying = lying[np.abs((lying - centre) @ n) < 0.001]
    info["rise_mm"] = round(1000 * float(cable_rise(X, up)[0]), 1)
    axis = max(AXES, key=lambda k: float(np.dot(up, AXES[k])))
    info["nearest_axis"] = axis
    info["tilt_from_axis_deg"] = round(float(np.degrees(np.arccos(np.clip(up @ AXES[axis], -1, 1)))), 2)
    return up, info


def rotation_to_z(up):
    """Smallest rotation R with R @ up = +z (Rodrigues)."""
    up = unit(up)
    v = np.cross(up, UP)
    c = float(up @ UP)
    if c < -1.0 + 1e-9:  # up = -z: half turn about x
        return np.diag([1.0, -1.0, -1.0])
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def load_scan(cfg):
    """Tube-fit centreline, turned so that up is +z, with node 0 at the gripper end.

    Everything after this works in that gravity-aligned 'world' frame. The
    rotation (scan -> world) is kept so outputs can be turned back into the
    scan's own frame.
    """
    scan_cfg = cfg["scan"]
    csv = repo_path(scan_cfg["centerline_csv"])
    if not csv.exists():
        raise SystemExit(
            f"scan centreline not found: {csv}\n"
            "Run the tube fit first, e.g.\n"
            "  python 04_pointcloud_vs_sim/tube_fit/fit_cable_tube.py Data/Ethernet.ply"
        )
    data = np.genfromtxt(csv, delimiter=",", names=True)
    X = np.column_stack([data["x_m"], data["y_m"], data["z_m"]])
    supported = data["supported"].astype(bool) if "supported" in data.dtype.names else np.ones(len(X), bool)
    # The tube fit's own unit tangents (from its spline): the exact direction the
    # cable leaves the fingers. Much better than differencing nodes, and it
    # matters: 1 deg at the grip moves the hanging part ~6 mm at 35 cm.
    T = np.column_stack([data["tx"], data["ty"], data["tz"]]) if "tx" in data.dtype.names else None

    radius = cfg["cable"].get("radius_m")
    summary = {}
    summary_path = repo_path(scan_cfg["summary_json"]) if scan_cfg.get("summary_json") else None
    if summary_path is not None and summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    if radius is None:
        if "radius_mm" not in summary:
            raise SystemExit("cable.radius_m is null and summary.json has no radius_mm: set the radius in the JSON")
        radius = summary["radius_mm"] / 1000.0

    up, up_info = find_up(X[supported], radius, scan_cfg.get("up", "auto"))
    R = rotation_to_z(up)
    X = X @ R.T  # into the gravity-aligned world frame
    if T is not None:
        T = T @ R.T

    end = scan_cfg.get("gripper_end", "auto")
    if end == "auto":
        end = "start" if X[0, 2] >= X[-1, 2] else "end"
    if end == "end":
        X, supported = X[::-1].copy(), supported[::-1].copy()
        if T is not None:
            T = -T[::-1].copy()  # tangents point along the cable, from the gripper on
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])

    ground_z = scan_cfg.get("ground_z_m")
    if ground_z is None:
        ground_z = float(X[supported, 2].min()) - radius

    return {
        "nodes": X,
        "tangents": T,
        "s": s,
        "supported": supported,
        "radius": float(radius),
        "ground_z": float(ground_z),
        "gripper_end": end,
        "up_scan": up,
        "up_info": up_info,
        "scan_to_world": R,
        "ply_units": summary.get("ply_units", "m"),
        "path": str(csv),
    }


# =============================================================================
# 2. Geometry helpers (numpy only)
# =============================================================================
def unit(v):
    return v / np.linalg.norm(v)


def end_tangent(X, at_start, span=0.02):
    """Average direction over the first/last `span` metres (less noisy than one segment)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    if at_start:
        k = max(1, int(np.searchsorted(s, span)))
        return unit(X[k] - X[0])
    k = min(len(X) - 2, int(np.searchsorted(s, s[-1] - span)))
    return unit(X[-1] - X[k])


def resample(X, n_seg):
    """n_seg + 1 points at equal arc length along polyline X."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], n_seg + 1)
    return np.column_stack([np.interp(t, s, X[:, i]) for i in range(3)])


def quat_from_matrix(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w), Newton/Warp order."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_rotate_z(q):
    """Local +Z axis of each quaternion (x, y, z, w), shape (N, 3)."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.column_stack([2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy)])


# =============================================================================
# 3. The cable path: grip + scan + plug
# =============================================================================
def segment_distances(A0, A1, B0, B1):
    """Closest distance between segments [A0, A1] and [B0, B1] (arrays of pairs), and the closest points."""
    d1, d2, w = A1 - A0, B1 - B0, A0 - B0
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * w).sum(-1), (d2 * w).sum(-1)
    den = a * e - b * b
    s = np.where(den > 1e-18, np.clip((b * f - c * e) / np.maximum(den, 1e-18), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    s = np.where(t < 0.0, np.clip(-c / a, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    pa, pb = A0 + s[:, None] * d1, B0 + t[:, None] * d2
    return np.linalg.norm(pa - pb, axis=1), pa, pb, s, t


def separate_crossings(X, r, margin=0.0003, sigma=0.02, keep_ends=0.02, max_rounds=12):
    """Make the start shape physically possible where the cable crosses itself.

    A real crossing has one strand lying ON the other: centres one diameter
    (2r) apart. The tube fit can put them closer (the touching sides are hidden
    from the scanner). Overlapping strands make self-contact push them apart
    for as long as the simulation runs, so the cable jumps and never settles.

    Here, where two strands are closer than 2r: at a crossing, the upper strand
    is lifted by exactly the missing amount; two strands side by side are
    pushed apart sideways. The change is a smooth bump (Gaussian, sigma 2 cm
    along the cable) and nodes within `keep_ends` of either end never move,
    so the grip and plug boundary conditions are untouched.
    Returns the new nodes and a list of what was changed.
    """
    X = X.copy()
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    movable = np.clip(np.minimum(s - keep_ends, s[-1] - keep_ends - s) / 0.01, 0.0, 1.0)
    need = 2.0 * r + margin
    m = len(X) - 1
    smid = 0.5 * (s[:-1] + s[1:])
    I, J = np.triu_indices(m, 1)
    far = np.abs(smid[J] - smid[I]) > 6.0 * r  # neighbours along the cable always "touch"
    I, J = I[far], J[far]
    changes = []
    for _ in range(max_rounds):
        d, pa, pb, ta, tb = segment_distances(X[I], X[I + 1], X[J], X[J + 1])
        bad = np.flatnonzero(d < need - 1e-5)
        if len(bad) == 0:
            break
        lift = np.zeros(len(X))
        push = np.zeros((len(X), 3))
        for k in bad:
            i, j = I[k], J[k]
            si, sj = s[i] + ta[k] * (s[i + 1] - s[i]), s[j] + tb[k] * (s[j + 1] - s[j])
            gap = pa[k] - pb[k]
            nz = gap[2] / d[k] if d[k] > 1e-6 else 0.0
            if d[k] < 5e-4 or abs(nz) >= 0.3:  # a crossing: lift the upper strand
                upper_s = si if (pa[k][2] > pb[k][2] or (abs(pa[k][2] - pb[k][2]) < 1e-6 and si > sj)) else sj
                horiz2 = d[k] ** 2 * (1.0 - nz * nz)
                delta = -d[k] * abs(nz) + np.sqrt(max(need ** 2 - horiz2, 0.0))
                bump = delta * np.exp(-0.5 * ((s - upper_s) / sigma) ** 2) * movable
                lift = np.maximum(lift, bump)
                changes.append(("lift", upper_s, delta))
            else:  # side by side: push both apart, horizontally
                n = gap.copy()
                n[2] = 0.0
                n /= max(np.linalg.norm(n), 1e-9)
                half = 0.5 * (need - d[k])
                for s0, sign in ((si, 1.0), (sj, -1.0)):
                    bump = half * np.exp(-0.5 * ((s - s0) / sigma) ** 2) * movable
                    stronger = bump > np.linalg.norm(push, axis=1)
                    push[stronger] = sign * bump[stronger, None] * n
                changes.append(("push", si, need - d[k]))
        X[:, 2] += lift
        X += push
    d = segment_distances(X[I], X[I + 1], X[J], X[J + 1])[0]
    left = float(max(0.0, need - margin - d.min())) if len(d) else 0.0
    return X, changes, left


def plan_cable(scan, cable_cfg, sim_cfg, init_mode):
    """Segment layout and the starting centreline of the simulated cable.

    Arc length s is measured from the first scanned node (the edge of the
    fingers): the grip is s in [-grip, 0], the scan s in [0, L_scan], the plug
    after it. The segment length is adjusted so the grip is a whole number of
    segments; the plug absorbs the rounding of the total.
    """
    X = scan["nodes"]
    L_scan = scan["s"][-1]
    separation = None
    if init_mode == "scan" and sim_cfg.get("separate_crossings", True):
        X, changes, left = separate_crossings(X, scan["radius"])
        if changes:
            lifts = [c for c in changes if c[0] == "lift"]
            pushes = [c for c in changes if c[0] == "push"]
            biggest = max(c[2] for c in changes)
            where = sorted({round(1000 * c[1], -1) for c in changes})
            separation = {"lifted": len(lifts), "pushed": len(pushes), "max_change_mm": round(1000 * biggest, 2),
                          "near_s_mm": where[:8], "overlap_left_mm": round(1000 * left, 2)}
            print(f"crossing fix: strands overlapped in the scan; moved apart by up to {1000 * biggest:.1f} mm "
                  f"near s = {', '.join(f'{w:.0f}' for w in where[:6])} mm"
                  + (f" (still {1000 * left:.1f} mm overlap: too close to an end)" if left > 1e-4 else ""))
    grip = float(cable_cfg["grip_length_m"])
    h0 = float(sim_cfg["segment_length_m"])
    n_grip = max(1, round(grip / h0))
    h = grip / n_grip
    n_seg = n_grip + max(2, round((L_scan + float(cable_cfg["plug_length_m"])) / h))
    plug = n_seg * h - grip - L_scan
    n_plug = max(1, round(plug / h)) if cable_cfg["plug_length_m"] > 0 else 0

    if scan.get("tangents") is not None:  # the tube fit's spline tangents
        t0, t1 = unit(scan["tangents"][0]), unit(scan["tangents"][-1])
    else:  # no tangents in the CSV: average direction over the first / last 2 cm
        t0 = end_tangent(X, at_start=True)
        t1 = end_tangent(X, at_start=False)
    grip_start = X[0] - grip * t0

    if init_mode == "scan":
        # Grip straight into the fingers, the scanned curve, the plug straight on.
        # If the crossing fix made the curve a little longer, the plug takes it
        # back, so the total stays n_seg * h (no pre-stretch).
        L_curve = float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())
        plug = n_seg * h - grip - L_curve
        dense = np.vstack([
            grip_start[None],
            X,
            (X[-1] + max(plug, 1e-4) * t1)[None],
        ])
        nodes = resample(dense, n_seg)
    elif init_mode == "drop":
        # Grip as scanned, then the rest straight and horizontal at grip height.
        horiz = t0 - np.dot(t0, UP) * UP
        if np.linalg.norm(horiz) < 1e-6:
            horiz = unit(X.mean(0) - X[0]) - np.dot(unit(X.mean(0) - X[0]), UP) * UP
        horiz = unit(horiz)
        s_rest = np.linspace(0.0, n_seg * h - grip, n_seg - n_grip + 1)
        rest = X[0] + s_rest[:, None] * horiz
        grip_nodes = grip_start + np.linspace(0.0, grip, n_grip + 1)[:, None] * t0
        nodes = np.vstack([grip_nodes[:-1], rest])
    else:
        raise SystemExit(f"sim.init must be 'scan' or 'drop', got {init_mode!r}")

    s_nodes = np.arange(n_seg + 1) * h - grip
    return {
        "nodes": nodes,
        "s": s_nodes,
        "h": h,
        "n_seg": n_seg,
        "n_grip": n_grip,
        "n_plug": n_plug,
        "plug_length": plug,
        "grip_tangent": t0,
        "grip_center": X[0] - 0.5 * grip * t0,
        "length_total": n_seg * h,
        "crossing_fix": separation,
    }


def quat_mul(a, b):
    """Quaternion product a*b, (x, y, z, w), arrays of shape (..., 4)."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def quat_rotvec(q):
    """Rotation vector (axis * angle) of quaternions (x, y, z, w)."""
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    n = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, q[..., 3:4])
    return np.where(n > 1e-12, q[..., :3] / np.maximum(n, 1e-12) * angle, 2.0 * q[..., :3])


def relax_twist(q_start, h, kappa, phi_deg, EI, GJ, n_fixed):
    """Roll of each segment about its own axis that a curled cable at rest would have.

    A centreline scan shows the cable's shape but not how it is twisted. For a
    straight-rest cable that doesn't matter; for a curled one it does: the curl
    points somewhere around the cable, and the twist decides where. The scanned
    cable is at rest, so its twist is the one with the least energy for that
    shape. Minimise, over the roll angles theta_i (the first n_fixed, in the
    gripper, stay 0):

        E = sum_i EI/h |R(-theta_i) b_i - k|^2  +  GJ/h (theta_{i+1} - theta_i)^2

    b_i = bend between segments i and i+1 in the start frames (2-D, parallel
    transport has no twist), k = kappa*h*(cos phi, sin phi) the rest bend.
    Returns theta (n_seg,), in radians.
    """
    from scipy.optimize import minimize

    n = len(q_start)
    rel = quat_mul(q_start[:-1] * np.array([-1.0, -1.0, -1.0, 1.0]), q_start[1:])
    b = quat_rotvec(rel)[:, :2]
    phi = np.radians(phi_deg)
    k = kappa * h * np.array([np.cos(phi), np.sin(phi)])
    wb, wt = EI / h, GJ / h
    free = np.arange(n_fixed, n)

    def energy(x):
        th = np.zeros(n)
        th[free] = x
        c, s_ = np.cos(th[:-1]), np.sin(th[:-1])
        lx, ly = c * b[:, 0] + s_ * b[:, 1], -s_ * b[:, 0] + c * b[:, 1]  # bend seen in the rolled frame
        dx, dy = lx - k[0], ly - k[1]
        dt = np.diff(th)
        e = 0.5 * (wb * (dx * dx + dy * dy).sum() + wt * (dt * dt).sum())
        g = np.zeros(n)
        g[:-1] += wb * (dx * ly - dy * lx)
        g[:-1] -= wt * dt
        g[1:] += wt * dt
        return e, g[free]

    res = minimize(energy, np.zeros(len(free)), jac=True, method="L-BFGS-B", options={"maxiter": 5000})
    theta = np.zeros(n)
    theta[free] = res.x
    return theta


def segment_poses(nodes):
    """Body transforms (p, q) for capsules with body_frame_origin='com' along `nodes`.

    Frames come from Newton's own parallel transport (newton.Rod), so the
    starting pose carries no twist.
    """
    q = np.asarray(newton.Rod(nodes).quaternions, dtype=float)
    p = 0.5 * (nodes[:-1] + nodes[1:])
    return p, q


def nodes_from_bodies(body_q, half_len):
    """Centreline nodes back from capsule transforms (body origin at the segment centre)."""
    p, q = body_q[:, :3], body_q[:, 3:7]
    z = quat_rotate_z(q)
    starts = p - half_len * z
    ends = p + half_len * z
    nodes = np.empty((len(p) + 1, 3))
    nodes[0] = starts[0]
    nodes[1:-1] = 0.5 * (ends[:-1] + starts[1:])
    nodes[-1] = ends[-1]
    return nodes


# =============================================================================
# 4. Franka: kinematic, placed with IK
# =============================================================================
FRANKA_URDF = "urdf/fr3_franka_hand.urdf"
FRANKA_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]


def rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_rotate(q, v):
    """Rotate vectors v (N, 3) by one quaternion q (x, y, z, w)."""
    u, w = np.asarray(q[:3], float), float(q[3])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def base_candidates(cfg_robot, scan, plan):
    """Base poses (position, yaw) to try.

    Given in the JSON: only that one. Automatic: on the table, on the far side of
    the grip from the cable, facing the grip, at a few distances.
    """
    grip = plan["grip_center"]
    if cfg_robot.get("base_xyz_m") is not None:  # given in the scan's own frame
        base = scan["scan_to_world"] @ np.asarray(cfg_robot["base_xyz_m"], float)
        if cfg_robot.get("base_yaw_deg") is not None:
            yaw = np.radians(cfg_robot["base_yaw_deg"])
        else:
            d = grip - base
            yaw = np.arctan2(d[1], d[0])
        return [(base, float(yaw))]
    away = scan["nodes"].mean(0) - grip  # where the cable is
    away[2] = 0.0
    if np.linalg.norm(away) < 1e-6:
        away = -plan["grip_tangent"].copy()
        away[2] = 0.0
    away = unit(away) if np.linalg.norm(away) > 1e-6 else np.array([1.0, 0.0, 0.0])
    d0 = float(cfg_robot.get("base_distance_m", 0.5))
    out = []
    for d in [d0] + [d for d in (0.4, 0.5, 0.6, 0.7) if abs(d - d0) > 1e-6]:
        base = grip - d * away
        base[2] = scan["ground_z"]
        out.append((base, float(np.arctan2(away[1], away[0]))))
    return out


def tcp_target(plan, grasp, flip, toward):
    """TCP pose on the grip: position and rotation matrix [x y z] of fr3_hand_tcp.

    The fingers close along the hand's y axis; the hand's z axis points from the
    wrist out through the fingertips.

      'across'  the finger pads pinch the cable from the sides: the cable runs
                along hand x, and the hand comes from above as much as the cable
                direction allows (like the reference pick-and-place example).
      'along'   the cable comes straight out of the fingertips: hand z along the
                cable. The hand then sits behind the grip, on the side away from
                the cable, which can put it under the table for a low grip or a
                cable leaving upward.
      flip      the same grasp turned 180 deg about hand z (the fingers are
                symmetric, the arm pose is not).
      toward    horizontal unit vector from the robot base to the grip, used when
                the cable is vertical and 'down' is not defined across it.
    """
    t = plan["grip_tangent"]
    if grasp == "along":
        z = t
        y = np.cross(UP, z)
        y = unit(y) if np.linalg.norm(y) > 1e-6 else unit(np.cross(UP, toward))
        x = np.cross(y, z)
    elif grasp == "across":
        x = t
        z = -UP - np.dot(-UP, x) * x  # the most downward direction across the cable
        if np.linalg.norm(z) < 1e-3:  # cable vertical: reach in horizontally from the robot
            z = toward - np.dot(toward, x) * x
        z = unit(z)
        y = np.cross(z, x)
    else:
        raise SystemExit(f"robot.grasp must be 'across' or 'along', got {grasp!r}")
    if flip:
        x, y = -x, -y
    return plan["grip_center"], np.column_stack([x, y, z])


def add_franka(builder, base, yaw):
    xform = wp.transform(wp.vec3(*base), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
    start = builder.body_count
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / FRANKA_URDF,
        xform=xform,
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
    )
    return list(range(start, builder.body_count))


class FrankaIK:
    """A Franka-only model with its base at the origin: IK, and the robot's lowest point.

    IK only places the TCP. Nothing stops the rest of the arm from going through
    the table, so every solution is also checked against the table using the
    robot's real geometry (mesh vertices and box corners of every moving link;
    the base and link0 stand on the table and are left out).
    """

    def __init__(self, device):
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        add_franka(b, np.zeros(3), 0.0)
        b.joint_q[:9] = FRANKA_HOME
        self.model = m = b.finalize(device=device)
        self.device = device
        self.names = [label.split("/")[-1] for label in m.body_label]
        self.tcp = self.names.index("fr3_hand_tcp")
        body, xf = m.shape_body.numpy(), m.shape_transform.numpy()
        kind, scale = m.shape_type.numpy(), m.shape_scale.numpy()
        corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        self.points = []  # (body index, shape points in the body frame)
        bolted = {"base", "fr3_link0"}  # stand on the table: not part of the check
        for i in range(m.shape_count):
            if body[i] < 0 or self.names[body[i]] in bolted:
                continue
            if kind[i] == int(newton.GeoType.MESH):
                v = np.asarray(m.shape_source[i].vertices, float) * scale[i]
            elif kind[i] == int(newton.GeoType.BOX):
                v = corners * scale[i]
            else:
                v = np.zeros((1, 3))
            self.points.append((int(body[i]), xf[i, :3] + quat_rotate(xf[i, 3:7], v)))

    def solve(self, target_p, target_R, finger):
        """Joint angles putting fr3_hand_tcp at (target_p, target_R), both in the base frame."""
        m, device = self.model, self.device
        target_q = quat_from_matrix(target_R)
        n = m.joint_coord_count
        q = wp.array(np.asarray(FRANKA_HOME, np.float32).reshape(1, n), dtype=float, device=device)
        pos_obj = ik.IKObjectivePosition(
            link_index=self.tcp, link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array([wp.vec3(*target_p)], dtype=wp.vec3, device=device))
        rot_obj = ik.IKObjectiveRotation(
            link_index=self.tcp, link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([wp.vec4(*target_q)], dtype=wp.vec4, device=device))
        lim_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=m.joint_limit_lower, joint_limit_upper=m.joint_limit_upper, weight=10.0)
        solver = ik.IKSolver(model=m, n_problems=1, objectives=[pos_obj, rot_obj, lim_obj],
                             lambda_initial=0.05, jacobian_mode=ik.IKJacobianType.ANALYTIC)
        solver.step(q, q, iterations=300)

        q_np = q.numpy().reshape(-1).copy()
        q_np[7:9] = finger
        st = m.state()
        newton.eval_fk(m, wp.array(q_np, dtype=float, device=device), m.joint_qd, st)
        body_q = st.body_q.numpy()
        got = body_q[self.tcp]
        pos_err = float(np.linalg.norm(got[:3] - target_p))
        ang_err = float(np.degrees(2 * np.arccos(min(1.0, abs(np.dot(got[3:7], target_q))))))
        return q_np, pos_err, ang_err, body_q

    def lowest(self, body_q):
        """Lowest point of the robot (base frame z) and the link it belongs to."""
        z_min, link = np.inf, ""
        for b, v in self.points:
            z = float((body_q[b, :3] + quat_rotate(body_q[b, 3:7], v))[:, 2].min())
            if z < z_min:
                z_min, link = z, self.names[b]
        return z_min, link


def place_franka(cfg_robot, scan, plan, finger, device):
    """Pick a base pose and joint angles so the TCP holds the grip and the arm stays above the table.

    Tries every base candidate x both hand flips, keeps the solutions that reach
    the grip, and of those the one with the most clearance above the table. This
    only places the (visual) robot: the cable's grip is the same in every case.
    """
    grasp = cfg_robot.get("grasp", "across")
    solver = FrankaIK(device)
    grip = plan["grip_center"]
    tried = []
    for base, yaw in base_candidates(cfg_robot, scan, plan):
        toward = grip - base
        toward[2] = 0.0
        toward = unit(toward) if np.linalg.norm(toward) > 1e-6 else np.array([1.0, 0.0, 0.0])
        to_base = rot_z(-yaw)  # world -> base frame (the base is only turned about z)
        for flip in (False, True):
            p, R = tcp_target(plan, grasp, flip, toward)
            q, pos_err, ang_err, body_q = solver.solve(to_base @ (p - base), to_base @ R, finger)
            low, link = solver.lowest(body_q)
            tried.append({"base": base, "yaw": yaw, "flip": flip, "q": q, "pos_err": pos_err, "ang_err": ang_err,
                          "clearance": base[2] + low - scan["ground_z"], "link": link,
                          "distance": float(np.linalg.norm((grip - base)[:2]))})
    reached = [c for c in tried if c["pos_err"] < 0.005 and c["ang_err"] < 2.0]
    best = max(reached, key=lambda c: c["clearance"]) if reached else min(tried, key=lambda c: c["pos_err"])

    print(f"Franka: grasp '{grasp}', base {best['distance']:.2f} m from the grip"
          f"{', hand flipped' if best['flip'] else ''}: IK {1000 * best['pos_err']:.1f} mm / "
          f"{best['ang_err']:.1f} deg, lowest point {1000 * best['clearance']:+.0f} mm above the table "
          f"({best['link']}); {len(reached)} of {len(tried)} poses tried reach the grip")
    if not reached:
        print("WARNING: no pose reaches the grip. The robot is only visual and the cable is unaffected; "
              "set robot.base_xyz_m (where your robot stands in the scan frame) or change robot.grasp.")
    elif best["clearance"] < 0.0:
        print(f"WARNING: the Franka goes {-1000 * best['clearance']:.0f} mm below the table ({best['link']}) in every "
              f"pose tried. The cable is unaffected; try robot.grasp "
              f"'{'along' if grasp == 'across' else 'across'}' or set robot.base_xyz_m.")
    info = {"grasp": grasp, "base_xyz_m": best["base"].round(4).tolist(),
            "base_yaw_deg": round(float(np.degrees(best["yaw"])), 2), "hand_flipped": best["flip"],
            "joint_q": np.round(best["q"], 5).tolist(), "ik_pos_err_mm": round(1000 * best["pos_err"], 2),
            "ik_ang_err_deg": round(best["ang_err"], 2), "clearance_above_table_mm": round(1000 * best["clearance"], 1),
            "lowest_link": best["link"], "poses_tried": len(tried), "poses_reaching": len(reached)}
    return best["base"], best["yaw"], best["q"], info


# =============================================================================
# 5. Build the Newton model
# =============================================================================
def build_model(cfg, scan, plan, bend_scale, device, placement=None):
    cab, con, rob = cfg["cable"], cfg["contact"], cfg["robot"]
    r = scan["radius"]
    h = plan["h"]

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverVBD.register_custom_attributes(builder)

    # --- Franka (kinematic) ---------------------------------------------------
    franka_bodies, robot_info = [], {}
    if rob.get("enabled", True):
        if placement is None:
            placement = place_franka(rob, scan, plan, finger=r, device=device)
        base, yaw, q_robot, robot_info = placement
        franka_bodies = add_franka(builder, base, yaw)
        builder.joint_q[:9] = q_robot.tolist()
        builder.joint_target_q[:9] = q_robot.tolist()
        for b in franka_bodies:  # zero mass = kinematic: VBD never moves it
            make_kinematic(builder, b)

    # --- Cable ----------------------------------------------------------------
    EI = cab["bend_rigidity_EI_Nm2"] * bend_scale
    GJ = cab["twist_rigidity_GJ_Nm2"] * bend_scale
    kappa = float(cab.get("rest_curvature_per_m", 0.0))
    phi = float(cab.get("rest_curl_direction_deg", 0.0))
    rigidities = dict(stretch_rigidity=cab["stretch_rigidity_EA_N"], shear_rigidity=cab["shear_rigidity_kGA_N"],
                      bend_rigidity=EI, twist_rigidity=GJ)
    # The rod is built in its REST shape (VBD reads the rest bend from this
    # pose); run() then puts it in its start pose. Newton turns the section
    # rigidities into per-joint stiffness (rigidity / segment length) by itself.
    if kappa > 0.0:
        points, quats = curled_rest_shape(plan["n_seg"], h, kappa, phi)
        rod = newton.Rod(points, quaternions=quats, radius=r, **rigidities)
    else:
        rod = newton.Rod.create_straight(
            start=wp.vec3(*plan["nodes"][0]), direction=wp.vec3(*plan["grip_tangent"]),
            length=plan["length_total"], segment_count=plan["n_seg"], radius=r, **rigidities)
    # Newton's mass = capsule volume x density, and each capsule has two
    # hemispherical caps on top of its length h. Correct the density so the mass
    # per metre is the real one.
    mu_lin = cab["mass_per_length_kg_m"]
    capsule_volume = np.pi * r * r * h + 4.0 / 3.0 * np.pi * r ** 3
    cable_cfg = newton.ModelBuilder.ShapeConfig(
        density=mu_lin * h / capsule_volume, ke=con["ke"], kd=con["kd"], mu=con["mu"],
        margin=0.0, gap=con["gap_m"])
    tau = cab["bend_damping_time_s"]
    cable_bodies, cable_joints = builder.add_rod(
        rod=rod, cfg=cable_cfg, body_frame_origin="com", label="ethernet",
        bend_damping=tau * EI / h, twist_damping=tau * GJ / h, stretch_damping=0.0,
    )
    for b in cable_bodies[: plan["n_grip"]]:  # clamped in the fingers
        make_kinematic(builder, b)
    fix_plug = bool(cab.get("fix_plug_end", False))
    if fix_plug and cfg["sim"]["init"] != "scan":
        print("NOTE: cable.fix_plug_end only works with sim.init 'scan' (the plug must start at its scanned "
              "place); the plug end is left free.")
        fix_plug = False
    if fix_plug:  # the far end is held still where the scan shows it (zero mass = kinematic)
        for b in cable_bodies[-max(1, plan["n_plug"]):]:
            make_kinematic(builder, b)
    elif plan["n_plug"] > 0 and cab["plug_mass_kg"] > 0:
        dm = cab["plug_mass_kg"] / plan["n_plug"]
        for b in cable_bodies[-plan["n_plug"]:]:
            add_mass(builder, b, dm)

    # --- Table ----------------------------------------------------------------
    ground_cfg = newton.ModelBuilder.ShapeConfig(ke=con["ke"], kd=con["kd"], mu=con["mu"], margin=0.0, gap=con["gap_m"])
    ground_shape = builder.add_ground_plane(height=scan["ground_z"], cfg=ground_cfg, label="table")

    builder.color()  # VBD needs the colouring
    model = builder.finalize(device=device)

    masses = model.body_mass.numpy()
    info = {
        "segments": plan["n_seg"], "segment_length_m": h, "grip_segments": plan["n_grip"],
        "plug_segments": plan["n_plug"], "plug_length_m": round(plan["plug_length"], 4),
        "length_total_m": round(plan["length_total"], 4), "radius_m": r,
        "EI_Nm2": EI, "GJ_Nm2": GJ, "bend_scale": bend_scale,
        "rest_curvature_per_m": kappa, "rest_curl_direction_deg": phi,
        "crossing_fix": plan.get("crossing_fix"),
        "per_joint_bend_stiffness_Nm_per_rad": EI / h,
        "mass_free_cable_kg": round(float(masses[cable_bodies[plan["n_grip"]:]].sum()), 5),
        "plug_end_fixed": fix_plug,
        "ground_z_m": scan["ground_z"], "robot": robot_info,
    }
    return model, cable_bodies, franka_bodies, ground_shape, info, placement


def curled_rest_shape(n_seg, h, kappa, phi_deg):
    """Points and frames of a rod whose natural (rest) shape has constant curvature.

    Real Ethernet cable remembers the coil it was wound in. Here every segment
    is turned by kappa*h about the same axis of its own frame,
    a = (cos phi, sin phi, 0), so the rest shape is a circle of radius 1/kappa
    and carries no twist. phi is measured in the cable's frame at the gripper
    (the start frame there: Newton's parallel transport, roll 0): it says to
    which side the cable wants to curl. Only the rest shape changes; the start
    pose still follows the scan, with its twist relaxed for this curl
    (relax_twist), because a scan cannot show twist.
    """
    phi = np.radians(phi_deg)
    a = np.array([np.cos(phi), np.sin(phi), 0.0])
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    quats, points = [], [np.zeros(3)]
    for i in range(n_seg):
        t = kappa * h * (i + 0.5)
        R = np.eye(3) + np.sin(t) * K + (1.0 - np.cos(t)) * K @ K  # rotation by t about a
        quats.append(quat_from_matrix(R))
        points.append(points[-1] + h * R[:, 2])  # segment along its frame's +Z
    return np.array(points), np.array(quats)


def make_kinematic(builder, b):
    builder.body_mass[b] = 0.0
    builder.body_inv_mass[b] = 0.0
    builder.body_inertia[b] = wp.mat33(0.0)
    builder.body_inv_inertia[b] = wp.mat33(0.0)


def add_mass(builder, b, dm):
    """Add mass dm to body b, scaling its inertia by the same factor."""
    m = builder.body_mass[b]
    f = (m + dm) / m
    I = np.array(builder.body_inertia[b], dtype=float).reshape(3, 3) * f
    builder.body_mass[b] = m + dm
    builder.body_inv_mass[b] = 1.0 / (m + dm)
    builder.body_inertia[b] = wp.mat33(*I.reshape(-1))
    builder.body_inv_inertia[b] = wp.mat33(*np.linalg.inv(I).reshape(-1))


def contact_pairs(model, cable_bodies, franka_bodies, self_collision, device):
    """Cable-table pairs, plus cable-cable pairs more than 3 segments apart.

    The Franka is left out: it is kinematic, and its fingers overlap the
    clamped cable on purpose.
    """
    shape_body = model.shape_body.numpy()
    robot = set(franka_bodies)
    seg = {b: i for i, b in enumerate(cable_bodies)}
    keep = []
    for a, c in model.shape_contact_pairs.numpy():
        ba, bc = int(shape_body[a]), int(shape_body[c])
        if ba in robot or bc in robot:
            continue
        if ba in seg and bc in seg:
            if not self_collision or abs(seg[ba] - seg[bc]) <= 3:
                continue
        keep.append((int(a), int(c)))
    return wp.array(np.asarray(keep, np.int32), dtype=wp.vec2i, device=device), len(keep)


# =============================================================================
# 6. Simulate one run
# =============================================================================
def run(cfg, scan, bend_scale, out_dir, viewer_kind="null", cache=None):
    """One simulation. `cache` (a dict) keeps the robot placement between sweep runs:
    the grip is the same for every stiffness, so the IK search is done once."""
    sim = cfg["sim"]
    device = sim.get("device")
    cache = {} if cache is None else cache
    plan = plan_cable(scan, cfg["cable"], sim, sim["init"])
    model, cable_bodies, franka_bodies, ground_shape, info, cache["robot"] = build_model(
        cfg, scan, plan, bend_scale, device, cache.get("robot"))
    device = model.device

    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)  # places the Franka; rod bodies untouched
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)

    # Start pose of the cable (the model itself keeps its rest shape).
    p, q = segment_poses(plan["nodes"])
    kappa = float(cfg["cable"].get("rest_curvature_per_m", 0.0))
    if kappa > 0.0:  # a curled cable: start with the twist it has at rest in this shape
        theta = relax_twist(q, plan["h"], kappa, float(cfg["cable"].get("rest_curl_direction_deg", 0.0)),
                            info["EI_Nm2"], info["GJ_Nm2"], plan["n_grip"])
        q = quat_mul(q, np.column_stack([np.zeros((len(theta), 2)), np.sin(0.5 * theta), np.cos(0.5 * theta)]))
        info["start_twist_deg"] = {"min": round(float(np.degrees(theta.min())), 1),
                                   "max": round(float(np.degrees(theta.max())), 1),
                                   "at_plug": round(float(np.degrees(theta[-1])), 1)}
        print(f"start twist (equilibrium for the curl): {np.degrees(theta[-1]):+.0f} deg at the plug end "
              f"(range {np.degrees(theta.min()):+.0f} .. {np.degrees(theta.max()):+.0f} deg)")
    warn_overlap(p, plan, scan["radius"])
    for st in (state_0, state_1):
        bq = st.body_q.numpy()
        bq[cable_bodies, :3] = p
        bq[cable_bodies, 3:7] = q
        st.body_q.assign(bq)
        bqd = st.body_qd.numpy()
        bqd[cable_bodies] = 0.0
        st.body_qd.assign(bqd)

    pairs, n_pairs = contact_pairs(model, cable_bodies, franka_bodies, cfg["contact"]["self_collision"], device)
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit", shape_pairs_filtered=pairs)
    contacts = pipeline.contacts()
    solver = SolverVBD(model, iterations=int(sim["iterations"]), friction_epsilon=float(sim["friction_epsilon"]),
                       rigid_compliant_alm=True, rigid_contact_history=False)

    viewer = None
    if viewer_kind == "gl":
        viewer = newton.viewer.ViewerGL()
        viewer.set_model(model)
        scan_X = wp.array(scan["nodes"].astype(np.float32), dtype=wp.vec3, device=device)

    fps, substeps = sim["fps"], int(sim["substeps"])
    frame_dt = 1.0 / fps
    dt = frame_dt / substeps
    half = 0.5 * plan["h"]
    window = max(1, round(sim["settle_window_s"] * fps))  # frames between shape checks
    snapshot = nodes_from_bodies(state_0.body_q.numpy()[cable_bodies], half)
    t, frame, drift, settled = 0.0, 0, float("nan"), False
    wall0 = time.time()
    print(f"[{info['segments']} segments of {1000 * plan['h']:.1f} mm, EI {info['EI_Nm2']:.3g} N m^2, "
          f"{n_pairs} contact pairs, device {device}]")

    def render():
        if viewer is not None:
            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.log_lines("scan", scan_X[:-1], scan_X[1:], (1.0, 0.2, 0.2), width=0.003)
            viewer.end_frame()

    while t < sim["max_time_s"]:
        if viewer is not None and not viewer.is_running():
            break
        for _ in range(substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        t += frame_dt
        frame += 1
        render()
        if frame % window:
            continue
        # Settled = the shape stopped changing (largest node displacement over one window).
        nodes = nodes_from_bodies(state_0.body_q.numpy()[cable_bodies], half)
        if not np.all(np.isfinite(nodes)):
            raise SystemExit("simulation blew up (NaN): try more substeps/iterations or a softer contact ke")
        drift = float(np.linalg.norm(nodes - snapshot, axis=1).max())
        snapshot = nodes
        print(f"  t = {t:4.1f} s   shape change over {sim['settle_window_s']} s: {1000 * drift:7.3f} mm")
        if drift < sim["settle_tol_m"] and t >= sim["min_time_s"]:
            settled = True
            break

    nodes = nodes_from_bodies(state_0.body_q.numpy()[cable_bodies], half)
    info.update({"init": sim["init"], "settled": settled, "sim_time_s": round(t, 3),
                 "wall_time_s": round(time.time() - wall0, 1), "final_drift_m_per_window": drift,
                 "arc_length_m": float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()),
                 "scan": scan["path"], "gripper_end": scan["gripper_end"],
                 "up_in_scan_frame": np.round(scan["up_scan"], 6).tolist(), "up_detection": scan["up_info"],
                 "scan_to_world": np.round(scan["scan_to_world"], 9).tolist(),
                 "frame_note": "all CSVs are in the world frame = scan frame turned so up is +z "
                               "(world = scan_to_world @ scan); sim_centerline.ply is in the scan's own frame"})
    info["stretch_percent"] = 100.0 * (info["arc_length_m"] / plan["length_total"] - 1.0)
    write_run(out_dir, plan, nodes, info, scan)
    print(f"  {'settled' if settled else 'NOT settled'} at t = {t:.2f} s ({info['wall_time_s']} s wall), "
          f"stretch {info['stretch_percent']:+.3f} %  ->  {out_dir}")

    if viewer is not None:  # keep showing the final shape until the window is closed
        while viewer.is_running():
            render()
        viewer.close()
    return info


def warn_overlap(centres, plan, r, tol=0.001):
    """Note where two strands of the start shape overlap (closer than one diameter).

    A real cable crossing itself lies ON the other strand (centres ~2r apart).
    If the scan puts them closer, self-contact pushes them apart at the start,
    which shows up as a jump in the first fraction of a second.
    """
    n = len(centres)
    d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=-1)
    d[np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) <= 3] = np.inf  # neighbours along the cable
    i, j = np.unravel_index(np.argmin(d), d.shape)
    if d[i, j] < 2 * r - tol:
        s = 0.5 * (plan["s"][:-1] + plan["s"][1:])
        print(f"NOTE: in the start shape two strands overlap by {1000 * (2 * r - d[i, j]):.1f} mm "
              f"(at s = {1000 * s[i]:.0f} and {1000 * s[j]:.0f} mm). Self-contact will push them apart at the start.")


# =============================================================================
# 7. Output
# =============================================================================
def write_run(out_dir, plan, nodes, info, scan):
    out_dir.mkdir(parents=True, exist_ok=True)
    part = np.where(plan["s"] < -1e-9, 0, np.where(plan["s"] <= scan["s"][-1] + 1e-9, 1, 2))
    header = "s_m,x_m,y_m,z_m,part"
    note = "part: 0 = in the gripper, 1 = scanned stretch, 2 = plug. s = 0 at the first scanned node."
    np.savetxt(out_dir / "sim_centerline.csv", np.column_stack([plan["s"], nodes, part]), delimiter=",",
               header=header, comments="", fmt=["%.6f"] * 4 + ["%d"])
    np.savetxt(out_dir / "init_centerline.csv", np.column_stack([plan["s"], plan["nodes"], part]), delimiter=",",
               header=header, comments="", fmt=["%.6f"] * 4 + ["%d"])
    info["columns_note"] = note
    with open(out_dir / "meta.json", "w") as f:
        json.dump(info, f, indent=2, default=float)
    # Same units as the scan's PLY, so it overlays the scan in CloudCompare.
    scale = 1000.0 if scan["ply_units"] == "mm" else 1.0
    in_scan_frame = resample(nodes, 4 * len(nodes)) @ scan["scan_to_world"]  # world -> scan frame
    write_ply(out_dir / "sim_centerline.ply", in_scan_frame * scale, (0, 120, 255))
    write_as_scan(out_dir / "as_scan", plan, nodes, scan)


def write_as_scan(folder, plan, nodes, scan):
    """The simulated cable in the tube-fit output format (centerline.csv + summary.json).

    Point scan.centerline_csv at it to get a synthetic scan whose stiffness you
    know: the comparison must then find that stiffness (see README, self-test).
    """
    folder.mkdir(exist_ok=True)
    on_scan = (plan["s"] >= -1e-9) & (plan["s"] <= scan["s"][-1] + 1e-9)
    X = nodes[on_scan]  # leave out the part in the fingers and the plug, as the tube fit does
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    # Central differences over the WHOLE cable, so the end tangents are centred on
    # the end nodes (like a spline's), not one-sided.
    T = np.gradient(nodes, axis=0)[on_scan]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    np.savetxt(folder / "centerline.csv", np.column_stack([s, X, T, np.ones(len(X))]), delimiter=",",
               header="s_m,x_m,y_m,z_m,tx,ty,tz,supported", comments="", fmt=["%.6f"] * 7 + ["%d"])
    with open(folder / "summary.json", "w") as f:
        json.dump({"radius_mm": 1000.0 * scan["radius"], "ply_units": "m", "length_mm": 1000.0 * s[-1],
                   "frame": "synthetic scan written by ethernet_scene.py"}, f, indent=2)


def write_ply(path, points, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def run_label(bend_scale):
    return f"bend_x{bend_scale:g}"


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True, help="JSON file, e.g. configs/ethernet_cat6.json")
    ap.add_argument("--bend-scale", type=float, default=1.0, help="multiplier on EI and GJ for a single run")
    ap.add_argument("--sweep", action="store_true", help="run every sweep.bend_scale value from the JSON")
    ap.add_argument("--viewer", choices=["null", "gl"], default="null", help="'gl' opens a window (single run only)")
    args = ap.parse_args()

    wp.config.quiet = True
    cfg = load_config(args.config)
    scan = load_scan(cfg)
    print(f"scan: {len(scan['nodes'])} nodes, {1000 * scan['s'][-1]:.1f} mm, radius {1000 * scan['radius']:.2f} mm, "
          f"gripper at the {scan['gripper_end']} of the CSV, table z = {scan['ground_z']:.4f} m")
    ui, up = scan["up_info"], scan["up_scan"]
    print(f"up in the scan frame: ({up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f})  [{ui['method']}"
          + (f": nearest axis {ui['nearest_axis']}, tilt {ui['tilt_from_axis_deg']} deg, cable climbs "
             f"{ui['rise_mm']} mm, {ui['nodes_lying_flat']} nodes lying flat" if ui["method"] == "auto" else "") + "]")
    if ui["method"] == "auto" and ui.get("contact_is_a_line"):
        print("NOTE: the cable rests on the table along a line only (a stiff loop arching up?). A line does not "
              "fix a plane, so the tilt about it is uncertain by up to a degree or two. If the hanging part looks "
              "tilted, set scan.up yourself (e.g. the table normal from CloudCompare).")
    if ui["method"] == "auto" and ui["nodes_lying_flat"] < 10:
        print(f"WARNING: only {ui['nodes_lying_flat']} nodes rest on a common plane, so 'up' is uncertain. "
              "Check the red scan line in the viewer, or set scan.up.")

    out_root = repo_path(cfg["output_dir"]) / cfg["sim"]["init"]
    scales = cfg["sweep"]["bend_scale"] if args.sweep else [args.bend_scale]
    cache = {}
    for k in scales:
        run(cfg, scan, float(k), out_root / run_label(float(k)), args.viewer if not args.sweep else "null", cache)
    with open(out_root / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
