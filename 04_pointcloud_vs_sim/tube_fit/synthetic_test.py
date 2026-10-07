"""Known-answer test for fit_cable_tube: a synthetic scan with a true centreline.

    python synthetic_test.py

Builds a looping cable whose strands almost touch where they cross, keeps only
the surface a scanner would see, adds noise and clutter, runs the fit and
compares the result with the truth. Exits non-zero if the fit is off.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.spatial import cKDTree

from fit_cable_tube import fit_cable, resample

R_TRUE = 3.55          # mm
NOISE_MM = 0.05
NORMAL_NOISE_DEG = 4.0


def true_centreline(n=40000):
    """A loop (prolate cycloid) lifted in z so the crossing strands are 0.3 mm apart."""
    a, b = 40.0, 80.0
    k = (2.0 * R_TRUE + 0.3) / (2.0 * 1.8955)
    phi = np.linspace(-4.0, 4.0, n)
    return np.column_stack([a * phi - b * np.sin(phi), -b * np.cos(phi), k * phi])


def frames(c):
    t = np.gradient(c, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    ref = np.array([0.0, 0.0, 1.0])
    u = np.cross(t, ref)
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return t, u, np.cross(t, u)


def make_scan(rng):
    c = true_centreline()
    t, u, v = frames(c)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))]
    n_pts = int(s[-1] * 2.0 * np.pi * R_TRUE / 0.4 ** 2)
    i = np.searchsorted(s, rng.uniform(0.0, s[-1], n_pts)).clip(0, len(c) - 1)
    th = rng.uniform(0.0, 2.0 * np.pi, n_pts)
    nrm = np.cos(th)[:, None] * u[i] + np.sin(th)[:, None] * v[i]
    pts = c[i] + R_TRUE * nrm

    views = np.array([[0.0, 0.0, 1.0], [0.5, 0.0, 0.866]])
    seen = ((nrm @ views.T) > 0.15).any(axis=1)
    # the upper strand hides the lower one where they cross (seen from +z)
    near = cKDTree(c[:, :2]).query_ball_point(pts[:, :2], R_TRUE)
    hidden = np.array([any(abs(j - ii) > 2000 and c[j, 2] > p[2] for j in nb[::50])
                       for nb, ii, p in zip(near, i, pts)])
    keep = seen & ~hidden
    pts, nrm = pts[keep], nrm[keep]

    # clutter: a flat plate, a fat cylinder (gripper-like), a box edge
    g = np.mgrid[-60:60:0.6, -60:60:0.6].reshape(2, -1).T
    plate = np.column_stack([g[:, 0] - 150.0, g[:, 1], np.full(len(g), -30.0)])
    plate_n = np.tile([0.0, 0.0, 1.0], (len(plate), 1))
    ang, h = rng.uniform(0, np.pi, 30000), rng.uniform(-40, 40, 30000)
    cyl_n = np.column_stack([np.cos(ang), np.zeros_like(ang), np.sin(ang)])
    cyl = 30.0 * cyl_n + np.column_stack([np.full_like(h, -150.0), h, np.full_like(h, 40.0)])
    e = rng.uniform(0, 1, (20000, 2))
    wall = np.column_stack([np.full(len(e), -90.0), e[:, 0] * 120 - 60, e[:, 1] * 60 - 30])
    wall_n = np.tile([1.0, 0.0, 0.0], (len(wall), 1))

    P = np.vstack([pts, plate, cyl, wall])
    N = np.vstack([nrm, plate_n, cyl_n, wall_n])
    P = P + rng.normal(0.0, NOISE_MM, P.shape)
    N = N + rng.normal(0.0, np.radians(NORMAL_NOISE_DEG), N.shape)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, N, c, s[-1], len(pts)


def main():
    rng = np.random.default_rng(0)
    P, N, c_true, len_true, n_cable = make_scan(rng)
    print(f"synthetic scan: {len(P)} points, {n_cable} on the cable, true length {len_true:.1f} mm")

    res = fit_cable(P, N)
    s_a, s_b = res["s_range"]
    _, X, _ = resample(res["spline"], s_a, s_b, int((s_b - s_a) / 0.5) + 1)
    err = cKDTree(c_true).query(X)[0]
    miss = cKDTree(X).query(c_true[::20])[0]

    report = {
        "centreline error, mean [mm]": (err.mean(), 0.10),
        "centreline error, max [mm]": (err.max(), 0.50),
        "radius error [mm]": (abs(res["radius"] - R_TRUE), 0.05),
        "length error [%]": (100.0 * abs((s_b - s_a) - len_true) / len_true, 1.0),
        "truth not covered, max [mm]": (miss.max(), 5.0),
    }
    ok = True
    for name, (value, limit) in report.items():
        passed = value <= limit
        ok &= passed
        print(f"  {name:32s} {value:8.3f}   limit {limit:5.2f}   {'ok' if passed else 'FAIL'}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
