#!/usr/bin/env python3
"""
Recover a cable's LENGTH from a photograph of it hanging.

WHY THIS EXISTS
---------------
``extract_cable_profile.py`` is the full pipeline: SAM segmentation, clicked
scale points, a fitted curve. It is interactive by design, because getting a
publication-grade profile out of a photograph needs a human in the loop.

This script answers a much narrower question that does not need one: given the
support positions (measured with a tape) and a photograph, what cable LENGTH is
consistent with the sag we can see? The cable in these photos is a charger cable
whose free length between the taped ends was never measured, and length is the
single most important input to every simulation here -- so it has to come from
somewhere. It comes from the picture.

HOW
---
The cable is bright and nearly unsaturated; the door behind it is a saturated
brown. That separates them robustly in HSV without any clicking. For each pixel
column the cable's centreline is the mean row of the bright pixels, which gives
a profile in pixels. The known support separation converts pixels to metres.

Then, for each span, the length is the catenary length whose sag matches the
measured sag -- solved by bisection against the same ``catenary`` module the
benchmark scores against, so the number is consistent with everything else.

WHY THE THREE-SUPPORT PHOTO IS THE INTERESTING ONE
--------------------------------------------------
Its two spans are independent catenaries sharing a pinned point, so each yields
its own length estimate, and they must sum to the same cable that the
two-support photo measures. That is three estimates of one number from two
photographs -- a real consistency check rather than a single unfalsifiable
measurement. It also measures how the arc length actually divides between the
spans, which the solvers and the analytic reference merely ASSUME to be
chord-proportional.

    python3 image_utils/measure_length.py --image images/two_point.jpg \
        --supports 0 0.35
    python3 image_utils/measure_length.py --image images/three_point.jpg \
        --supports 0 0.15 0.35
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "methods"))
import catenary  # noqa: E402


def extract_centreline(path: str, sat_max: int, val_min: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_px, y_px, mask) for the cable's centreline, one point per column.

    The cable is bright (high V) and nearly grey (low S); the door is a
    saturated brown. Thresholding on both is far more robust than brightness
    alone, which would also pick up the specular highlights on the door.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"could not read {path}")
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = ((s < sat_max) & (v > val_min)).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # Pick the cable component by SHAPE, not by area. A glare patch on the door
    # is bright and unsaturated too, and is often LARGER than the cable -- an
    # earlier version selected exactly that and reported a 44 mm sag for a
    # cable that visibly sags 380 mm. What separates them is that a cable is
    # thin and spans the supports, while glare is a blob: the cable fills only
    # a few percent of its bounding box, glare fills most of its own.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        candidates = []
        for i in range(1, n):
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            fill = area / float(max(w * h, 1))
            if fill < 0.35 and w > 0.2 * mask.shape[1]:
                candidates.append((w, i))
        if not candidates:
            raise SystemExit(
                "no thin, wide component found -- every bright region looks like a blob. "
                "Check the --overlay image: the cable may be washed out by glare, or the "
                "thresholds (--sat-max / --val-min) may need adjusting.")
        mask = (labels == max(candidates)[1]).astype(np.uint8)

    return mask


def geodesic_endpoints(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of the cable, found by walking the mask itself.

    WHY NOT THE HORIZONTAL EXTREMES
    -------------------------------
    Taking the leftmost and rightmost mask pixels seems obvious -- the cable is
    taped at both ends, so surely it cannot reach past them. It can. The tape
    clamps the cable's ANGLE as well as its position, so the cable leaves along
    the tape's direction and must bend to join the catenary; with a slack cable
    that bend carries it horizontally PAST its own anchor. The leftmost pixel is
    then a point on the descending cable, not the clamp.

    Worse, the error was asymmetric: one end came from the mask extreme and the
    other from where tracing happened to stop, so the two ends were defined by
    different rules. The resulting frame was tilted enough to put two physically
    level clamps 81 mm apart in height.

    The double-sweep below is the standard way to find the extremities of an
    elongated shape and makes no assumption about orientation: breadth-first
    search from an arbitrary pixel reaches one true end; searching again from
    there reaches the other. Distances are geodesic -- measured ALONG the
    cable -- so a curve that doubles back is handled correctly.
    """
    from collections import deque

    h, w = mask.shape
    solid = mask > 0
    ys, xs = np.nonzero(solid)
    if xs.size == 0:
        raise SystemExit("empty mask")

    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def farthest_from(sy: int, sx: int) -> tuple[int, int]:
        dist = np.full((h, w), -1, np.int32)
        dist[sy, sx] = 0
        q = deque([(sy, sx)])
        last = (sy, sx)
        while q:
            cy, cx = q.popleft()
            last = (cy, cx)
            d = dist[cy, cx] + 1
            for dy, dx in nbrs:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and solid[ny, nx] and dist[ny, nx] < 0:
                    dist[ny, nx] = d
                    q.append((ny, nx))
        return last  # BFS dequeues in nondecreasing distance, so the last is farthest

    y0, x0 = int(ys[0]), int(xs[0])
    ay, ax = farthest_from(y0, x0)
    by, bx = farthest_from(ay, ax)
    return np.array([float(ax), float(ay)]), np.array([float(bx), float(by)])


def trace_centreline(mask: np.ndarray, step: float = 4.0,
                     halfwidth: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Walk along the cable, returning its centreline ordered by ARC LENGTH.

    WHY NOT ONE POINT PER PIXEL COLUMN
    ----------------------------------
    Averaging the mask rows in each column is simple and completely wrong for
    this cable. Where it hangs steeply -- near the supports, and everywhere once
    L/S is large -- a single column spans a long vertical run of cable, and
    collapsing that run to its mean destroys precisely the region where the
    supports are. It also cannot represent a curve that doubles back in x.
    Symptom seen here: an extracted profile that stopped 15 mm short of both
    supports and put their heights 90 mm apart when they are level to 16 mm.

    This instead steps ALONG the curve. From the current point it advances one
    step in the local tangent direction, takes the mask's cross-section
    PERPENDICULAR to that direction, and re-centres on it. A cross-section is
    always narrow regardless of how the cable is oriented, so vertical, diagonal
    and doubling-back sections are all handled identically.

    Args:
        step: advance per iteration [px]. Smaller follows curvature more
            closely; too small and mask noise dominates the direction estimate.
        halfwidth: half-length of the perpendicular probe [px]. Must exceed the
            cable's half-thickness and stay below the gap to any neighbouring
            part of the cable, or the probe jumps between strands.

    Returns (x, y) arrays in pixels, ordered from one end to the other.
    """
    start, end = geodesic_endpoints(mask)
    if end[0] < start[0]:  # trace left to right for a predictable ordering
        start, end = end, start

    h, w = mask.shape
    probe = np.arange(-halfwidth, halfwidth + 1e-9, 1.0)
    widths: list[float] = []  # cable thickness seen at each cross-section [px]

    def cross_section_centre(p: np.ndarray, d: np.ndarray) -> np.ndarray | None:
        """Centroid of the mask run that the perpendicular probe through p hits."""
        n = np.array([-d[1], d[0]])  # unit normal
        qs = p[None, :] + probe[:, None] * n[None, :]
        ix = np.rint(qs[:, 0]).astype(int)
        iy = np.rint(qs[:, 1]).astype(int)
        ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        hit = np.zeros(probe.size, bool)
        hit[ok] = mask[iy[ok], ix[ok]] > 0
        if not hit.any():
            return None
        # Keep only the run containing (or nearest) the probe centre, so a
        # neighbouring strand crossing the probe cannot drag the centroid.
        idx = np.flatnonzero(hit)
        centre = np.argmin(np.abs(idx - probe.size // 2))
        c = idx[centre]
        lo = c
        while lo - 1 in idx and lo - 1 >= 0:
            lo -= 1
        hi = c
        while hi + 1 in idx:
            hi += 1
        run = np.arange(lo, hi + 1)
        widths.append(float(run.size))
        return p + probe[run].mean() * n

    # Initial direction: toward the far endpoint, refined immediately by the
    # first cross-section.
    d = end - start
    d /= np.linalg.norm(d)
    p = start.copy()
    c = cross_section_centre(p, d)
    if c is not None:
        p = c

    path = [p.copy()]
    for _ in range(int(20 * (mask.shape[0] + mask.shape[1]) / step)):
        # Losing the mask usually means the step overshot a tight bend rather
        # than that the cable ended, so shorten the step before giving up.
        c = None
        for shrink in (1.0, 0.5, 0.25):
            c = cross_section_centre(p + shrink * step * d, d)
            if c is not None:
                break
        if c is None:
            break
        new_d = c - p
        nrm = np.linalg.norm(new_d)
        if nrm < 1e-6:
            break
        new_d /= nrm
        # Blend directions: pure re-estimation each step chases mask noise,
        # pure persistence cuts corners. 0.5 tracks curvature without jitter.
        d = 0.5 * d + 0.5 * new_d
        d /= np.linalg.norm(d)
        p = c
        path.append(p.copy())
        if np.linalg.norm(p - end) < step:
            break

    # Do NOT append the far endpoint unconditionally. An earlier version did,
    # and whenever the tracer stopped short it drew a straight line across the
    # gap -- a 586 px jump here, worth 88 mm of arc length that no cable ever
    # occupied, silently inflating the headline measurement. If the trace does
    # not reach the end, that is a detection failure and must be reported, not
    # papered over.
    arr = np.asarray(path)

    # COVERAGE CHECK. Distance to the mask's extreme pixel is NOT a usable test
    # of completeness: a clamped cable bulges horizontally past its own anchor,
    # so a correct trace legitimately stops short of the extreme and the naive
    # test cries wolf on every good run.
    #
    # What does work is conservation of area. A curve of arc length s and
    # thickness t covers about s*t pixels, and the tracer measures t directly at
    # every cross-section. If the traced length explains the mask's area, the
    # trace covered the cable; if the area implies a cable twice as thick as the
    # one actually measured, roughly half of it was missed.
    traced_px = float(np.sum(np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))))
    if widths and traced_px > 0:
        t_seen = float(np.median(widths))
        t_implied = float(mask.sum()) / traced_px
        if t_implied > 1.6 * t_seen:
            print(f"[warn] trace looks INCOMPLETE: the mask's area implies a cable "
                  f"{t_implied:.0f} px thick, but cross-sections measure {t_seen:.0f} px. "
                  f"Roughly {100 * (1 - t_seen / t_implied):.0f}% of the cable was likely "
                  f"missed, so the length is a LOWER BOUND. Try a larger --halfwidth "
                  f"or a smaller --step.")
    return arr[:, 0], arr[:, 1]


def detection_figure(image_path: str, mask: np.ndarray, xm: np.ndarray, ym: np.ndarray,
                     supports: list[float], per_span: list[float], out_path: str) -> None:
    """Two-panel figure showing WHAT was detected and HOW WELL it fits.

    A single number ("L = 0.83 m") asks the reader to trust the extraction. This
    shows it: the left panel is the photograph with every detected cable pixel
    tinted, so a mis-segmentation (glare, shadow, a missed section) is obvious
    at a glance; the right panel puts the extracted profile against the fitted
    catenary, so a good segmentation that still fits badly is equally obvious.
    Those are different failures and each needs its own panel.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SERIES_1, SERIES_2 = "#2a78d6", "#008300"
    INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"

    bgr = cv2.imread(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Tint detected pixels rather than replacing them, so the cable underneath
    # stays visible and the reader can judge the fit of the mask to the object.
    tint = rgb.copy()
    sel = mask.astype(bool)
    tint[sel] = (0.35 * rgb[sel] + 0.65 * np.array([234, 88, 12])).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4),
                                   gridspec_kw={"width_ratios": [1.05, 1]})

    ax1.imshow(tint)
    ax1.set_title("1. Detected cable pixels", color=INK, fontsize=11.5,
                  weight="bold", loc="left", pad=10)
    ax1.set_xlabel(f"{int(sel.sum()):,} pixels segmented  (HSV: low saturation, high value)",
                   color=INK_2, fontsize=9)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for s in ax1.spines.values():
        s.set_color(GRID)

    # Right: metric profile vs the fitted catenary, per span.
    ax2.plot(xm, ym, ".", ms=1.4, color=SERIES_1, label="extracted centreline", zorder=3)
    residuals = []
    for i in range(len(supports) - 1):
        a0, b0 = supports[i] - supports[0], supports[i + 1] - supports[0]
        if i >= len(per_span):
            break
        sol = catenary.solve(b0 - a0, per_span[i], 0.0, 2)
        gx = np.linspace(a0, b0, 400)
        seg = (xm >= a0 - 1e-9) & (xm <= b0 + 1e-9)
        if seg.sum() < 5:
            continue
        # Align the analytic span to the measured ends before comparing: the
        # fit determines the SHAPE, the photograph determines where it sits.
        top = max(ym[seg][0], ym[seg][-1])
        gz = sol.z(gx - a0) + top
        ax2.plot(gx, gz, "-", lw=2.0, color=SERIES_2, zorder=4,
                 label="fitted catenary" if i == 0 else None)
        resid = np.interp(xm[seg], gx, gz) - ym[seg]
        residuals.append(resid)

    # Nearest traced point, not np.interp: the trace is ordered by ARC LENGTH,
    # so its x is not guaranteed monotonic and np.interp would silently return
    # nonsense wherever it is not.
    for k, sx in enumerate(supports):
        target = sx - supports[0]
        j = int(np.argmin(np.abs(xm - target)))
        ax2.plot(xm[j], ym[j], "o", ms=9, color=INK, zorder=6,
                 label="supports (measured)" if k == 0 else None)

    ax2.set_aspect("equal", "box")
    ax2.grid(True, color=GRID, lw=0.6)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax2.spines[side].set_color(GRID)
    ax2.tick_params(colors=INK_2, labelsize=9)
    ax2.set_xlabel("x [m]", color=INK_2)
    ax2.set_ylabel("z [m]", color=INK_2)
    rms = float(np.sqrt(np.mean(np.concatenate(residuals) ** 2))) * 1e3 if residuals else float("nan")
    ax2.set_title(f"2. Profile vs catenary   (RMS residual {rms:.1f} mm)",
                  color=INK, fontsize=11.5, weight="bold", loc="left", pad=10)
    leg = ax2.legend(frameon=False, fontsize=9, loc="lower center")
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.suptitle(f"Cable detection: {os.path.basename(image_path)}",
                 color=INK, fontsize=13, weight="bold", x=0.008, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)


def length_from_sag(span: float, sag: float) -> float:
    """Cable length whose catenary over `span` sags by `sag`. Bisection, because
    the closed form inverts a transcendental and this is exact enough."""
    lo, hi = span * 1.0000001, span * 50.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if catenary.solve(span, mid, 1.0, 2).sag < sag:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True)
    p.add_argument("--supports", type=float, nargs="+", required=True,
                   help="support x positions in metres, left to right, e.g. 0 0.15 0.35")
    p.add_argument("--height", type=float, default=1.0,
                   help="support height [m], used to place the exported profile in the "
                        "same frame as the simulations")
    p.add_argument("--sat-max", type=int, default=70, help="max HSV saturation for cable pixels")
    p.add_argument("--val-min", type=int, default=150, help="min HSV value for cable pixels")
    p.add_argument("--overlay", default=None, help="write a raw PNG of the detection mask")
    p.add_argument("--figure", default=None,
                   help="write a two-panel detection + fit figure (the one to show people)")
    p.add_argument("--profile-csv", default=None,
                   help="write the extracted profile as x_m,z_m -- the format the "
                        "comparison tools consume as the REAL cable")
    p.add_argument("--step", type=float, default=4.0,
                   help="tracer advance per iteration [px]")
    p.add_argument("--halfwidth", type=float, default=40.0,
                   help="half-length of the tracer's perpendicular probe [px]; must "
                        "exceed the cable's half-thickness but stay under the gap to "
                        "any neighbouring strand")
    args = p.parse_args()

    supports = sorted(args.supports)
    if len(supports) not in (2, 3):
        raise SystemExit("--supports needs 2 or 3 positions")
    span_m = supports[-1] - supports[0]

    mask = extract_centreline(args.image, args.sat_max, args.val_min)
    x, y = trace_centreline(mask, step=args.step, halfwidth=args.halfwidth)
    if x.size < 50:
        raise SystemExit("cable not found -- try raising --sat-max or lowering --val-min")

    # TWO-POINT FRAME CALIBRATION.
    #
    # The trace now starts and ends at the cable's true extremities (the clamps),
    # found geodesically. Those two points carry all the calibration needed:
    #
    #   SCALE       their separation is the measured span.
    #   HORIZONTAL  the clamps are level in reality, so the line joining them IS
    #               the horizontal. Adopting it removes any camera roll, which
    #               would otherwise tilt the whole profile and make one clamp
    #               appear higher than the other.
    #
    # Using the image's own axes instead assumes the camera was perfectly level,
    # which it never quite is. The residual tilt here is small in degrees but
    # large in millimetres across a 350 mm span.
    p_left = np.array([x[0], y[0]])
    p_right = np.array([x[-1], y[-1]])
    d = p_right - p_left
    chord_px = float(np.hypot(*d))
    scale = span_m / chord_px
    tilt_rad = np.arctan2(d[1], d[0])  # image y grows downward

    cos_t, sin_t = np.cos(-tilt_rad), np.sin(-tilt_rad)
    rx = (x - p_left[0]) * cos_t - (y - p_left[1]) * sin_t
    ry = (x - p_left[0]) * sin_t + (y - p_left[1]) * cos_t

    xm = rx * scale          # metres along the clamp-to-clamp line
    ym = -ry * scale         # metres, up positive

    print(f"image      : {os.path.basename(args.image)}")
    print(f"clamps px  : ({p_left[0]:.0f}, {p_left[1]:.0f}) -> ({p_right[0]:.0f}, {p_right[1]:.0f})")
    print(f"scale      : {scale * 1e3:.4f} mm/px   (from a {span_m:.3f} m clamp separation)")
    print(f"frame tilt : {np.degrees(tilt_rad):+.2f} deg, corrected "
          f"(clamps are level by construction)")
    print(f"end levels : left z = {ym[0] * 1e3:+.1f} mm, right z = {ym[-1] * 1e3:+.1f} mm "
          f"(difference {abs(ym[0] - ym[-1]) * 1e3:.1f} mm)")
    print()

    total = 0.0
    per_span: list[float] = []
    for i in range(len(supports) - 1):
        a, b = supports[i] - supports[0], supports[i + 1] - supports[0]
        sel = (xm >= a - 1e-9) & (xm <= b + 1e-9)
        if sel.sum() < 10:
            print(f"span {i + 1}: too few pixels, skipped")
            continue
        seg_x, seg_y = xm[sel], ym[sel]
        # Support height = the higher of the two span ends; sag is measured down
        # from there. Using the ends rather than the global top keeps a
        # neighbouring span's peak from contaminating this one.
        top = max(seg_y[0], seg_y[-1])
        sag = top - seg_y.min()
        sub_span = b - a
        L = length_from_sag(sub_span, sag)
        total += L
        per_span.append(L)
        print(f"span {i + 1}: x {a:.3f}..{b:.3f} m (S={sub_span:.3f} m)  "
              f"sag={sag * 1e3:6.1f} mm  ->  L={L:.4f} m")

    print()
    print(f"TOTAL implied cable length = {total:.4f} m   (L/S = {total / span_m:.3f})")

    # Arc length straight off the traced pixels: independent of any catenary
    # assumption, so a large disagreement means the fit is being asked to
    # describe a shape that is not a catenary.
    #
    # The tracer already returns points ordered along the curve, so this is a
    # simple sum over consecutive points -- no sorting by x, which would be
    # wrong the moment the cable doubles back.
    #
    # Smoothing first still matters: the tracer's re-centring jitters by a
    # fraction of a pixel, and summing hypot() over thousands of steps turns
    # that into millimetres of fictitious length, because noise only ever adds.
    sx, sy = xm, ym
    win = max(5, int(0.01 * sx.size) | 1)  # ~1% of the path, forced odd
    kern = np.ones(win) / win
    sy_s = np.convolve(sy, kern, mode="same")
    sx_s = np.convolve(sx, kern, mode="same")
    sy_s[:win], sy_s[-win:] = sy[:win], sy[-win:]  # convolution edge artefacts
    sx_s[:win], sx_s[-win:] = sx[:win], sx[-win:]
    raw = float(np.sum(np.hypot(np.diff(sx_s), np.diff(sy_s))))
    raw_unsmoothed = float(np.sum(np.hypot(np.diff(sx), np.diff(sy))))
    print(f"traced polyline arc length = {raw:.4f} m   "
          f"(model-free check; differs by {raw - total:+.4f} m = {(raw - total) * 1e3:+.1f} mm)")
    print(f"  (unsmoothed trace would read {raw_unsmoothed:.4f} m; "
          f"{(raw_unsmoothed - raw) * 1e3:.1f} mm of that is pixel jitter)")

    if len(supports) == 3 and len(per_span) == 2:
        chord = [(supports[i + 1] - supports[i]) / span_m for i in range(2)]
        meas = [L / total for L in per_span]
        print()
        print("ARC-LENGTH SPLIT between the two spans (left / right):")
        print(f"  measured from the photo          : {meas[0] * 100:.1f}% / {meas[1] * 100:.1f}%")
        print(f"  chord-proportional (code assumes): {chord[0] * 100:.1f}% / {chord[1] * 100:.1f}%")
        delta = (meas[0] - chord[0]) * 100
        print(f"  difference                       : {delta:+.1f} percentage points")
        if abs(delta) > 1.0:
            print()
            print("  The real cable does NOT divide chord-proportionally. Where the pinch")
            print("  grips is an experimental fact, not something statics decides, so a")
            print("  simulation using the chord-proportional default is solving a")
            print("  different problem. Pass --mid-fraction "
                  f"{meas[0]:.4f} to match this photograph.")

    if args.profile_csv:
        # Emit with the supports at the stated height, so the real profile
        # shares a coordinate frame with the simulations and can be overlaid
        # without any further alignment step.
        import csv as _csv
        z_at_supports = float(np.mean([ym[int(np.argmin(np.abs(xm - (s - supports[0]))))]
                                       for s in (supports[0], supports[-1])]))
        with open(args.profile_csv, "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["x_m", "z_m"])
            for xi, zi in zip(xm, ym - z_at_supports + args.height):
                w.writerow([f"{xi:.6f}", f"{zi:.6f}"])
        print(f"\n[out] {args.profile_csv}  ({xm.size} points)")

    if args.figure:
        detection_figure(args.image, mask, xm, ym, supports, per_span, args.figure)
        print(f"\n[out] {args.figure}")

    if args.overlay:
        bgr = cv2.imread(args.image)
        bgr[mask.astype(bool)] = (0, 0, 255)
        for sx in supports:
            px = int(x_lo + (sx - supports[0]) / scale)
            cv2.line(bgr, (px, 0), (px, bgr.shape[0]), (0, 255, 0), 6)
        cv2.imwrite(args.overlay, bgr)
        print(f"\n[out] {args.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
