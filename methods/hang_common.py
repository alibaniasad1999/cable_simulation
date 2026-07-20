"""
Shared scenario definition and result-writing protocol for every method in the
cable-hang benchmark.

Deliberately free of Isaac / Newton / Warp imports: each method script runs
under a different interpreter (Isaac Sim's ``python.sh`` for the engine-backed
methods, a plain venv for the pure-Warp one), and all of them import this.

THE SCENARIO.  A cable of length L is held at 2 or 3 supports, all at the SAME
height H, evenly spaced across a horizontal span S in the x-z plane:

    2 supports:  x = 0,  x = S
    3 supports:  x = 0,  x = S/2,  x = S

The driver (``run_benchmark.py``) writes the scenario once to ``run.json`` and
passes it to each method with ``--scenario``, so every method provably solves
the identical problem -- no chance of two methods drifting apart through
separately-parsed environment variables.

THE OUTPUT PROTOCOL.  Each method writes into its own directory:

    profile.csv     settled shape, columns ``x_m,z_m``
                    -> read by image_utils CableProfile.read_csv
    trajectory.csv  wide time series, ``t, n000_x.., n000_z..``
                    -> read by image_utils load_sim_profile(frame=...)
    meta.json       method label, timings, convergence, arc length, node count
    profile.png     the settled shape against the analytic catenary

The two CSV formats are exactly the ones ``image_utils/extract_cable_profile.py``
already consumes, so a real photo profile can be compared to any method with

    python image_utils/extract_cable_profile.py compare \\
        --real results/<img>/profile.csv \\
        --sim  results/<run>/<method>/trajectory.csv
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np

import catenary


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """One benchmark configuration -- the same problem for every method."""

    length: float = 1.0        # [m]  cable length
    span: float = 0.8          # [m]  distance between the outer supports
    height: float = 1.0        # [m]  common support height
    num_points: int = 2        # 2 or 3 supports
    num_segments: int = 60     # discretisation (nodes = num_segments + 1)
    max_time: float = 8.0      # [s]  sim-time cap before giving up
    settle_vel: float = 2.0e-3 # [m/s] settled when max node speed drops below
    min_time: float = 1.0      # [s]  never declare settled before this
    settle_samples: int = 3    # consecutive quiet samples required to settle
    gravity: float = 9.81      # [m/s^2]
    seed: int = 0
    # [m] middle support position measured from the LEFT support, 3-point only.
    # None = the midpoint, span/2, which is what every earlier run used, so an
    # old run.json without this field still deserialises to its old geometry.
    mid_x: float | None = None

    # -- supports --------------------------------------------------------
    @property
    def support_x(self) -> list[float]:
        """Support x positions, left to right, starting at 0."""
        if self.num_points == 2:
            return [0.0, self.span]
        return [0.0, self.mid_x if self.mid_x is not None else 0.5 * self.span, self.span]

    @property
    def supports(self) -> np.ndarray:
        """(num_points, 3) support positions in cable order, y = 0."""
        return np.array([[x, 0.0, self.height] for x in self.support_x])

    @property
    def mid_fraction(self) -> float | None:
        """Fraction of the cable's ARC LENGTH lying left of the middle support.

        The middle support pins a material point, so this split is a property
        of the experiment rather than something statics determines. The
        convention -- here, in ``initial_polyline`` and in ``catenary.solve``
        -- is chord-proportional, and all three must agree or the analytic
        reference would describe a different cable from the simulated one.
        """
        if self.num_points == 2:
            return None
        return self.support_x[1] / self.span

    @property
    def mid_node(self) -> int | None:
        """Node index pinned at the middle support (None in the 2-point case).

        With the middle support at the midpoint this is num_segments // 2, kept
        exact by forcing num_segments even. Off-centre, the node is the nearest
        one to the chord-proportional arc split, and is clamped away from the
        ends so that both sides keep at least two segments -- a one-segment side
        cannot bend and would report a spurious kink.
        """
        if self.num_points == 2:
            return None
        node = int(round(self.mid_fraction * self.num_segments))
        return min(max(node, 2), self.num_segments - 2)

    def __post_init__(self):
        if self.num_points not in (2, 3):
            raise ValueError(f"num_points must be 2 or 3, got {self.num_points}")
        if self.length <= self.span:
            raise ValueError(
                f"cable length {self.length:.4f} m must exceed the span "
                f"{self.span:.4f} m, otherwise the cable hangs taut with no sag")
        if self.num_points == 3 and self.mid_x is not None:
            if not 0.0 < self.mid_x < self.span:
                raise ValueError(
                    f"middle support x={self.mid_x:.4f} m must lie strictly between "
                    f"the outer supports at 0 and {self.span:.4f} m")
        # Only meaningful for a centred middle support; off-centre the node is
        # rounded anyway, so forcing parity would buy nothing.
        if self.num_points == 3 and self.mid_x is None and self.num_segments % 2:
            self.num_segments += 1

    @property
    def tag(self) -> str:
        return f"{self.num_points}pt"

    def catenary(self) -> catenary.CatenarySolution:
        """The analytic reference shape for this scenario."""
        return catenary.solve(self.span, self.length, self.height,
                              self.num_points, mid_x=self.mid_x)

    # -- serialisation ---------------------------------------------------
    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "Scenario":
        with open(path) as fh:
            return cls(**json.load(fh))

    def describe(self) -> str:
        xs = ", ".join(f"{x:.3f}" for x in self.support_x)
        return (f"scenario {self.tag}: L={self.length:.3f} m over span S={self.span:.3f} m "
                f"(L/S={self.length / self.span:.3f}), supports at x=[{xs}] all z={self.height:.3f} m, "
                f"{self.num_segments} segments")


def add_scenario_args(parser) -> None:
    """Attach the scenario flags to an ``argparse`` parser.

    Shared by the driver and by each method script, so a method can also be run
    standalone with the same flags instead of a ``--scenario`` file.
    """
    g = parser.add_argument_group("scenario")
    g.add_argument("--length", type=float, default=1.0, help="cable length [m]")
    g.add_argument("--span", type=float, default=0.8,
                   help="horizontal distance between the outer supports [m]")
    g.add_argument("--height", type=float, default=1.0,
                   help="support height [m] (all supports share it)")
    g.add_argument("--points", type=int, default=2, choices=[2, 3], dest="num_points",
                   help="number of support points")
    # Measured support positions, as they come off a photograph: the leftmost
    # support is the origin, so only the others need stating. These override
    # --span, which cannot express an off-centre middle support.
    g.add_argument("--x1", type=float, default=None,
                   help="2 supports: x of the far support [m]. "
                        "3 supports: x of the MIDDLE support [m]. "
                        "(the left support is always x=0; overrides --span)")
    g.add_argument("--x2", type=float, default=None,
                   help="3 supports: x of the far support [m] (overrides --span)")
    g.add_argument("--segments", type=int, default=60, dest="num_segments",
                   help="cable discretisation (nodes = segments + 1)")
    g.add_argument("--max-time", type=float, default=8.0,
                   help="sim-time cap [s] before the run is stopped unsettled")
    g.add_argument("--settle-vel", type=float, default=2.0e-3,
                   help="settled when the fastest node drops below this [m/s]")


def resolve_supports(args) -> tuple[float, float | None]:
    """Turn the CLI's support flags into ``(span, mid_x)``.

    ``--x1``/``--x2`` describe supports the way a photograph gives them --
    leftmost at the origin, the rest measured from it -- while ``--span``
    describes only the outer separation and forces any middle support to the
    midpoint. Both are accepted; the explicit positions win.

        2 supports:  --x1 X          -> span = X
        3 supports:  --x1 M --x2 F   -> span = F, middle at M
                     --x1 M          -> span from --span, middle at M

    Raises SystemExit with an actionable message on an impossible layout,
    rather than letting it surface later as a catenary that will not solve.
    """
    x1 = getattr(args, "x1", None)
    x2 = getattr(args, "x2", None)
    span, mid_x = args.span, None

    if args.num_points == 2:
        if x2 is not None:
            raise SystemExit("--x2 needs --points 3 (a 2-support cable has no middle support)")
        if x1 is not None:
            span = x1
    else:
        if x2 is not None:
            span = x2
        if x1 is not None:
            mid_x = x1
        if mid_x is not None and not 0.0 < mid_x < span:
            raise SystemExit(
                f"--x1 {mid_x} must lie strictly between the outer supports at 0 and {span}. "
                f"With 3 supports --x1 is the MIDDLE one and --x2 the far one, so --x1 "
                f"must be the smaller of the two.")

    if span <= 0.0:
        raise SystemExit(f"support span must be positive, got {span}")
    return span, mid_x


def scenario_from_args(args) -> Scenario:
    """Build a Scenario from ``--scenario file`` if given, else from the flags."""
    path = getattr(args, "scenario", None)
    if path:
        return Scenario.load(path)
    span, mid_x = resolve_supports(args)
    return Scenario(
        length=args.length,
        span=span,
        height=args.height,
        num_points=args.num_points,
        num_segments=args.num_segments,
        max_time=args.max_time,
        settle_vel=args.settle_vel,
        mid_x=mid_x,
    )


def method_parser(description: str):
    """Standard argument parser shared by every method script."""
    import argparse

    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scenario", default=None,
                   help="path to run.json (overrides the scenario flags below)")
    p.add_argument("--out", default="results/adhoc",
                   help="output directory for this method's results")
    p.add_argument("--headless", action="store_true", default=None,
                   help="run without a viewport (default: headless)")
    p.add_argument("--gui", dest="headless", action="store_false",
                   help="show the Isaac Sim viewport")
    p.add_argument("--average-window", type=float, default=0.5,
                   help="average the reported shape over this many final seconds, "
                        "to remove residual low-amplitude swing")
    add_scenario_args(p)
    return p


# ---------------------------------------------------------------------------
# Initial geometry
# ---------------------------------------------------------------------------
def initial_polyline(scenario: Scenario) -> np.ndarray:
    """Starting cable shape: the analytic catenary, resampled to equal arc
    length between nodes.

    Starting ON the analytic solution rather than on a guessed sine sag means
    every solver begins in (near) equilibrium, so what the benchmark measures
    is how far each solver DRIFTS from equilibrium -- not how well it recovers
    from an arbitrary initial transient. It also settles far faster.

    Returns:
        ``(num_segments + 1, 3)`` node positions in the x-z plane, y = 0, with
        equal arc-length spacing and total arc length equal to the cable length.
    """
    sol = scenario.catenary()
    n = scenario.num_segments

    # Dense sample -> cumulative arc length -> invert onto an equal-arc grid.
    dense_x, dense_z = sol.sample(20001)
    seg = np.hypot(np.diff(dense_x), np.diff(dense_z))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, s[-1], n + 1)

    x = np.interp(targets, s, dense_x)
    z = np.interp(targets, s, dense_z)

    pts = np.zeros((n + 1, 3))
    pts[:, 0] = x
    pts[:, 2] = z

    # Pin the constrained nodes exactly onto their supports: interpolation can
    # leave them a few microns off, and the solvers weld there.
    pts[0] = scenario.supports[0]
    pts[-1] = scenario.supports[-1]
    if scenario.mid_node is not None:
        pts[scenario.mid_node] = scenario.supports[1]
    return pts


def arc_length(nodes: np.ndarray) -> float:
    """Total polyline arc length [m]."""
    return float(np.sum(np.linalg.norm(np.diff(np.asarray(nodes), axis=0), axis=1)))


# ---------------------------------------------------------------------------
# Settling detection
# ---------------------------------------------------------------------------
class SettleMonitor:
    """Tracks node motion between samples and decides when the cable is at rest.

    Settling requires ``scenario.settle_samples`` CONSECUTIVE quiet samples, not
    one. An oscillating cable passes through zero velocity twice per period, so
    a single-sample test reports "settled" at the turning points of a cable that
    is still swinging happily -- which is exactly the failure mode the Newton
    VBD run showed at low iteration counts (max|v| alternating between 4e-3 and
    2.6e-2 m/s while the single-sample test fired on the low ones).

    Also records the full trajectory so ``write_outputs`` can emit a real time
    series rather than a single settled row.
    """

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.times: list[float] = []
        self.frames: list[np.ndarray] = []
        self.speeds: list[float] = []
        self._prev: np.ndarray | None = None
        self._prev_t: float | None = None
        self.max_speed = float("inf")
        self.settled = False
        self.settled_time: float | None = None
        self._quiet_streak = 0
        self._t0 = time.perf_counter()

    def update(self, t: float, nodes: np.ndarray, record: bool = True) -> bool:
        """Feed a sample; returns True once the cable counts as settled."""
        nodes = np.asarray(nodes, dtype=float)
        if not np.isfinite(nodes).all():
            raise RuntimeError(
                f"solver diverged at t={t:.3f} s (non-finite node positions)")
        if record:
            self.times.append(float(t))
            self.frames.append(nodes.copy())

        if self._prev is not None and t > self._prev_t:
            self.max_speed = float(
                np.linalg.norm(nodes - self._prev, axis=1).max() / (t - self._prev_t))
        self._prev, self._prev_t = nodes.copy(), float(t)
        if record:
            self.speeds.append(self.max_speed)

        if self.max_speed < self.scenario.settle_vel:
            self._quiet_streak += 1
        else:
            self._quiet_streak = 0

        if (not self.settled
                and t >= self.scenario.min_time
                and self._quiet_streak >= self.scenario.settle_samples):
            self.settled = True
            self.settled_time = float(t)
        return self.settled

    @property
    def wall_time(self) -> float:
        return time.perf_counter() - self._t0

    def equilibrium_nodes(self, window: float = 0.5) -> np.ndarray:
        """Node positions time-averaged over the final ``window`` seconds.

        Several methods reach the right shape but keep a small residual swing
        rather than coming fully to rest -- the PhysX capsule chain holds a
        ~1.7 mm pendulum oscillation indefinitely, because a hanging chain's
        lowest mode is barely damped. Reporting the instantaneous final frame
        would then sample a random phase of that swing and add noise to every
        error metric.

        The equilibrium is the mean of a small symmetric oscillation, so
        averaging over a window covering at least one period recovers it
        without artificially over-damping the model (which would distort the
        very dynamics being compared). Falls back to the last frame if the
        window contains nothing.
        """
        if not self.frames:
            raise RuntimeError("no frames recorded")
        t_end = self.times[-1]
        sel = [f for t, f in zip(self.times, self.frames) if t >= t_end - window]
        if not sel:
            return self.frames[-1].copy()
        return np.mean(np.stack(sel), axis=0)

    def report(self, t: float, nodes: np.ndarray) -> str:
        return (f"  t={t:5.2f}s  max|v|={self.max_speed:.2e} m/s  "
                f"min_z={nodes[:, 2].min():.4f} m  arc={arc_length(nodes):.4f} m")


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------
def write_outputs(
    out_dir: str,
    scenario: Scenario,
    monitor: SettleMonitor,
    nodes: np.ndarray,
    method: str,
    label: str,
    extra: dict | None = None,
) -> dict:
    """Write ``profile.csv``, ``trajectory.csv``, ``meta.json`` and ``profile.png``.

    Returns the metadata dict that was written, so the caller can print it.
    """
    os.makedirs(out_dir, exist_ok=True)
    nodes = np.asarray(nodes, dtype=float)
    sol = scenario.catenary()

    # -- profile.csv : settled shape, image_utils CableProfile format --------
    with open(os.path.join(out_dir, "profile.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_m", "z_m"])
        for p in nodes:
            w.writerow([f"{p[0]:.6f}", f"{p[2]:.6f}"])

    # -- trajectory.csv : wide time series, load_sim_profile format ----------
    names = [f"n{i:03d}" for i in range(len(nodes))]
    with open(os.path.join(out_dir, "trajectory.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + [f"{n}_x" for n in names] + [f"{n}_z" for n in names])
        frames = monitor.frames or [nodes]
        times = monitor.times or [0.0]
        for t, fr in zip(times, frames):
            w.writerow([f"{t:.4f}"]
                       + [f"{v:.6f}" for v in fr[:, 0]]
                       + [f"{v:.6f}" for v in fr[:, 2]])

    # -- error against the analytic catenary ---------------------------------
    err = nodes[:, 2] - sol.z(nodes[:, 0])
    sim_sag = scenario.height - float(nodes[:, 2].min())

    meta = {
        "method": method,
        "label": label,
        "scenario": asdict(scenario),
        "num_nodes": int(len(nodes)),
        "settled": bool(monitor.settled),
        "settled_time_s": monitor.settled_time,
        "sim_time_s": float(monitor.times[-1]) if monitor.times else 0.0,
        "wall_time_s": round(monitor.wall_time, 3),
        "final_max_speed_mps": monitor.max_speed,
        "arc_length_m": arc_length(nodes),
        "arc_length_target_m": scenario.length,
        "arc_drift_pct": 100.0 * (arc_length(nodes) - scenario.length) / scenario.length,
        "sag_m": sim_sag,
        "sag_reference_m": sol.sag,
        "sag_error_mm": 1e3 * (sim_sag - sol.sag),
        "rmse_mm": 1e3 * float(np.sqrt(np.mean(err ** 2))),
        "max_abs_error_mm": 1e3 * float(np.max(np.abs(err))),
        "lowest_z_m": float(nodes[:, 2].min()),
    }
    if extra:
        meta["solver"] = extra
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True, default=str)

    _plot_profile(out_dir, scenario, nodes, sol, label, meta)

    print(f"[out] {out_dir}/profile.csv, trajectory.csv, meta.json, profile.png")
    print(f"[result] {label}: "
          f"{'settled' if monitor.settled else 'NOT settled'} at t={meta['sim_time_s']:.2f} s "
          f"({meta['wall_time_s']:.1f} s wall), arc {meta['arc_length_m']:.4f} m "
          f"({meta['arc_drift_pct']:+.2f}%), sag {sim_sag * 1e3:.1f} mm "
          f"(catenary {sol.sag * 1e3:.1f} mm), RMSE {meta['rmse_mm']:.1f} mm")
    return meta


def _plot_profile(out_dir, scenario, nodes, sol, label, meta) -> None:
    """Settled shape vs the analytic catenary. Best-effort: the CSVs are the data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] no profile.png ({exc})")
        return

    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(9, 7), height_ratios=[3, 1], sharex=True)

    cx, cz = sol.sample(400)
    ax.plot(cx, cz, "--", color="0.35", lw=1.8, label="analytic catenary")
    ax.plot(nodes[:, 0], nodes[:, 2], "-", color="royalblue", lw=2.2, label=label)
    sup = scenario.supports
    ax.plot(sup[:, 0], sup[:, 2], "o", color="crimson", ms=9,
            zorder=5, label="supports")
    ax.set_ylabel("z  [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(f"{label}  --  {scenario.tag}, L={scenario.length:.2f} m, "
                 f"S={scenario.span:.2f} m\nRMSE {meta['rmse_mm']:.1f} mm, "
                 f"sag error {meta['sag_error_mm']:+.1f} mm, "
                 f"arc drift {meta['arc_drift_pct']:+.2f}%")

    err_mm = 1e3 * (nodes[:, 2] - sol.z(nodes[:, 0]))
    axe.axhline(0.0, color="0.35", ls="--", lw=1.2)
    axe.plot(nodes[:, 0], err_mm, "-", color="darkorange", lw=1.8)
    axe.fill_between(nodes[:, 0], 0.0, err_mm, color="darkorange", alpha=0.25)
    axe.set_xlabel("x  [m]")
    axe.set_ylabel("error  [mm]")
    axe.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "profile.png"), dpi=150)
    plt.close(fig)
