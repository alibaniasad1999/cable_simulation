"""
Analytic reference: the ideal catenary.

A perfectly flexible (EI = 0), inextensible (EA = inf) cable of length L hung
between supports at the SAME height, separated horizontally by span S, takes
the shape

    z(x) = H + a * (cosh((x - S/2) / a) - cosh(S / (2a)))

where the catenary parameter a = T0 / w  [m] (horizontal tension over weight
per unit length) is fixed by the inextensibility constraint

    L = 2 a sinh(S / (2a)).

Substituting u = S / (2a) turns that into the scalar root problem

    sinh(u) / u = L / S,

whose left side is smooth and strictly increasing on u > 0 from 1 to infinity,
so a solution exists and is unique for any L > S. That is solved below by
bisection -- no SciPy required, and it cannot fail to converge the way a
Newton iteration can when started badly.

THREE-POINT CASE.  Because this benchmark holds every point at the same
height, a cable held at x = 0, S/2, S decouples exactly into two independent
sub-catenaries, each of span S/2 carrying length L/2. No extra theory needed:
the middle support takes the two inner tangents' vertical load, and each half
is the two-point problem again. That symmetry is why "all supports at equal
height" is a genuinely good experimental choice -- both scenarios stay exactly
solvable.

CAVEAT -- read before quoting these numbers as truth.  A real cable has finite
bending stiffness EI. The scale on which bending competes with gravity is the
elasto-gravitational length l = (EI / w)^(1/3). The ideal catenary is the
EI -> 0 limit and is only an accurate description of the true shape when
l << L. ``cable_config.CableProperties.gravito_bending_length`` reports l, and
the benchmark prints l / L so the deviation you should EXPECT between a
correct simulation and this reference is visible up front. Near the supports a
stiff cable is flatter than the catenary over a boundary layer of order l.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# The scalar root problem:  sinh(u) / u = ratio
# ---------------------------------------------------------------------------
def _sinhc(u: float) -> float:
    """sinh(u) / u, numerically safe at u -> 0 (where the limit is 1)."""
    if abs(u) < 1.0e-8:
        return 1.0 + u * u / 6.0
    return math.sinh(u) / u


def solve_shape_parameter(span: float, length: float) -> float:
    """Solve ``L = 2 a sinh(S / 2a)`` for the catenary parameter ``a`` [m].

    Args:
        span: horizontal distance S between the two supports [m].
        length: cable arc length L between them [m]; must exceed the span.

    Returns:
        The catenary parameter a = T0 / w [m]. Large a means a taut, shallow
        cable; small a means a deep, floppy sag.

    Raises:
        ValueError: if the cable is not longer than the span (it would have to
            stretch, which an inextensible catenary cannot do).
    """
    if span <= 0.0:
        raise ValueError(f"span must be > 0, got {span}")
    if length <= span:
        raise ValueError(
            f"cable length {length:.4f} m must exceed the span {span:.4f} m "
            "for a hanging catenary (the cable would be taut / stretched)")

    ratio = length / span

    # Bracket the root of sinhc(u) = ratio. sinhc is strictly increasing, so
    # doubling until we overshoot is guaranteed to find an upper bound.
    lo, hi = 0.0, 1.0
    while _sinhc(hi) < ratio:
        hi *= 2.0
        if hi > 1.0e6:  # ratio ~ 1e6 means a length ~1e5 x the span
            raise ValueError(f"length/span = {ratio:.3e} is unreasonably large")

    for _ in range(200):  # bisection to ~1e-15 relative on a 1e6 bracket
        mid = 0.5 * (lo + hi)
        if _sinhc(mid) < ratio:
            lo = mid
        else:
            hi = mid
    u = 0.5 * (lo + hi)
    return span / (2.0 * u)


# ---------------------------------------------------------------------------
# One catenary span
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CatenarySpan:
    """One analytic catenary between two equal-height supports."""

    x0: float      # [m] left support x
    span: float    # [m] horizontal distance to the right support
    height: float  # [m] common support height
    length: float  # [m] cable arc length in this span
    a: float       # [m] catenary parameter T0 / w

    def z(self, x: np.ndarray | float) -> np.ndarray | float:
        """Height [m] at horizontal position(s) x [m]."""
        xi = (np.asarray(x, dtype=float) - self.x0 - 0.5 * self.span) / self.a
        return self.height + self.a * (np.cosh(xi) - math.cosh(0.5 * self.span / self.a))

    @property
    def sag(self) -> float:
        """Vertical drop [m] from the supports to the lowest point."""
        return self.a * (math.cosh(0.5 * self.span / self.a) - 1.0)

    @property
    def lowest_z(self) -> float:
        return self.height - self.sag

    def horizontal_tension(self, weight_per_length: float) -> float:
        """T0 = w a [N] -- constant along the whole cable."""
        return weight_per_length * self.a

    def max_tension(self, weight_per_length: float) -> float:
        """Tension at the supports [N], where the cable carries the most load."""
        return weight_per_length * self.a * math.cosh(0.5 * self.span / self.a)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """`n` points sampled uniformly in x across the span."""
        x = np.linspace(self.x0, self.x0 + self.span, n)
        return x, np.asarray(self.z(x))


# ---------------------------------------------------------------------------
# Full scenario (2 or 3 equal-height supports)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CatenarySolution:
    """The analytic shape for a whole 2- or 3-support scenario."""

    spans: list[CatenarySpan]
    total_span: float
    total_length: float
    height: float

    @property
    def num_supports(self) -> int:
        return len(self.spans) + 1

    @property
    def sag(self) -> float:
        """Deepest drop below the support height, over all spans [m]."""
        return max(s.sag for s in self.spans)

    @property
    def a(self) -> float:
        """Catenary parameter [m] (identical across spans by symmetry)."""
        return self.spans[0].a

    def z(self, x: np.ndarray | float) -> np.ndarray:
        """Height at x, piecewise across the spans."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x)
        for i, s in enumerate(self.spans):
            lo, hi = s.x0, s.x0 + s.span
            # Right-open except on the final span, so the shared interior
            # support x is evaluated exactly once.
            sel = (x >= lo) & (x <= hi) if i == len(self.spans) - 1 else (x >= lo) & (x < hi)
            out[sel] = s.z(x[sel])
        # Anything outside [0, total_span] clamps to the nearest span's value.
        out[x < self.spans[0].x0] = self.height
        out[x > self.spans[-1].x0 + self.spans[-1].span] = self.height
        return out

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """`n` points across the whole scenario, in x order."""
        x = np.linspace(0.0, self.total_span, n)
        return x, self.z(x)

    def arc_length(self, n: int = 20001) -> float:
        """Numerically integrated arc length -- a self-check that should
        reproduce ``total_length`` to ~1e-6 m."""
        x, z = self.sample(n)
        return float(np.sum(np.hypot(np.diff(x), np.diff(z))))

    def summary(self, weight_per_length: float | None = None) -> str:
        lines = [
            f"analytic catenary: {self.num_supports} supports at z={self.height:.3f} m, "
            f"span S={self.total_span:.3f} m, length L={self.total_length:.3f} m",
            f"  L/S = {self.total_length / self.total_span:.4f}   "
            f"a = {self.a:.4f} m   sag = {self.sag * 1e3:.1f} mm   "
            f"lowest z = {self.height - self.sag:.4f} m",
            f"  arc-length self-check: {self.arc_length():.6f} m "
            f"(target {self.total_length:.6f} m)",
        ]
        if weight_per_length is not None:
            s = self.spans[0]
            lines.append(
                f"  horizontal tension T0 = {s.horizontal_tension(weight_per_length):.4f} N, "
                f"max tension at supports = {s.max_tension(weight_per_length):.4f} N")
        return "\n".join(lines)


def solve(span: float, length: float, height: float, num_supports: int = 2) -> CatenarySolution:
    """Analytic shape for a cable on equal-height supports.

    Args:
        span: total horizontal distance between the two OUTER supports [m].
        length: total cable length [m].
        height: common height of every support [m].
        num_supports: 2 (ends only) or 3 (ends plus an evenly-spaced middle).

    Returns:
        A :class:`CatenarySolution`. For ``num_supports == 3`` this is two
        identical sub-catenaries of span S/2 and length L/2 -- exact, because
        equal support heights make the halves symmetric and independent.
    """
    if num_supports not in (2, 3):
        raise ValueError(f"num_supports must be 2 or 3, got {num_supports}")

    n_spans = num_supports - 1
    sub_span = span / n_spans
    sub_length = length / n_spans
    a = solve_shape_parameter(sub_span, sub_length)

    spans = [
        CatenarySpan(x0=i * sub_span, span=sub_span, height=height, length=sub_length, a=a)
        for i in range(n_spans)
    ]
    return CatenarySolution(spans=spans, total_span=span, total_length=length, height=height)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Analytic catenary reference.")
    p.add_argument("--length", type=float, default=1.0, help="cable length [m]")
    p.add_argument("--span", type=float, default=0.8, help="outer support separation [m]")
    p.add_argument("--height", type=float, default=1.0, help="support height [m]")
    p.add_argument("--points", type=int, default=2, choices=[2, 3], help="number of supports")
    a = p.parse_args()

    from cable_config import CABLE

    sol = solve(a.span, a.length, a.height, a.points)
    print(sol.summary(weight_per_length=CABLE.weight_per_length))
