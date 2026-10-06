#!/usr/bin/env python3
"""
The mathematical model: the governing ODE for a hanging cable, solved
numerically without ever assuming its closed-form answer.

WHY THIS EXISTS SEPARATELY FROM catenary.py
-------------------------------------------
``catenary.py`` evaluates the closed-form solution, ``z = a cosh((x-x0)/a) + c``.
That is the right tool for scoring simulations, because it is exact and free.
But it presupposes the answer: if the derivation were wrong, or if the boundary
conditions in this benchmark did not actually match the classical catenary
problem, every "error" reported against it would be measured from the wrong
curve, consistently and invisibly.

This module closes that gap. It integrates the governing differential equation
directly and shoots on its free parameter to satisfy the boundary conditions,
using the cosh solution nowhere. Agreement between the two is then a genuine
check on the mathematics rather than a restatement of it.

That gives the study three independent legs -- MATH (here), REAL (photograph),
and SIMULATION (the five solvers) -- which is the comparison this project is
for.

================================================================================
DERIVATION -- the static equation
================================================================================
Take a perfectly flexible, inextensible cable of uniform weight per unit length
w = rho * A * g, hanging in a vertical plane. "Perfectly flexible" means it
carries no bending moment, so the internal force at any cross-section is a pure
tension T directed along the tangent.

Consider the equilibrium of an element of arc length ds, at angle theta to the
horizontal. Two forces act: the tension at each end, and its own weight w ds.

  Horizontal:   d(T cos(theta)) = 0        =>  T cos(theta) = H, a constant.
  Vertical:     d(T sin(theta)) = w ds

The horizontal equation is the key structural fact: the horizontal component of
tension is the SAME everywhere in the cable. Writing tan(theta) = dz/dx = z'
and substituting T = H / cos(theta) into the vertical equation:

      d(H z') = w ds

and since ds = sqrt(1 + z'^2) dx,

      ---------------------------------------------
      z'' = (w / H) * sqrt(1 + z'^2)        ... (1)
      ---------------------------------------------

which is the governing ODE. It is second-order, nonlinear, and autonomous in x.
Defining the CATENARY PARAMETER a = H / w (units of length) puts it in the form
used throughout this project:

      z'' = sqrt(1 + z'^2) / a              ... (1')

BOUNDARY CONDITIONS AND THE ARC-LENGTH CONSTRAINT
-------------------------------------------------
Equation (1') is two-point: z(0) = z(S) = height for level supports. But a is
not known in advance -- it is fixed by the requirement that the cable has the
length it actually has:

      integral_0^S sqrt(1 + z'^2) dx = L    ... (2)

So this is a boundary-value problem with an unknown parameter, and (2) is what
determines it. Physically: a is set by how much cable you hung, and H = w a is
then the horizontal tension the supports must supply. A slacker cable (larger
L/S) has smaller a and therefore carries LESS tension -- which is why a taut
cable is the one that snaps.

WHAT THE CLOSED FORM IS
-----------------------
Integrating (1') gives z' = sinh((x - x0)/a) and hence z = a cosh((x - x0)/a) + c.
This module does NOT use that. It integrates (1') numerically and solves (2) by
bisection on a.

================================================================================
THE DYNAMIC EQUATION (what "shape over time" obeys)
================================================================================
The simulations do not solve (1') -- they integrate the cable's dynamics until
it stops moving. The corresponding equation of motion, for a position field
r(s, t) parametrised by arc length s, is

      rho A d^2r/dt^2 = d/ds ( T(s,t) dr/ds ) + rho A g      ... (3)

subject to |dr/ds| = 1 (inextensibility), which is what T enforces -- T is the
Lagrange multiplier of that constraint, not a constitutive property. Setting
d^2r/dt^2 = 0 in (3) recovers (1'), so the static catenary is the equilibrium
of the dynamic problem, which is exactly the claim the benchmark relies on when
it scores a settled simulation against the analytic curve.

Equation (3) is a wave equation with a spatially varying wave speed
c(s) = sqrt(T(s)/(rho A)). Transverse waves travel faster where the cable is
tauter -- near the supports -- and slowest at the lowest point, where T = H.
This is why a hanging cable rings at a set of discrete frequencies, and why the
solvers' residual swing decays slowly: the lowest mode is weakly damped and its
period is set by the cable's own geometry, not by the timestep.

A REAL cable adds a bending term, EI d^4r/ds^4, dropped in (1) by the perfectly
flexible assumption. Its importance is set by the elasto-gravitational length
l = (EI/w)^(1/3) against the span; ``analysis/compare.py`` reports that ratio.
For the cable measured here it is millimetric against a 350 mm span, so (1') is
an excellent model -- which the numbers bear out.

    python3 methods/cable_ode.py --span 0.35 --length 0.83
    python3 methods/cable_ode.py --span 0.35 --length 0.83 --compare
"""

from __future__ import annotations

import argparse
import math

import numpy as np


def _rhs(state: np.ndarray, a: float) -> np.ndarray:
    """Right-hand side of (1') written as a first-order system.

    state = [z, z'];  returns [z', z''] with z'' = sqrt(1 + z'^2)/a.
    """
    z_prime = state[1]
    return np.array([z_prime, math.sqrt(1.0 + z_prime * z_prime) / a])


def integrate_half(a: float, half_span: float, n: int = 20001) -> tuple[np.ndarray, np.ndarray]:
    """Integrate (1') from the cable's lowest point outward, using RK4.

    Starting at the vertex exploits the only thing symmetry gives us for free:
    for level supports the lowest point is at mid-span and the slope there is
    zero. That turns a two-point BVP into an initial-value problem in the single
    unknown ``a``, with no appeal to the closed-form solution.

    Returns (x, z) on [0, half_span] with z(0) = 0 at the vertex.
    """
    x = np.linspace(0.0, half_span, n)
    h = x[1] - x[0]
    out = np.empty((n, 2))
    state = np.array([0.0, 0.0])  # z = 0, z' = 0 at the vertex
    out[0] = state
    for i in range(1, n):
        k1 = _rhs(state, a)
        k2 = _rhs(state + 0.5 * h * k1, a)
        k3 = _rhs(state + 0.5 * h * k2, a)
        k4 = _rhs(state + h * k3, a)
        state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[i] = state
    return x, out[:, 0]


def arc_length_half(a: float, half_span: float, n: int = 20001) -> float:
    """Arc length of the integrated half-cable -- the left side of (2)."""
    x, z = integrate_half(a, half_span, n)
    return float(np.sum(np.hypot(np.diff(x), np.diff(z))))


def solve_a(span: float, length: float, n: int = 20001) -> float:
    """Find the catenary parameter a satisfying the length constraint (2).

    Bisection, because arc length decreases monotonically in a (a larger a is a
    tauter, shallower cable) and monotonicity is all bisection needs -- no
    derivative of a numerically integrated quantity is required.
    """
    if length <= span:
        raise ValueError(
            f"length {length:.4f} m must exceed span {span:.4f} m: a shorter cable "
            "cannot bridge the supports without stretching, which (1) forbids")

    half_span, half_len = 0.5 * span, 0.5 * length
    lo, hi = 1.0e-6, max(span, length)
    while arc_length_half(hi, half_span, 2001) > half_len:
        hi *= 2.0
        if hi > 1.0e9:
            raise RuntimeError("failed to bracket a")

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if arc_length_half(mid, half_span, 2001) > half_len:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_profile(span: float, length: float, height: float = 0.0,
                  n: int = 20001) -> tuple[np.ndarray, np.ndarray, float]:
    """Full cable profile from the ODE alone.

    Returns (x, z, a) with x on [0, span] and z(0) = z(span) = height.
    """
    a = solve_a(span, length, n)
    xr, zr = integrate_half(a, 0.5 * span, n // 2 + 1)
    # Mirror the integrated half about mid-span, then lift so the ends sit at
    # `height` -- the ODE fixes the shape, the boundary condition fixes the offset.
    x = np.concatenate([0.5 * span - xr[::-1], 0.5 * span + xr[1:]])
    z = np.concatenate([zr[::-1], zr[1:]])
    z = z - z[0] + height
    return x, z, a


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--span", type=float, default=0.35, help="support separation S [m]")
    p.add_argument("--length", type=float, default=0.83, help="cable length L [m]")
    p.add_argument("--height", type=float, default=1.0, help="support height [m]")
    p.add_argument("--compare", action="store_true",
                   help="check the numerical ODE solution against the closed form")
    p.add_argument("--samples", type=int, default=20001, help="integration points")
    args = p.parse_args()

    x, z, a = solve_profile(args.span, args.length, args.height, args.samples)
    arc = float(np.sum(np.hypot(np.diff(x), np.diff(z))))
    sag = float(z.max() - z.min())

    print("ODE model (numerically integrated, closed form not used)")
    print(f"  governing equation : z'' = sqrt(1 + z'^2) / a")
    print(f"  span S             : {args.span:.4f} m")
    print(f"  length L           : {args.length:.4f} m   (L/S = {args.length / args.span:.4f})")
    print(f"  solved a = H/w     : {a:.6f} m")
    print(f"  sag                : {sag * 1e3:.3f} mm")
    print(f"  lowest point       : {z.min():.6f} m")
    print(f"  arc-length check   : {arc:.6f} m vs target {args.length:.6f} m "
          f"({(arc - args.length) * 1e3:+.4f} mm)")

    if args.compare:
        import catenary

        sol = catenary.solve(args.span, args.length, args.height, 2)
        z_closed = sol.z(x)
        diff = np.abs(z - z_closed)
        print()
        print("closed form  z = a cosh((x - S/2)/a) + c")
        print(f"  a (closed)         : {sol.a:.6f} m   "
              f"(ODE: {a:.6f} m, differs by {abs(sol.a - a) * 1e6:.3f} um)")
        print(f"  sag (closed)       : {sol.sag * 1e3:.3f} mm "
              f"(ODE: {sag * 1e3:.3f} mm, differs by {abs(sol.sag - sag) * 1e6:.3f} um)")
        print(f"  max |z_ODE - z_closed| : {diff.max() * 1e6:.3f} um")
        print()
        if diff.max() < 1.0e-5:
            print("  AGREE. The numerical solution of the ODE and the closed form describe")
            print("  the same curve, so the analytic reference the benchmark scores against")
            print("  is verified, not merely asserted.")
        else:
            print("  DISAGREE -- investigate before trusting any error measured against the")
            print("  closed form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
