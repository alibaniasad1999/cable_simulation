"""
Physical parameters of the cable under study -- ONE source of truth for every
method in this benchmark.

The reference cable is the one that gets photographed for the real-vs-sim
comparison: a 1 m Apple woven USB-C charge cable. It is a composite (copper
core + shielding + insulation + woven polyester jacket), so there is no
published Young's modulus; the values below are the calibrated starting point
documented in ``image_utils/README.md``.

Every method re-discretises and re-scales these numbers in its own way -- the
FEM script fattens the rod and rescales E/density to keep the element aspect
ratio sane, the Warp/Newton rods use a Cosserat stiffness derived from beam
theory, the capsule chain derives D6 joint stiffness from EI -- but they must
all START from the same physical target, otherwise the comparison is
meaningless. That single starting point lives HERE.

Do not hardcode these values in the individual method scripts. Import them:

    from cable_config import CABLE, bend_stiffness_EI

Every value is overridable per-run through the same ``CABLE_*`` environment
variables the older scripts used, so existing run commands keep working.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw)


@dataclass(frozen=True)
class CableProperties:
    """The physical cable, independent of any solver."""

    length: float          # [m]     total cable length
    radius: float          # [m]     cross-section radius
    youngs_modulus: float  # [Pa]    effective E of the whole cable
    poisson_ratio: float   # [-]
    density: float         # [kg/m3] effective bulk density
    mass: float            # [kg]    total mass

    # -- derived ---------------------------------------------------------
    @property
    def area(self) -> float:
        """Cross-section area A = pi r^2 [m^2]."""
        return math.pi * self.radius ** 2

    @property
    def second_moment(self) -> float:
        """Area moment of inertia I = pi r^4 / 4 [m^4] (solid circle)."""
        return 0.25 * math.pi * self.radius ** 4

    @property
    def bending_stiffness(self) -> float:
        """EI [N m^2] -- what actually sets how much a cable resists bending."""
        return self.youngs_modulus * self.second_moment

    @property
    def axial_stiffness(self) -> float:
        """EA [N] -- resistance to stretching."""
        return self.youngs_modulus * self.area

    @property
    def shear_modulus(self) -> float:
        """G = E / (2 (1 + nu)) [Pa]."""
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    @property
    def torsional_stiffness(self) -> float:
        """GJ [N m^2], J = 2 I for a solid circular section."""
        return self.shear_modulus * 2.0 * self.second_moment

    @property
    def mass_per_length(self) -> float:
        """mu [kg/m] -- the only material quantity a pure catenary cares about."""
        return self.mass / self.length

    @property
    def weight_per_length(self) -> float:
        """w = mu g [N/m]."""
        return self.mass_per_length * GRAVITY

    def gravito_bending_length(self) -> float:
        """Elasto-gravitational length l = (EI / w)^(1/3) [m].

        The scale over which bending stiffness competes with gravity. When
        l << L the cable behaves as an ideal (bending-free) catenary, which is
        exactly the regime that makes the analytic catenary a fair reference.
        """
        return (self.bending_stiffness / self.weight_per_length) ** (1.0 / 3.0)

    def summary(self) -> str:
        l = self.gravito_bending_length()
        return (
            f"cable: L={self.length:.3f} m  r={self.radius * 1e3:.2f} mm  "
            f"m={self.mass * 1e3:.1f} g  mu={self.mass_per_length * 1e3:.1f} g/m\n"
            f"       E={self.youngs_modulus / 1e6:.1f} MPa  nu={self.poisson_ratio:.2f}  "
            f"EI={self.bending_stiffness:.3e} N m^2  EA={self.axial_stiffness:.3e} N\n"
            f"       elasto-gravitational length l=(EI/w)^(1/3)={l * 1e3:.1f} mm "
            f"({l / self.length:.4f} L) -> "
            + ("catenary-dominated regime" if l < 0.1 * self.length else "bending matters")
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            area=self.area,
            second_moment=self.second_moment,
            bending_stiffness=self.bending_stiffness,
            axial_stiffness=self.axial_stiffness,
            torsional_stiffness=self.torsional_stiffness,
            mass_per_length=self.mass_per_length,
            gravito_bending_length=self.gravito_bending_length(),
        )
        return d


GRAVITY = _env_float("CABLE_GRAVITY", 9.81)  # [m/s^2]

_LENGTH = _env_float("CABLE_LENGTH", 1.0)
_RADIUS = _env_float("CABLE_RADIUS", 2.0e-3)
_DENSITY = _env_float("CABLE_DENSITY", 2390.0)

CABLE = CableProperties(
    length=_LENGTH,
    radius=_RADIUS,
    youngs_modulus=_env_float("CABLE_E", 80.0e6),
    poisson_ratio=_env_float("CABLE_NU", 0.45),
    density=_DENSITY,
    # rho * pi r^2 L ~= 30 g for the reference cable; override with CABLE_MASS.
    mass=_env_float("CABLE_MASS", _DENSITY * math.pi * _RADIUS ** 2 * _LENGTH),
)


def with_length(length: float) -> CableProperties:
    """Same physical cable, different total length.

    Mass scales with length (same cross-section and material), so mass per
    length -- and therefore the catenary shape for a given span -- is preserved.
    """
    from dataclasses import replace

    return replace(CABLE, length=length, mass=CABLE.mass_per_length * length)


def rod_stiffness(cable: CableProperties, segment_length: float) -> tuple[float, float]:
    """Per-joint (stretch, bend) stiffness for Newton's ``add_rod``.

    Mirrors ``newton._src.utils.cable.create_cable_stiffness_from_elastic_moduli``:
        stretch = E A / L_seg  [N/m]
        bend    = E I / L_seg  [N m]
    Kept here so the Warp and capsule methods can use the identical numbers
    without importing Newton.
    """
    if segment_length <= 0.0:
        raise ValueError("segment_length must be > 0")
    return (
        cable.axial_stiffness / segment_length,
        cable.bending_stiffness / segment_length,
    )


if __name__ == "__main__":
    print(CABLE.summary())
