#!/usr/bin/env python3
"""
Experiment definition: one YAML file describing a cable, a rig, and a photo.

WHY A CONFIG FILE
-----------------
Everything this project measures depends on numbers that belong to a PHYSICAL
SETUP, not to the code: how long the cable is, where it is clamped, what it is
made of, which photograph shows it. Hard-coding those means a new cable is a
code change, and a code change is something to review, break, and forget to
undo. Describing them in a file means a new cable is a new file, and the two
experiments can be re-run side by side afterwards because both descriptions
still exist.

    python3 run_experiment.py --config experiments/apple_cable_3pt.yaml

MEASURED VERSUS SPECIFIED
-------------------------
Two fields -- ``cable.length_m`` and ``supports.mid_fraction`` -- may be left
null, in which case they are measured from the photograph. That is the DEFAULT
and the recommended setting:

  * Cable length is the single most influential input, and the free length
    between two clamps is not the length printed on the packaging. Measuring it
    from the same photograph the simulation is compared against removes a whole
    class of silent mismatch.
  * The arc-length split at a middle clamp is a property of where the clamp
    grips, which no tape measure gives you. Measured here it is 45.7/54.3 where
    the chord-proportional convention assumes 42.9/57.1 -- so assuming it
    quietly simulates a different experiment.

Set them explicitly to override, e.g. when the photograph is poor or when
running a hypothetical. Whichever route is taken, the resolved value and its
PROVENANCE ("measured" or "specified") are recorded in the run's metadata, so a
report can state which numbers were observed and which were assumed.

DEFAULTS
--------
Solver defaults are the ones this repository's own sweep found, not upstream
Newton's: 4 substeps x 200 iterations with ``stretch_damping = 0``. That
configuration was strictly better than the previous default on every axis
measured -- accuracy, arc-length drift, settling, and wall-clock -- see
``run_sweep.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any

# Solver settings established by run_sweep.py; see the module docstring.
DEFAULT_SUBSTEPS = 4
DEFAULT_ITERATIONS = 200
DEFAULT_STRETCH_DAMPING = 0.0
DEFAULT_BEND_DAMPING = 1.0e-4


@dataclass
class CableSpec:
    """The physical cable. Defaults describe the TPU cable in cable_config.py."""
    radius_m: float = 2.0e-3
    youngs_modulus_pa: float = 80.0e6
    poisson_ratio: float = 0.45
    density_kg_m3: float = 2390.0
    length_m: float | None = None  # None -> measure from the photograph


@dataclass
class SupportSpec:
    """Where the cable is clamped. x measured from the leftmost clamp."""
    x_m: list[float] = field(default_factory=lambda: [0.0, 0.35])
    height_m: float = 1.0
    mid_fraction: float | None = None  # None -> measure (3 supports only)


@dataclass
class ImageSpec:
    """The photograph, and the knobs the extractor needs for it."""
    path: str | None = None
    sat_max: int = 70          # HSV: cable is nearly grey, door is saturated
    val_min: int = 150         # HSV: cable is bright
    step_px: float = 4.0       # tracer advance
    halfwidth_px: float = 40.0 # tracer probe half-length


@dataclass
class SimulationSpec:
    methods: list[str] = field(default_factory=lambda: ["newton_cable"])
    segments: int = 60
    max_time: float = 8.0
    settle_vel: float = 2.0e-3
    substeps: int = DEFAULT_SUBSTEPS
    iterations: int = DEFAULT_ITERATIONS
    stretch_damping: float = DEFAULT_STRETCH_DAMPING
    bend_damping: float = DEFAULT_BEND_DAMPING
    mid_support: str = "clamp"   # matches a taped clamp; "pin" frees the angle
    # Half-width [m] of the flat plateau used by the 'newton_cable_tape' method,
    # which models the middle support as a strip of tape rather than a point pin.
    tape_halfwidth: float = 0.012
    # Bending length l=(EI/w)^(1/3) [m] for the Warp rod. null -> the material EI,
    # which for the stand-in moduli is ~0.15 m, where a slack rod buckles. The
    # real cable's ~2 cm rounding scale implies ~0.02 m, where it drapes.
    warp_bend_length_m: float | None = None


@dataclass
class Experiment:
    name: str = "experiment"
    description: str = ""
    cable: CableSpec = field(default_factory=CableSpec)
    supports: SupportSpec = field(default_factory=SupportSpec)
    image: ImageSpec = field(default_factory=ImageSpec)
    simulation: SimulationSpec = field(default_factory=SimulationSpec)

    # -- derived -----------------------------------------------------------
    @property
    def num_points(self) -> int:
        return len(self.supports.x_m)

    @property
    def span_m(self) -> float:
        return self.supports.x_m[-1] - self.supports.x_m[0]

    @property
    def mid_x(self) -> float | None:
        if self.num_points != 3:
            return None
        return self.supports.x_m[1] - self.supports.x_m[0]

    def validate(self) -> None:
        """Reject impossible setups here, with an explanation.

        A config error caught now costs a second; the same error caught after a
        forty-minute benchmark costs forty minutes and looks like a physics
        result.
        """
        xs = self.supports.x_m
        if len(xs) not in (2, 3):
            raise SystemExit(
                f"[{self.name}] supports.x_m needs 2 or 3 positions, got {len(xs)}")
        if list(xs) != sorted(xs):
            raise SystemExit(
                f"[{self.name}] supports.x_m must increase left to right, got {xs}")
        if len(set(xs)) != len(xs):
            raise SystemExit(f"[{self.name}] supports.x_m has duplicate positions: {xs}")
        if self.span_m <= 0:
            raise SystemExit(f"[{self.name}] supports must span a positive distance")

        if self.cable.radius_m <= 0:
            raise SystemExit(f"[{self.name}] cable.radius_m must be positive")
        if self.cable.density_kg_m3 <= 0:
            raise SystemExit(f"[{self.name}] cable.density_kg_m3 must be positive")

        L = self.cable.length_m
        if L is not None and L <= self.span_m:
            raise SystemExit(
                f"[{self.name}] cable.length_m ({L}) must exceed the support span "
                f"({self.span_m:.3f} m), or the cable is taut and has no hanging shape")

        mf = self.supports.mid_fraction
        if mf is not None:
            if self.num_points != 3:
                raise SystemExit(
                    f"[{self.name}] supports.mid_fraction only applies with 3 supports")
            if not 0.0 < mf < 1.0:
                raise SystemExit(
                    f"[{self.name}] supports.mid_fraction must be in (0, 1), got {mf}")

        if (L is None or (self.num_points == 3 and mf is None)) and not self.image.path:
            raise SystemExit(
                f"[{self.name}] cable.length_m or supports.mid_fraction is null "
                "(meaning 'measure it from the photograph'), but image.path is not "
                "set. Either give the photograph, or state the values explicitly.")

        if self.image.path and not os.path.isfile(self.image.path):
            raise SystemExit(f"[{self.name}] image.path not found: {self.image.path}")

        if self.simulation.mid_support not in ("clamp", "pin"):
            raise SystemExit(
                f"[{self.name}] simulation.mid_support must be 'clamp' or 'pin', "
                f"got {self.simulation.mid_support!r}")

    # -- IO ----------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "Experiment":
        """Read a YAML (or JSON) experiment file, rejecting unknown keys.

        Unknown keys are an error, not a warning: a typo like ``lenght_m``
        would otherwise be silently ignored and the run would quietly measure
        the length from the photo instead of using the value the author typed.
        """
        with open(path) as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            import yaml
            raw = yaml.safe_load(text) or {}
        else:
            import json
            raw = json.loads(text)

        def coerce(value, annotation, where, key):
            """Coerce a YAML scalar to the field's declared type.

            This exists because of a genuine YAML 1.1 trap: PyYAML only reads
            scientific notation as a float when the exponent carries a SIGN, so
            ``80.0e6`` silently becomes the string "80.0e6" while ``80.0e+6``
            becomes a float. Left uncoerced that surfaces much later as
            "unsupported operand type(s) for /: 'str' and 'float'", pointing at
            arithmetic rather than at the config line that caused it.
            """
            if value is None:
                return None
            ann = str(annotation)
            try:
                if "list[float]" in ann:
                    return [float(v) for v in value]
                if "int" in ann and "float" not in ann:
                    return int(value)
                if "float" in ann:
                    return float(value)
                if "list[str]" in ann:
                    return [str(v) for v in value]
                if "str" in ann:
                    return str(value)
            except (TypeError, ValueError):
                raise SystemExit(
                    f"{path}: '{where}.{key}' = {value!r} cannot be read as "
                    f"{ann.replace('typing.', '')}")
            return value

        def build(kind, data, where):
            if data is None:
                return kind()
            if not isinstance(data, dict):
                raise SystemExit(f"{path}: '{where}' must be a mapping")
            fields = kind.__dataclass_fields__
            unknown = set(data) - set(fields)
            if unknown:
                raise SystemExit(
                    f"{path}: unknown key(s) in '{where}': {', '.join(sorted(unknown))}\n"
                    f"  known keys: {', '.join(sorted(fields))}")
            return kind(**{k: coerce(v, fields[k].type, where, k) for k, v in data.items()})

        top_known = {"name", "description", "cable", "supports", "image", "simulation"}
        unknown = set(raw) - top_known
        if unknown:
            raise SystemExit(
                f"{path}: unknown top-level key(s): {', '.join(sorted(unknown))}\n"
                f"  known keys: {', '.join(sorted(top_known))}")

        exp = cls(
            name=raw.get("name", os.path.splitext(os.path.basename(path))[0]),
            description=raw.get("description", ""),
            cable=build(CableSpec, raw.get("cable"), "cable"),
            supports=build(SupportSpec, raw.get("supports"), "supports"),
            image=build(ImageSpec, raw.get("image"), "image"),
            simulation=build(SimulationSpec, raw.get("simulation"), "simulation"),
        )
        exp.validate()
        return exp

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def describe(self) -> str:
        xs = ", ".join(f"{x:.3f}" for x in self.supports.x_m)
        L = "measure from photo" if self.cable.length_m is None else f"{self.cable.length_m:.4f} m"
        mf = self.supports.mid_fraction
        mf_s = ("measure from photo" if mf is None else f"{mf:.4f}") if self.num_points == 3 else "n/a"
        return (
            f"experiment  : {self.name}\n"
            f"  supports  : x = [{xs}] m, all at z = {self.supports.height_m:.3f} m "
            f"({self.num_points} points, span {self.span_m:.3f} m)\n"
            f"  length    : {L}\n"
            f"  mid split : {mf_s}\n"
            f"  cable     : r = {self.cable.radius_m * 1e3:.2f} mm, "
            f"E = {self.cable.youngs_modulus_pa / 1e6:.1f} MPa, "
            f"rho = {self.cable.density_kg_m3:.0f} kg/m^3\n"
            f"  photo     : {self.image.path or '(none)'}\n"
            f"  methods   : {', '.join(self.simulation.methods)}"
        )


def resolve_from_image(exp: Experiment) -> tuple[Experiment, dict[str, Any]]:
    """Fill any null measured field from the photograph.

    Returns (experiment, provenance) where provenance records, per field,
    whether the value was 'specified' or 'measured' and -- when measured -- the
    supporting numbers, so a report can show its working rather than presenting
    a bare number.
    """
    import copy
    import sys

    prov: dict[str, Any] = {}
    out = copy.deepcopy(exp)

    needs_length = out.cable.length_m is None
    needs_split = out.num_points == 3 and out.supports.mid_fraction is None
    if not (needs_length or needs_split):
        prov["cable.length_m"] = {"source": "specified", "value": out.cable.length_m}
        if out.num_points == 3:
            prov["supports.mid_fraction"] = {"source": "specified",
                                             "value": out.supports.mid_fraction}
        return out, prov

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_utils"))
    from measure_length import extract_centreline, trace_centreline, length_from_sag
    import numpy as np

    mask = extract_centreline(out.image.path, out.image.sat_max, out.image.val_min)
    x, y = trace_centreline(mask, step=out.image.step_px, halfwidth=out.image.halfwidth_px)

    x_lo, x_hi = float(x[0]), float(x[-1])
    if x_hi < x_lo:
        x_lo, x_hi = x_hi, x_lo
    scale = out.span_m / (x_hi - x_lo)
    xm = (x - x_lo) * scale
    ym = -(y - y.min()) * scale

    xs = [v - out.supports.x_m[0] for v in out.supports.x_m]
    per_span, sags = [], []
    for i in range(len(xs) - 1):
        a, b = xs[i], xs[i + 1]
        sel = (xm >= a - 1e-9) & (xm <= b + 1e-9)
        if sel.sum() < 10:
            raise SystemExit(
                f"[{out.name}] too few traced points between x={a:.3f} and {b:.3f} m "
                "to measure this span -- check the detection figure")
        sy = ym[sel]
        sag = max(sy[0], sy[-1]) - sy.min()
        per_span.append(length_from_sag(b - a, sag))
        sags.append(sag)

    total = float(sum(per_span))
    if needs_length:
        out.cable.length_m = total
        prov["cable.length_m"] = {
            "source": "measured",
            "value": total,
            "image": out.image.path,
            "per_span_m": per_span,
            "sag_m": sags,
            "scale_mm_per_px": scale * 1e3,
        }
    else:
        prov["cable.length_m"] = {"source": "specified", "value": out.cable.length_m}

    if out.num_points == 3:
        if needs_split:
            frac = per_span[0] / total
            out.supports.mid_fraction = frac
            chord = (xs[1] - xs[0]) / out.span_m
            prov["supports.mid_fraction"] = {
                "source": "measured",
                "value": frac,
                "chord_proportional": chord,
                "difference_pp": (frac - chord) * 100.0,
            }
        else:
            prov["supports.mid_fraction"] = {"source": "specified",
                                             "value": out.supports.mid_fraction}

    out.validate()  # measured values must still make physical sense
    return out, prov


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Load and validate an experiment file.")
    p.add_argument("config")
    p.add_argument("--resolve", action="store_true",
                   help="also measure any null fields from the photograph")
    a = p.parse_args()

    exp = Experiment.load(a.config)
    print(exp.describe())
    if a.resolve:
        exp, prov = resolve_from_image(exp)
        print()
        print("resolved:")
        print(exp.describe())
        print()
        print("provenance:")
        for k, v in prov.items():
            print(f"  {k}: {v['source']} = {v['value']}")
