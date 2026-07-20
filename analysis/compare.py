#!/usr/bin/env python3
"""
Compare every method in a benchmark run and emit the deliverables.

Reads a run directory produced by ``run_benchmark.py``:

    results/<run>/
        run.json                  the shared scenario
        <method>/profile.csv      settled shape, x_m,z_m
        <method>/meta.json        timings, convergence, solver settings

and writes, into the same directory:

    metrics.csv     one row per method -- THE comparison table
    overlay.png     every method plus the analytic catenary on one axis
    errors.png      height error vs x, per method
    report.tex      a standalone LaTeX report of the whole comparison

Optionally also compares against a real cable profile extracted from a photo by
``image_utils/extract_cable_profile.py`` (same ``x_m,z_m`` format):

    python3 analysis/compare.py --run results/<run> --real path/to/profile.csv

METRICS, AND WHAT THEY ARE WORTH.

  rmse_mm        RMS height difference from the reference, sampled on a common
                 x grid. The headline number, but see the caveat below.
  max_err_mm     Worst single-point height error.
  sag_err_mm     Error in the single most quotable scalar, the lowest point.
  arc_drift_pct  How much the cable's own length changed during the run. This
                 one is reference-FREE: a cable that grows 2% has violated
                 inextensibility no matter what shape it settled into, so it
                 catches solver error that shape metrics can hide.
  wall_s         Wall-clock time to the settled state.
  settled        Whether the motion actually decayed, or the run just hit its
                 time cap. An unsettled run's other numbers are provisional.

CAVEAT ON THE CATENARY REFERENCE.  The analytic catenary assumes zero bending
stiffness. A real cable's bending matters over the elasto-gravitational length
l = (EI/w)^(1/3); the reference is trustworthy only where l is small compared
with the span it must curve over. The report prints l/S for the scenario and
warns when the catenary is being used outside that range -- which it is for the
3-support case with a stiff cable, where the middle support demands a slope
kink that no rod with real EI can reproduce.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "methods"))

import catenary  # noqa: E402
from cable_config import with_length  # noqa: E402
from hang_common import Scenario  # noqa: E402

ORDER = ["newton_cable", "newton_engine", "physx_capsule", "physx_fem", "warp_rod"]

FAMILY = {
    "newton_cable": "Newton",
    "newton_engine": "Newton",
    "physx_capsule": "PhysX (pure Isaac Sim)",
    "physx_fem": "PhysX (pure Isaac Sim)",
    "warp_rod": "Reference",
}

COLORS = {
    "newton_cable": "#0072B2",
    "newton_engine": "#56B4E9",
    "physx_capsule": "#D55E00",
    "physx_fem": "#E69F00",
    "warp_rod": "#009E73",
    "real": "#CC79A7",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_profile(path: str) -> np.ndarray:
    """Read an ``x_m,z_m`` profile CSV into an (N, 2) array."""
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        cols = {c.lower().strip(): c for c in (r.fieldnames or [])}
        if "x_m" not in cols or "z_m" not in cols:
            raise ValueError(f"{path}: expected columns x_m,z_m, got {r.fieldnames}")
        return np.array([[float(row[cols["x_m"]]), float(row[cols["z_m"]])] for row in r])


def load_run(run_dir: str):
    """Load the scenario and every method result present in a run directory."""
    scenario = Scenario.load(os.path.join(run_dir, "run.json"))
    results = {}
    for method in ORDER:
        mdir = os.path.join(run_dir, method)
        prof, meta = os.path.join(mdir, "profile.csv"), os.path.join(mdir, "meta.json")
        if not (os.path.isfile(prof) and os.path.isfile(meta)):
            continue
        with open(meta) as fh:
            results[method] = {"profile": read_profile(prof), "meta": json.load(fh)}
    return scenario, results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def resample(profile: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate z onto `grid`, sorting by x first."""
    o = np.argsort(profile[:, 0])
    return np.interp(grid, profile[o, 0], profile[o, 1])


def compute_metrics(scenario: Scenario, results: dict, real: np.ndarray | None,
                    n_grid: int = 400) -> list[dict]:
    sol = scenario.catenary()
    grid = np.linspace(0.0, scenario.span, n_grid)
    ref_z = np.asarray(sol.z(grid))
    real_z = resample(real, grid) if real is not None else None

    rows = []
    for method, data in results.items():
        prof, meta = data["profile"], data["meta"]
        z = resample(prof, grid)
        err = z - ref_z
        row = {
            "method": method,
            "family": FAMILY.get(method, ""),
            "label": meta.get("label", method),
            "settled": meta.get("settled", False),
            "sim_time_s": round(meta.get("sim_time_s", 0.0), 3),
            "wall_s": round(meta.get("wall_time_s", 0.0), 2),
            "num_nodes": meta.get("num_nodes", len(prof)),
            "rmse_mm": round(1e3 * float(np.sqrt(np.mean(err ** 2))), 3),
            "max_err_mm": round(1e3 * float(np.max(np.abs(err))), 3),
            "sag_mm": round(1e3 * meta.get("sag_m", float("nan")), 3),
            "sag_ref_mm": round(1e3 * sol.sag, 3),
            "sag_err_mm": round(1e3 * (meta.get("sag_m", float("nan")) - sol.sag), 3),
            "arc_m": round(meta.get("arc_length_m", float("nan")), 6),
            "arc_drift_pct": round(meta.get("arc_drift_pct", float("nan")), 4),
        }
        if real_z is not None:
            rerr = z - real_z
            row["rmse_vs_real_mm"] = round(1e3 * float(np.sqrt(np.mean(rerr ** 2))), 3)
            row["max_err_vs_real_mm"] = round(1e3 * float(np.max(np.abs(rerr))), 3)
        rows.append(row)

    rows.sort(key=lambda r: (ORDER.index(r["method"]) if r["method"] in ORDER else 99))
    return rows


def write_metrics_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for r in rows:  # keep a stable union of columns
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(run_dir, scenario, results, real, rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] no figures ({exc})")
        return []

    sol = scenario.catenary()
    cx, cz = sol.sample(500)
    grid = np.linspace(0.0, scenario.span, 400)
    ref_z = np.asarray(sol.z(grid))
    made = []

    # ---- overlay ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cx, cz, "--", color="0.3", lw=2.0, label="analytic catenary", zorder=3)
    for method, data in results.items():
        p = data["profile"]
        ax.plot(p[:, 0], p[:, 1], "-", lw=2.0, color=COLORS.get(method),
                label=data["meta"].get("label", method))
    if real is not None:
        ax.plot(real[:, 0], real[:, 1], ":", lw=2.5, color=COLORS["real"],
                label="real cable (photo)")
    sup = scenario.supports
    ax.plot(sup[:, 0], sup[:, 2], "o", color="crimson", ms=11, zorder=6, label="supports")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("z  [m]")
    ax.set_aspect("equal", "box")
    ax.grid(True, alpha=0.3)
    # The cable sags away from the top of the axes, so upper-centre is the one
    # region guaranteed to be clear of the curves in every scenario.
    ax.legend(loc="upper center", fontsize=9, ncol=2, framealpha=0.92)
    ax.set_title(f"Cable hang, {scenario.num_points} supports  "
                 f"(L={scenario.length:.2f} m, S={scenario.span:.2f} m)")
    fig.tight_layout()
    p = os.path.join(run_dir, "overlay.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    made.append(p)

    # ---- error curves ----
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(0.0, color="0.3", ls="--", lw=1.5)
    for method, data in results.items():
        z = resample(data["profile"], grid)
        ax.plot(grid, 1e3 * (z - ref_z), "-", lw=2.0, color=COLORS.get(method),
                label=data["meta"].get("label", method))
    for x in scenario.support_x:
        ax.axvline(x, color="crimson", alpha=0.35, lw=1.2)
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("z - z_catenary  [mm]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.set_title("Height error against the analytic catenary")
    fig.tight_layout()
    p = os.path.join(run_dir, "errors.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    made.append(p)
    return made


# ---------------------------------------------------------------------------
# LaTeX report
# ---------------------------------------------------------------------------
def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write_report(path, run_dir, scenario, results, rows, real_path, figures):
    sol = scenario.catenary()
    cable = with_length(scenario.length)
    l_eg = cable.gravito_bending_length()
    sub_span = scenario.span / (scenario.num_points - 1)
    ratio = l_eg / sub_span

    def fmt(v, nd=2):
        if isinstance(v, bool):
            return r"\checkmark" if v else "--"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return _tex_escape(str(v))

    body_rows = "\n".join(
        f"    {_tex_escape(r['label'])} & {fmt(r['rmse_mm'])} & {fmt(r['max_err_mm'])} & "
        f"{fmt(r['sag_mm'],1)} & {fmt(r['sag_err_mm'],1)} & {fmt(r['arc_drift_pct'],3)} & "
        f"{fmt(r['wall_s'],1)} & {fmt(r['settled'])} \\\\"
        for r in rows)

    real_section = ""
    if real_path and any("rmse_vs_real_mm" in r for r in rows):
        real_rows = "\n".join(
            f"    {_tex_escape(r['label'])} & {fmt(r.get('rmse_vs_real_mm', float('nan')))} & "
            f"{fmt(r.get('max_err_vs_real_mm', float('nan')))} \\\\" for r in rows)
        real_section = rf"""
\section{{Comparison with the real cable}}

Profile extracted from a photograph with \texttt{{image\_utils/extract\_cable\_profile.py}}
(SAM segmentation, metric scale from a measured in-plane reference), source
\texttt{{{_tex_escape(os.path.basename(real_path))}}}.

\begin{{tabular}}{{lrr}}
\toprule
Method & RMSE vs real [mm] & max error [mm] \\
\midrule
{real_rows}
\bottomrule
\end{{tabular}}
"""

    validity = (
        rf"Here $\ell/S_{{\mathrm{{span}}}} = {ratio:.3f}$, so bending is a "
        r"\emph{small correction} and the catenary is a sound reference."
        if ratio < 0.1 else
        rf"Here $\ell/S_{{\mathrm{{span}}}} = {ratio:.3f}$, which is \emph{{not}} small. "
        r"The catenary is being used outside its range of validity and the "
        r"residuals below should be read as the physical difference between an "
        r"elastic rod and an ideal chain, not as solver error.")

    figure_blocks = "\n".join(
        rf"""
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=\linewidth]{{{os.path.basename(f)}}}
  \caption{{{'All methods against the analytic catenary.' if 'overlay' in f else 'Height error against the analytic catenary. Vertical lines mark the supports.'}}}
\end{{figure}}""" for f in figures)

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.4cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{{hyperref}}

\title{{Cable hang: comparison of simulation methods\\
\large {scenario.num_points} supports, $L = {scenario.length:.3f}$\,m,
$S = {scenario.span:.3f}$\,m}}
\author{{Generated by \texttt{{analysis/compare.py}}}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{Setup}}

A cable of length $L = {scenario.length:.3f}$\,m hangs from
{scenario.num_points} supports, all at the same height
$z = {scenario.height:.3f}$\,m, evenly spaced across a horizontal span
$S = {scenario.span:.3f}$\,m. Every method solves this identical scenario,
discretised into {scenario.num_segments} segments, and each is initialised on
the analytic catenary so that what is measured is how far a solver
\emph{{drifts}} from equilibrium rather than how it recovers from an arbitrary
transient.

\paragraph{{Cable.}} $r = {cable.radius * 1e3:.2f}$\,mm,
$E = {cable.youngs_modulus / 1e6:.1f}$\,MPa, $\nu = {cable.poisson_ratio:.2f}$,
$\rho = {cable.density:.0f}$\,kg/m$^3$, mass ${cable.mass * 1e3:.1f}$\,g
($\mu = {cable.mass_per_length * 1e3:.1f}$\,g/m), so
$EI = {cable.bending_stiffness:.3e}$\,N\,m$^2$ and
$EA = {cable.axial_stiffness:.3e}$\,N.

\section{{Analytic reference}}

For a perfectly flexible, inextensible cable on equal-height supports,
\begin{{equation}}
  z(x) = H + a\left[\cosh\!\left(\frac{{x - S/2}}{{a}}\right)
                    - \cosh\!\left(\frac{{S}}{{2a}}\right)\right],
  \qquad L = 2a\sinh\!\left(\frac{{S}}{{2a}}\right),
\end{{equation}}
with $a = T_0/w$ solved from the length constraint. Because all supports share
one height, the {scenario.num_points}-support case separates exactly into
{scenario.num_points - 1} independent sub-catenaries of span
$S/{scenario.num_points - 1}$ and length $L/{scenario.num_points - 1}$.
For this scenario $a = {sol.a:.4f}$\,m and the sag is
${sol.sag * 1e3:.1f}$\,mm.

\paragraph{{Validity.}} The catenary is the $EI \to 0$ limit. Bending competes
with gravity over the elasto-gravitational length
$\ell = (EI/w)^{{1/3}} = {l_eg * 1e3:.1f}$\,mm. {validity}

\section{{Results}}

\begin{{table}}[htbp]
\centering
\small
\begin{{tabular}}{{lrrrrrrc}}
\toprule
Method & RMSE & max err & sag & sag err & arc drift & wall & settled \\
       & [mm] & [mm] & [mm] & [mm] & [\%] & [s] & \\
\midrule
{body_rows}
\bottomrule
\end{{tabular}}
\caption{{Every method against the analytic catenary. Reference sag
${sol.sag * 1e3:.1f}$\,mm. \emph{{Arc drift}} is the change in the cable's own
length during the run and is reference-free: it measures violation of
inextensibility directly.}}
\end{{table}}
{figure_blocks}
{real_section}

\section{{Reading these numbers}}

\begin{{itemize}}
  \item \textbf{{Arc drift is the honest accuracy check.}} It needs no
        reference shape. A solver that stretches the cable has failed
        regardless of how catenary-like its profile looks.
  \item \textbf{{Agreeing with the catenary is not automatically a win.}}
        Methods with no bending stiffness (a spherical-joint chain, for
        instance) match the ideal catenary because they share its idealisation,
        not because they model the cable more faithfully.
  \item \textbf{{An unsettled run is provisional.}} Where \emph{{settled}} is
        blank the motion had not decayed within the time cap, and the profile
        is a time-average over the final window rather than a true equilibrium.
\end{{itemize}}

\end{{document}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description="Compare cable-hang benchmark methods.")
    p.add_argument("--run", required=True, help="run directory from run_benchmark.py")
    p.add_argument("--real", default=None,
                   help="real cable profile.csv from image_utils/extract_cable_profile.py")
    p.add_argument("--pdf", action="store_true",
                   help="also compile report.tex with pdflatex if available")
    args = p.parse_args()

    scenario, results = load_run(args.run)
    if not results:
        print(f"!! no method results found in {args.run}")
        return 1
    real = read_profile(args.real) if args.real else None

    rows = compute_metrics(scenario, results, real)
    write_metrics_csv(os.path.join(args.run, "metrics.csv"), rows)
    figures = make_figures(args.run, scenario, results, real, rows)
    report = os.path.join(args.run, "report.tex")
    write_report(report, args.run, scenario, results, rows, args.real, figures)

    # -- console table --
    sol = scenario.catenary()
    print(f"\n{scenario.describe()}")
    print(f"catenary: a={sol.a:.4f} m, sag={sol.sag * 1e3:.1f} mm\n")
    hdr = f"{'method':16s} {'RMSE':>8s} {'max':>8s} {'sag':>8s} {'sagerr':>8s} {'arc%':>8s} {'wall':>7s}  settled"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['method']:16s} {r['rmse_mm']:8.2f} {r['max_err_mm']:8.2f} "
              f"{r['sag_mm']:8.1f} {r['sag_err_mm']:+8.1f} {r['arc_drift_pct']:+8.3f} "
              f"{r['wall_s']:7.1f}  {'yes' if r['settled'] else 'NO'}")
    print(f"\nwrote {args.run}/metrics.csv, report.tex, " + ", ".join(
        os.path.basename(f) for f in figures))

    if args.pdf:
        import shutil
        import subprocess
        if shutil.which("pdflatex"):
            subprocess.call(["pdflatex", "-interaction=nonstopmode", "report.tex"],
                            cwd=args.run,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"wrote {args.run}/report.pdf")
        else:
            print("[warn] pdflatex not found; report.tex was still written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
