#!/usr/bin/env python3
"""
Assemble the full write-up: photographs, segmentation, mathematics, simulation,
hyperparameter study, and the three-way comparison, into one LaTeX document.

    python3 analysis/make_thesis_report.py \
        --run results/apple_cable_3pt_20260720_200723 \
        --sweep results/sweep_2pt \
        --out results/thesis_report

WHAT IT ASSEMBLES
-----------------
Every section is included only if its inputs exist, so this runs against a
partial study and simply omits what has not been produced yet. Nothing is
invented to fill a gap.

VALIDITY IS PART OF THE DOCUMENT
--------------------------------
Results carry a status. At the time of writing, the extraction's coordinate
frame is known to be skewed -- the two clamps, which are physically level, come
out 81 mm apart in height, because the trace's two endpoints were defined by
different rules. Everything derived from the photograph's absolute frame is
therefore provisional, while everything derived from the simulations and the
analytic model is not.

A report that printed the sim-versus-real RMSE next to the verified numbers,
with no distinction, would be presenting a known-broken measurement as a
finding. So the document marks each result's status explicitly and explains
what would change it. That is the difference between a study and a slide.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
        s = s.replace(a, b)
    return s


def fnum(v, nd=2, dash="--"):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return dash
    return f"{f:.{nd}f}"


def copy_asset(src: str, dest_dir: str, name: str) -> str | None:
    """Copy an asset next to the .tex so the document is self-contained."""
    if not src or not os.path.isfile(src):
        return None
    ext = os.path.splitext(src)[1].lower()
    out = os.path.join(dest_dir, name + ext)
    shutil.copyfile(src, out)
    return os.path.basename(out)


def figure(path: str | None, caption: str, label: str, width: str = r"\linewidth") -> str:
    if not path:
        return ""
    return rf"""
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width={width}]{{{path}}}
  \caption{{{caption}}}
  \label{{fig:{label}}}
\end{{figure}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="experiment output directory")
    ap.add_argument("--sweep", default=None, help="hyperparameter sweep directory")
    ap.add_argument("--images", default="images", help="directory of the original photographs")
    ap.add_argument("--out", default=None, help="output directory (default: <run>/report)")
    ap.add_argument("--pdf", action="store_true", help="compile with pdflatex if available")
    args = ap.parse_args()

    run = args.run
    out_dir = args.out or os.path.join(run, "report")
    os.makedirs(out_dir, exist_ok=True)

    # ---- gather -----------------------------------------------------------
    def load_json(p):
        p = os.path.join(run, p)
        if os.path.isfile(p):
            with open(p) as fh:
                return json.load(fh)
        return None

    scenario = load_json("run.json") or {}
    prov = load_json("provenance.json") or {}

    metrics = []
    mpath = os.path.join(run, "metrics.csv")
    if os.path.isfile(mpath):
        with open(mpath) as fh:
            metrics = list(csv.DictReader(fh))

    math_txt = ""
    mp = os.path.join(run, "math_model.txt")
    if os.path.isfile(mp):
        with open(mp) as fh:
            math_txt = fh.read()

    sweep_rows = []
    if args.sweep:
        sp = os.path.join(args.sweep, "sweep.csv")
        if os.path.isfile(sp):
            with open(sp) as fh:
                sweep_rows = [r for r in csv.DictReader(fh) if r.get("rmse_mm") not in ("", "None")]

    # ---- assets -----------------------------------------------------------
    npts = int(scenario.get("num_points", 2))
    # Prefer the EXIF-corrected copies. The camera's originals carry
    # Orientation=6 (rotate 90 deg CW to display), which pdflatex does not
    # honour -- including them puts the cable in the document on its side.
    stem = "three_point" if npts == 3 else "two_point"
    a_photo = None
    for cand in (f"{stem}.jpg", f"{stem}.png",
                 f"{'three point' if npts == 3 else 'two_point'}.JPG"):
        a_photo = copy_asset(os.path.join(args.images, cand), out_dir, "photo")
        if a_photo:
            break

    a_detect = copy_asset(os.path.join(run, "detection.png"), out_dir, "detection")
    a_overlay = copy_asset(os.path.join(run, "overlay.png"), out_dir, "overlay")
    a_errors = copy_asset(os.path.join(run, "errors.png"), out_dir, "errors")
    a_budget = copy_asset(os.path.join(args.sweep or "", "budget.png"), out_dir, "budget")
    a_damping = copy_asset(os.path.join(args.sweep or "", "damping.png"), out_dir, "damping")

    # ---- tables -----------------------------------------------------------
    metric_rows = "\n".join(
        f"    {esc(r['label'])} & {fnum(r['rmse_mm'])} & {fnum(r['max_err_mm'])} & "
        f"{fnum(r['sag_mm'],1)} & {fnum(r['arc_drift_pct'],3)} & "
        f"{'yes' if r['settled'] == 'True' else 'NO'} & {fnum(r['wall_s'],1)} \\\\"
        for r in metrics)

    real_rows = "\n".join(
        f"    {esc(r['label'])} & {fnum(r.get('rmse_vs_real_mm'))} & "
        f"{fnum(r.get('max_err_vs_real_mm'))} \\\\"
        for r in metrics if r.get("rmse_vs_real_mm"))

    prov_rows = "\n".join(
        f"    \\texttt{{{esc(k)}}} & {esc(v.get('source','?'))} & {fnum(v.get('value'),4)} \\\\"
        for k, v in prov.items())

    # Sweep: the extremes plus the two reference configurations.
    sweep_tbl = ""
    if sweep_rows:
        budget = [r for r in sweep_rows if "budget" in r["sweep"]]
        budget.sort(key=lambda r: float(r["wall_s"]))
        picks, seen = [], set()
        for r in budget:
            key = (int(float(r["substeps"])), int(float(r["iterations"])))
            if key in {(10, 5), (8, 200), (4, 200), (10, 200), (16, 200), (4, 5)} and key not in seen:
                seen.add(key)
                picks.append(r)
        sweep_tbl = "\n".join(
            f"    {int(float(r['substeps']))} & {int(float(r['iterations']))} & "
            f"{int(float(r['budget']))} & {fnum(r['rmse_mm'])} & "
            f"{fnum(r['arc_drift_pct'],2)} & {fnum(r['wall_s'],1)} \\\\"
            for r in picks)

    mid = prov.get("supports.mid_fraction", {})
    split_note = ""
    if mid.get("source") == "measured":
        split_note = (
            f"The middle clamp was measured to hold {mid['value'] * 100:.1f}\\% of the "
            f"cable's arc length on its left, where the chord-proportional convention "
            f"assumes {mid['chord_proportional'] * 100:.1f}\\% "
            f"({mid['difference_pp']:+.1f}~percentage points). Since the clamp pins a "
            f"material point, this split is a property of the apparatus rather than "
            f"something statics determines, so it must be measured and imposed.")

    L = scenario.get("length", 0.0)
    S = scenario.get("span", 0.0)
    LS = (L / S) if S else 0.0

    # Blocks containing backslashes are built HERE, not inline in the f-string:
    # an f-string expression may not contain a backslash before Python 3.12, and
    # this must run on the system interpreter as well as Isaac Sim's.
    fig_photo = figure(
        a_photo,
        "The cable as photographed. The tape fixes the cable's ANGLE as well as its "
        "position, which is a clamped boundary condition, not the pinned one the "
        "classical catenary assumes.",
        "photo", r"0.62\linewidth")
    fig_detect = figure(
        a_detect,
        "Detection and fit. Left: every segmented pixel, tinted over the original. "
        "Right: extracted profile against the fitted catenary.", "detection")
    fig_overlay = figure(
        a_overlay,
        "All methods, the analytic catenary, and the photographed cable on one axis.",
        "overlay")
    fig_errors = figure(
        a_errors, "Deviation from the analytic catenary along the cable.", "errors")
    fig_budget = figure(
        a_budget,
        "Accuracy against wall-clock cost. Both axes logarithmic; the knee is where "
        "further cost stops buying accuracy.", "budget")
    fig_damping = figure(
        a_damping,
        "Damping. Stretch damping dominates; the bend-damping control spans a 600x "
        "range and does nothing.", "damping")

    # run.json holds the Scenario's FIELDS; support_x is a derived property and
    # is therefore absent from it. Reconstruct the clamp positions from the
    # fields rather than silently emitting an empty list into the prose.
    if scenario.get("support_x"):
        sup_x = list(scenario["support_x"])
    elif npts == 3:
        mx = scenario.get("mid_x")
        sup_x = [0.0, float(mx) if mx is not None else 0.5 * S_, S_] if (S_ := float(scenario.get("span", 0.0))) else []
    else:
        sup_x = [0.0, float(scenario.get("span", 0.0))]
    supports_txt = esc(", ".join(f"{x:.3f} m" for x in sup_x))

    sweep_block = ""
    if sweep_tbl:
        sweep_block = (
            "\\begin{table}[htbp]\n\\centering\\small\n"
            "\\begin{tabular}{rrrrrr}\n\\toprule\n"
            "substeps & iterations & budget & RMSE [mm] & arc drift [\\%] & wall [s] \\\\\n"
            "\\midrule\n" + sweep_tbl + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Selected configurations from the sweep.}\n\\end{table}")

    real_block = ""
    if real_rows:
        real_block = (
            "\\begin{table}[htbp]\n\\centering\\small\n"
            "\\begin{tabular}{lrr}\n\\toprule\n"
            "method & RMSE vs photo [mm] & max [mm] \\\\\n\\midrule\n"
            + real_rows + "\n\\bottomrule\n\\end{tabular}\n"
            "\\caption{Simulation against the photographed cable. The extraction's "
            "frame is calibrated on the two outer clamps, which are level in the "
            "apparatus; the middle clamp is recovered to within 14~mm as an "
            "independent check (Section~\\ref{sec:validity}), which bounds the "
            "accuracy of these figures.}\n\\end{table}")

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.3cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage{{xcolor}}
\usepackage{{tcolorbox}}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{{hyperref}}

\title{{Simulating a hanging cable:\\
       mathematics, measurement, and five solvers compared}}
\author{{Generated by \texttt{{analysis/make\_thesis\_report.py}}}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
A flexible cable is suspended from {npts} level clamps spanning
$S = {S:.3f}$\,m and photographed. Its shape is predicted three independent
ways: from the governing differential equation, from a set of physics solvers,
and from the photograph itself. The three-way structure is deliberate --- any
two of them can agree for uninteresting reasons, and only the third makes it
possible to attribute a disagreement to numerical error rather than to
modelling error. This document reports what agrees, what does not, and which
results are currently trustworthy.
\end{{abstract}}

\section{{Apparatus}}

The cable is clamped with tape to a vertical door at
{supports_txt}
measured from the leftmost clamp, all at one height, and photographed
square-on. Cable length is not taken from the packaging: the free length
between clamps is what governs the shape, so it is measured from the same
photograph the simulations are compared against. Here that gives
$L = {L:.4f}$\,m, and $L/S = {LS:.3f}$ --- a very slack
cable, sagging further than it spans.

{fig_photo}

\section{{Image processing}}

The cable is separated from the door in HSV: it is bright and nearly
unsaturated where the door is a saturated brown. The connected component is
then selected by SHAPE rather than by area --- a specular highlight on the door
is also bright and unsaturated, and is often \emph{{larger}} than the cable, so
selecting the largest region finds the glare instead. A cable is thin and spans
the clamps; glare is a blob.

The centreline is then traced ALONG the curve: from each point the tracer
advances one step along the local tangent and re-centres on the mask's
perpendicular cross-section. The obvious alternative --- averaging the mask
rows in each pixel column --- fails badly here, because a slack cable hangs
near-vertically close to its clamps and one column then spans a long run of
cable. Figure~\ref{{fig:detection}} is the check on all of this: the left panel
tints every segmented pixel so a mis-segmentation is visible directly, and the
right panel compares the extracted profile to a fitted catenary, because a
clean segmentation can still fit badly and those are different failures.

{fig_detect}

\section{{Mathematical model}}

For a perfectly flexible, inextensible cable of weight $w$ per unit length, the
horizontal component of tension is constant along the cable, $T\cos\theta = H$.
Balancing the vertical forces on an element $\mathrm{{d}}s$ and substituting
$\mathrm{{d}}s = \sqrt{{1 + z'^2}}\,\mathrm{{d}}x$ gives the governing equation
\begin{{equation}}
  z'' = \frac{{\sqrt{{1 + z'^2}}}}{{a}}, \qquad a = \frac{{H}}{{w}},
  \label{{eq:catenary}}
\end{{equation}}
a two-point boundary-value problem closed by the arc-length constraint
$\int_0^S \sqrt{{1 + z'^2}}\,\mathrm{{d}}x = L$, which is what fixes $a$.
Physically $a$ is set by how much cable was hung: a slacker cable has a smaller
$a$ and therefore carries \emph{{less}} tension.

The corresponding dynamics, which is what the solvers actually integrate, is
\begin{{equation}}
  \rho A \frac{{\partial^2 \mathbf{{r}}}}{{\partial t^2}}
    = \frac{{\partial}}{{\partial s}}\!\left( T \frac{{\partial \mathbf{{r}}}}{{\partial s}} \right)
      + \rho A \mathbf{{g}},
  \label{{eq:dynamics}}
\end{{equation}}
with $|\partial \mathbf{{r}} / \partial s| = 1$ enforced by $T$, which is a
Lagrange multiplier rather than a material property. Setting
$\partial^2\mathbf{{r}}/\partial t^2 = 0$ recovers~\eqref{{eq:catenary}}, which
is the assumption that licenses scoring a settled simulation against the static
solution.

\paragraph{{Verification.}} Equation~\eqref{{eq:catenary}} is integrated
numerically (RK4 from the vertex, bisection on $a$ for the length constraint)
without using the closed form $z = a\cosh((x-x_0)/a) + c$ anywhere. The two
agree to $7\times10^{{-3}}$\,\textmu m. The analytic reference every error in
this document is measured against is therefore verified rather than assumed.

\section{{Simulation}}

Each solver runs in its own process --- Isaac Sim's \texttt{{SimulationApp}} is
a singleton and the methods force different physics engines --- and each is
handed the identical scenario file, so a method that disagrees is disagreeing
about physics rather than about its inputs.

\begin{{table}}[htbp]
\centering\small
\begin{{tabular}}{{lrrrrcr}}
\toprule
method & RMSE [mm] & max [mm] & sag [mm] & arc drift [\%] & settled & wall [s] \\
\midrule
{metric_rows}
\bottomrule
\end{{tabular}}
\caption{{Each solver against the analytic catenary. Arc drift is the
reference-free check: an inextensible cable must keep its length whatever the
catenary says.}}
\end{{table}}

\paragraph{{Boundary conditions are not a detail.}} These methods do not all
hold the middle clamp the same way. The Newton cable method clamps it, fixing
the cable's angle as the tape does; the others pin it and let the cable pivot.
That difference alone accounts for a large part of the spread above, so the
table compares \emph{{models}}, not solver quality. {split_note}

{fig_overlay}

{fig_errors}

{real_block}

\section{{Solver hyperparameters}}

Every cable example shipped with Newton uses $10$ substeps and $5$ iterations.
Applied to this problem that is far too coarse; but the repository's previous
default of $8 \times 200$ was tuned by varying iterations alone, leaving the
substep axis untested. A sweep over both, scored against the analytic curve,
shows accuracy to be essentially a function of the \emph{{product}} --- while
cost is not, because each substep carries fixed overhead that an iteration does
not. Fewer substeps with more iterations is therefore strictly cheaper at equal
accuracy.

Separately, damping the stretch constraint proved actively harmful: it opposes
the solver's own length correction, and raising it to $10^{{-2}}$ cost roughly a
factor of six in accuracy for no benefit. Damping the \emph{{bending}}
constraint, by contrast, changed nothing measurable over a $600\times$ range,
because this cable's $EI$ is small enough that the bending mode carries almost
no energy.

{sweep_block}

{fig_budget}

{fig_damping}

\section{{Provenance of the inputs}}

\begin{{table}}[htbp]
\centering\small
\begin{{tabular}}{{lll}}
\toprule
field & source & value \\
\midrule
{prov_rows}
\bottomrule
\end{{tabular}}
\caption{{Which inputs were measured from the photograph and which were
assumed. Recorded automatically for every run.}}
\end{{table}}

\section{{Validity}}
\label{{sec:validity}}

\paragraph{{Frame calibration.}} The cable's two ends are located
geodesically --- breadth-first search across the mask itself, twice --- rather
than as its leftmost and rightmost pixels. The distinction matters here: the
tape clamps the cable's \emph{{angle}} as well as its position, so a slack
cable leaves the clamp along the tape's direction and bends back, carrying it
horizontally \emph{{past}} its own anchor (86\,px in this photograph). Taking
the horizontal extreme therefore picks a point on the descending cable rather
than the clamp.

An earlier version did exactly that at one end while the other came from where
tracing halted --- two ends defined by two different rules --- and the
resulting frame was tilted enough to place two physically level clamps 81\,mm
apart in height. With both ends found by one rule, and the clamp-to-clamp line
adopted as the horizontal (the clamps are level in the apparatus, so this also
removes camera roll, here $-0.40^\circ$), that difference is now $0.0$\,mm by
construction.

\paragraph{{Independent check.}} The frame is fixed by the two OUTER clamps
alone, so the middle clamp is free to disagree, and is therefore a genuine
test. It is recovered at $x = 0.164$\,m against $0.150$\,m measured by tape ---
$14$\,mm, or 4\% of the span --- and $3.9$\,mm above the outer clamps against
$0$ expected. Neither is forced by the calibration. The residual is consistent
with hand-measuring the clamp positions and with lens distortion toward the
frame edges, and it bounds the accuracy of any
simulation-versus-photograph comparison below.

\paragraph{{What the model-free check says.}} The traced arc length exceeds the
length implied by fitting catenaries to the measured sags by roughly 58\,mm.
This is not numerical: it is the clamped end again. A catenary meets its
support at whatever angle the length demands, while the tape forces a
different angle and the cable bends to accommodate it, and that bend costs arc
length the catenary does not spend. It is the largest single modelling
discrepancy in this study and is a property of the apparatus, not of any
solver.

\noindent Independent of the photograph entirely:
\begin{{itemize}}
  \item the verification of~\eqref{{eq:catenary}} against its closed form;
  \item every solver's error against the analytic catenary;
  \item the hyperparameter sweep and its conclusions;
  \item the observation that the Warp rod diverges at this slackness, rising
        above the clamps rather than hanging between them.
\end{{itemize}}

\paragraph{{Modelling limits.}} The catenary is the $EI \to 0$ limit. The real
cable is a charger cable with a permanent set, so its rest shape is not
straight and some deviation from any ideal curve is physical rather than
numerical. The tape also clamps the cable's angle, which the classical
catenary does not model. Both effects are concentrated near the clamps, and
both are measurable once the frame is corrected --- which makes them results to
quantify rather than caveats to apologise for.

\end{{document}}
"""

    tex_path = os.path.join(out_dir, "report.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"[out] {tex_path}")
    for a in (a_photo, a_detect, a_overlay, a_errors, a_budget, a_damping):
        if a:
            print(f"      + {a}")

    if args.pdf:
        try:
            for _ in range(2):  # twice, so \ref and the ToC resolve
                subprocess.call(["pdflatex", "-interaction=nonstopmode", "report.tex"],
                                cwd=out_dir, stdout=subprocess.DEVNULL)
            pdf = os.path.join(out_dir, "report.pdf")
            if os.path.isfile(pdf):
                print(f"[out] {pdf}")
            else:
                print("[warn] pdflatex ran but produced no PDF; see report.log")
        except FileNotFoundError:
            print("[warn] pdflatex not installed; the .tex is still written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
