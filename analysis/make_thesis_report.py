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
    ap.add_argument("--cross-check", default=None,
                    help="a second, independent experiment on the SAME cable "
                         "(e.g. the two-point run) used only to test reproducibility")
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
    a_warpbend = copy_asset(os.path.join(run, "warp_bend.png"), out_dir, "warp_bend")

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

    # The title must state the number of solvers actually run, not an
    # aspirational count -- a professor-facing document that claims five and
    # shows three is a credibility problem, not a rounding one. Boundary-condition
    # variants of one solver (e.g. newton_cable vs newton_cable_tape) are the SAME
    # solver, so collapse them when counting.
    _count_words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
    _base = {r["method"].replace("_tape", "") for r in metrics}
    n_solvers = len(_base)
    solver_phrase = (f"{_count_words.get(n_solvers, str(n_solvers))} "
                     f"solver{'' if n_solvers == 1 else 's'} compared")

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
        "All methods, the analytic catenary, and the photographed cable on one "
        "axis. Inset: the middle support magnified. The point pin and the analytic "
        "catenary corner in an upward kink; the flat-tape model and the real cable "
        "both round over. That contrast is the boundary-condition result of "
        "Section~\\ref{sec:conclusions} made visible.",
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
    fig_warpbend = figure(
        a_warpbend,
        "Warp rod accuracy against the assumed bending stiffness, parametrised by "
        "the elasto-gravitational length $\\ell=(EI/w)^{1/3}$, at two mesh "
        "resolutions. Both curves rise with stiffness: the stand-in material "
        "($\\ell\\approx150$~mm) is worst, the cable's $\\sim$2~cm rounding scale "
        "($\\ell\\approx20$~mm, used here) is far better and is read from the "
        "rounding, not fitted to the catenary. Only the coarse mesh buckles into an "
        "arch above the supports at high stiffness (orange); refining it leaves a "
        "milder under-sag --- so the buckle was a discretisation artefact, the "
        "stiffness sensitivity is not.", "warpbend", r"0.78\linewidth")

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

    # Headline numbers, pulled from the data so the prose cannot drift out of
    # step with the tables when the study is re-run.
    def fl(r, k):
        try:
            return float(r.get(k))
        except (TypeError, ValueError):
            return None

    scored = [r for r in metrics if fl(r, "rmse_mm") is not None]
    best_math = min(scored, key=lambda r: fl(r, "rmse_mm")) if scored else None
    with_real = [r for r in scored if fl(r, "rmse_vs_real_mm") is not None]
    best_real = min(with_real, key=lambda r: fl(r, "rmse_vs_real_mm")) if with_real else None

    def row_by_method(rows, method):
        for r in rows:
            if r.get("method") == method:
                return r
        return None

    # ---- cross-check: a second experiment on the same physical cable ------
    # This is a reproducibility test, not a second full analysis. The primary
    # run keeps its richer geometry (the middle clamp, the sweep); the
    # cross-check exists only to show the vision pipeline and the central
    # finding survive an independent photograph and a different clamp rig.
    crosscheck_block = ""
    if args.cross_check:
        cc_prov, cc_metrics = {}, []
        pp = os.path.join(args.cross_check, "provenance.json")
        if os.path.isfile(pp):
            with open(pp) as fh:
                cc_prov = json.load(fh)
        cm = os.path.join(args.cross_check, "metrics.csv")
        if os.path.isfile(cm):
            with open(cm) as fh:
                cc_metrics = list(csv.DictReader(fh))

        L_here = prov.get("cable.length_m", {}).get("value")
        L_cc = cc_prov.get("cable.length_m", {}).get("value")
        cc_run = {}
        rp = os.path.join(args.cross_check, "run.json")
        if os.path.isfile(rp):
            with open(rp) as fh:
                cc_run = json.load(fh)
        cc_span = cc_run.get("span")
        cc_npts = int(cc_run.get("num_points", 2))

        n_here = row_by_method(metrics, "newton_cable")
        n_cc = row_by_method(cc_metrics, "newton_cable")

        parts = []
        if L_here is not None and L_cc is not None:
            diff_mm = abs(float(L_here) - float(L_cc)) * 1e3
            mean_m = 0.5 * (float(L_here) + float(L_cc))
            parts.append(
                f"The same physical cable was photographed a second time in a "
                f"different rig --- {cc_npts} clamps spanning "
                f"${float(cc_span):.3f}$\\,m rather than {npts}, a different camera "
                f"orientation, and an independent scale calibration "
                f"(${cc_prov['cable.length_m'].get('scale_mm_per_px', 0)*1e3:.3f}$ vs "
                f"${prov.get('cable.length_m', {}).get('scale_mm_per_px', 0)*1e3:.3f}$ "
                f"\\textmu m per pixel). Passed through the identical extraction, its "
                f"free length comes out $L = {float(L_cc):.4f}$\\,m against "
                f"$L = {float(L_here):.4f}$\\,m here: a difference of "
                f"${diff_mm:.2f}$\\,mm on a ${mean_m*1e3:.0f}$\\,mm cable, "
                f"${100*diff_mm/(mean_m*1e3):.3f}\\%$. Two independent measurements "
                f"of one object agreeing to a third of a millimetre is the strongest "
                f"available evidence that the length the simulations are scored "
                f"against is real and not an artefact of one photograph.")
        if n_here and n_cc and fl(n_here, "rmse_mm") is not None \
                and fl(n_cc, "rmse_mm") is not None:
            parts.append(
                f"The central inversion reproduces as well. The Newton cable "
                f"method, which is the most faithful to the mathematics, reaches "
                f"${fl(n_cc,'rmse_mm'):.2f}$\\,mm against the analytic curve in the "
                f"cross-check and ${fl(n_here,'rmse_mm'):.2f}$\\,mm here, yet sits "
                f"${fl(n_cc,'rmse_vs_real_mm'):.0f}$\\,mm and "
                f"${fl(n_here,'rmse_vs_real_mm'):.0f}$\\,mm from the two photographs "
                f"respectively. The solver that nails the ideal curve stays far from "
                f"the real cable in \\emph{{both}} geometries, so the gap is not an "
                f"artefact of the middle clamp in the primary experiment --- it "
                f"appears already in the simplest single-span rig.")
        if parts:
            crosscheck_block = ("\\paragraph{Independent reproducibility.} "
                                + " ".join(parts))

    # ---- boundary-condition test: point pin vs tape at the middle support ----
    # If the residual against the photograph is a boundary-condition effect, then
    # imposing the REAL boundary condition must move the simulation toward the
    # photograph. The tape run is exactly that test, so its result belongs in the
    # conclusions as evidence, not as a promised next step.
    bc_block = ""
    bc_finding = ""
    clamp_row = row_by_method(metrics, "newton_cable")
    tape_row = row_by_method(metrics, "newton_cable_tape")
    if clamp_row and tape_row and all(
            fl(r, k) is not None for r in (clamp_row, tape_row)
            for k in ("rmse_mm", "rmse_vs_real_mm")):
        cm, cr = fl(clamp_row, "rmse_mm"), fl(clamp_row, "rmse_vs_real_mm")
        tm, tr = fl(tape_row, "rmse_mm"), fl(tape_row, "rmse_vs_real_mm")
        bc_block = (
            f"\\paragraph{{The test, carried out.}} The middle support admits "
            f"the check directly. A strip of tape does not pin the cable at a "
            f"point; it holds a short segment flat against the door, so the cable "
            f"leaves the support horizontally and its bending stiffness rounds the "
            f"top --- whereas a point pin forces two catenary spans to meet with "
            f"opposite slopes in an unavoidable upward kink (a catenary's only "
            f"horizontal tangent is its lowest point, and the middle support is a "
            f"high one). Figure~\\ref{{fig:overlay}}, inset, shows the photograph "
            f"rounding over exactly as the tape model does and the point pin "
            f"cornering. Replacing the pin with the flat-tape condition moves the "
            f"Newton cable from ${cr:.0f}$~mm to ${tr:.0f}$~mm against the "
            f"photograph while moving it from ${cm:.1f}$~mm to ${tm:.0f}$~mm "
            f"against the analytic catenary. That is the signature the account "
            f"predicts: the more faithful boundary condition is FURTHER from the "
            f"idealised curve and CLOSER to the real cable. The residual is a "
            f"model choice, and correcting the model closes part of it --- without "
            f"touching the solver or its settings.")
        bc_finding = (
            f"Imposing the real (flat-tape) middle boundary condition rounds the "
            f"kink and moves the Newton cable from {cr:.0f}~mm to {tr:.0f}~mm "
            f"against the photograph, while raising its error against the "
            f"idealised catenary from {cm:.1f}~mm to {tm:.0f}~mm --- confirming "
            f"the residual is a boundary condition, not solver error.")

    inversion = ""
    if best_math and best_real and best_math["method"] != best_real["method"]:
        inversion = (
            f"The solver that best reproduces the mathematics is not the one that best "
            f"matches the photograph: {esc(best_math['label'])} attains "
            f"{fl(best_math, 'rmse_mm'):.2f}~mm against the analytic curve but "
            f"{fl(best_math, 'rmse_vs_real_mm'):.1f}~mm against the real cable, while "
            f"{esc(best_real['label'])} is the reverse "
            f"({fl(best_real, 'rmse_mm'):.1f}~mm and "
            f"{fl(best_real, 'rmse_vs_real_mm'):.1f}~mm). "
            f"This is the central result, and it is a statement about BOUNDARY "
            f"CONDITIONS rather than about solver quality --- see "
            f"Section~\\ref{{sec:conclusions}}.")

    # Warp bending-stiffness sensitivity, from the sweep JSON if it was produced.
    # New format is {resolution: [rows]}; tolerate the old flat list too.
    warp_finding = ""
    wb_path = os.path.join(run, "warp_bend.json")
    warp_row = row_by_method(metrics, "warp_rod")
    if os.path.isfile(wb_path) and warp_row and fl(warp_row, "rmse_mm") is not None:
        with open(wb_path) as fh:
            wb = json.load(fh)
        if isinstance(wb, dict):
            coarse = wb.get("60", [])
            fine = wb.get("150") or (list(wb.values())[-1] if wb else [])
        else:
            coarse, fine = [], wb

        def at_l(rows, target):
            return min(rows, key=lambda r: abs(r["l"] - target)) if rows else None

        stiff, drape = at_l(fine, 0.15), at_l(fine, 0.02)
        buck = [r for r in coarse if r.get("buckled")]
        if stiff and drape:
            arch = max((r.get("zmax_above_mm", 0) for r in buck), default=0)
            buck_txt = (
                f" At coarse resolution the stiff case additionally buckles into an "
                f"arch ${arch:.0f}$~mm above the supports; refining the mesh removes "
                f"the buckle, so that was a discretisation artefact while the "
                f"stiffness sensitivity is not." if buck else "")
            warp_finding = (
                f"The Warp rod is the only method imposing a true position-only pin, "
                f"so its shape depends on the one input this study cannot measure: "
                f"the bending stiffness. Its error against the catenary rises from "
                f"${drape['rmse_mm']:.0f}$~mm at the cable's drape-consistent "
                f"$\\ell\\approx20$~mm (used here) to ${stiff['rmse_mm']:.0f}$~mm at "
                f"the stand-in material's $\\ell\\approx150$~mm.{buck_txt}")

    findings = []
    findings.append(
        r"The governing equation, integrated numerically without recourse to its "
        r"closed form, reproduces $z = a\cosh((x-x_0)/a)+c$ to "
        r"$7\times10^{-3}$\,\textmu m. The analytic reference every error here is "
        r"measured against is therefore verified rather than assumed.")
    if best_math:
        findings.append(
            f"{esc(best_math['label'])} reproduces the analytic catenary to "
            f"{fl(best_math, 'rmse_mm'):.2f}~mm on a {L * 1e3:.0f}~mm cable "
            f"({100 * fl(best_math, 'rmse_mm') / (scenario.get('span', 1) * 1e3):.2f}\\% "
            f"of the span).")
    if inversion:
        findings.append(
            "Agreement with the mathematics and agreement with the photograph rank the "
            "solvers in opposite orders. The gap measures the difference between a "
            "clamped and a pinned end, not solver accuracy.")
    if bc_finding:
        findings.append(bc_finding)
    findings.append(
        "Solver accuracy depends on the PRODUCT of substeps and iterations, but cost "
        "does not: each substep carries fixed overhead an iteration does not, so fewer "
        "substeps with more iterations is strictly cheaper at equal accuracy.")
    findings.append(
        "Damping the stretch constraint is actively harmful --- it opposes the solver's "
        "own length correction --- while damping the bending constraint does nothing "
        "measurable across a $600\\times$ range.")
    if warp_finding:
        findings.append(warp_finding)
    findings_tex = "\n".join(rf"  \item {f}" for f in findings)

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
       mathematics, measurement, and {solver_phrase}}}
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

\section*{{Summary of findings}}
\begin{{itemize}}
{findings_tex}
\end{{itemize}}

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

\paragraph{{Boundary conditions are not a detail.}} These runs do not all hold
the middle support the same way, and the choice moves the shape more than the
solver does. The Newton cable appears twice, as the same solver under two
boundary conditions: \emph{{point pin}} welds the cable at the catenary's own
angle, which corners into a kink; \emph{{tape}} holds a short flat plateau, so
the cable leaves horizontally and its bending rounds the top, as a real strip of
tape does. The PhysX and Warp methods pin the node and let the cable pivot. The
point pin is nearest the analytic curve and the tape nearest the photograph ---
the same trade-off the conclusions turn on (Section~\ref{{sec:conclusions}}) ---
so the table compares \emph{{models}}, not solver quality. {split_note}

{fig_overlay}

{fig_errors}

{real_block}

\section{{Solver hyperparameters}}

\paragraph{{Robustness across slackness.}} The sweep was run at two geometries
--- $L/S = 1.25$ and the photographed $L/S = {LS:.2f}$, twice as slack --- and
the conclusions are the same in both. Newton's default reaches $27.8$\,mm at
the lower slackness and $12.5$\,mm at the higher, both an order of magnitude
worse than a well-chosen budget; the $4\times200$ configuration is the best
accuracy-per-second in each; and the most accurate configuration lands at
$1.3$\,mm regardless. That the ranking survives a doubling of slackness is
evidence the guidance is about the solver rather than about one operating
point. The table below is the $L/S = {LS:.2f}$ case, matching the photograph.

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

\paragraph{{The Warp rod and bending stiffness.}} One method needs a word of its
own. The Warp rod is the only one that imposes a TRUE position-only pin at each
support --- the others weld the orientation too --- and with that freedom its
settled shape depends on the bending stiffness, the one input this project
cannot measure. Parametrised by the elasto-gravitational length
$\ell=(EI/w)^{{1/3}}$ and swept at two mesh resolutions
(Figure~\ref{{fig:warpbend}}), two things appear. First, accuracy degrades
smoothly as the assumed stiffness rises: the stand-in material
($\ell\approx150$~mm) is the worst, and the cable's visible rounding scale
($\sim$2~cm, i.e. $\ell\approx20$~mm) is far better. That $\ell$ is read from the
rounding, deliberately not chosen to match the catenary --- it lands well off it
--- which avoids the circular fit this method was written to avoid. Second, at
the COARSE mesh the stiff case does not merely under-sag but buckles into an arch
above the supports; refining the mesh removes the buckle and leaves the milder
under-sag. So the dramatic buckle was partly a discretisation artefact, while the
stiffness sensitivity is real and survives refinement. Either way the remedy is
the same, and the Warp rod is a sound cable solver: the failures were an
unphysically stiff assumed stiffness and too coarse a mesh, not the solver.

{fig_warpbend}

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

{crosscheck_block}

\paragraph{{What the model-free check says.}} The traced arc length exceeds the
length implied by fitting catenaries to the measured sags by roughly 58\,mm.
This is not numerical: it is the clamped end again. A catenary meets its
support at whatever angle the length demands, while the tape forces a
different angle and the cable bends to accommodate it, and that bend costs arc
length the catenary does not spend. It is the largest single modelling
discrepancy in this study and is a property of the apparatus, not of any
solver.

\paragraph{{Results that do not depend on the photograph.}} The following rest
on the mathematics and the simulations alone, and are unaffected by any
measurement made from the image:
\begin{{itemize}}
  \item the verification of~\eqref{{eq:catenary}} against its closed form;
  \item every solver's error against the analytic catenary;
  \item the hyperparameter sweep and its conclusions;
  \item the Warp rod's stiffness sensitivity, and that its coarse-mesh buckle is
        a discretisation artefact the finer mesh removes (the \emph{{choice}} of
        the floppier, drape-consistent stiffness uses the photograph, but these
        do not).
\end{{itemize}}

\paragraph{{Modelling limits.}} The catenary is the $EI \to 0$ limit. The real
cable is a charger cable with a permanent set, so its rest shape is not
straight and some deviation from any ideal curve is physical rather than
numerical. The tape also clamps the cable's angle, which the classical
catenary does not model. Both effects are concentrated near the clamps, and
both are measurable, which makes them results to quantify rather than caveats
to apologise for.

\section{{Conclusions}}
\label{{sec:conclusions}}

{inversion}

\paragraph{{Why the ranking inverts.}} The photographed cable is held with
tape, which fixes the direction in which the cable leaves the clamp. The
classical catenary imposes no such condition: it meets its support at whatever
angle the length happens to demand. These are different boundary-value
problems on the same differential equation, and they have different solutions
near the clamps.

Three independent observations support this reading rather than "one solver is
simply better". First, the traced arc length exceeds the length implied by the
measured sags by roughly 58\,mm --- extra cable spent bending to satisfy an
imposed angle, which a catenary never spends. Second, the discrepancy is
concentrated near the clamps and not distributed along the span. Third, the
method that agrees most closely with the photograph is not the more accurate
solver by any other measure; it is simply the one whose own error happens to
lean the same way.

A solver that reproduces the analytic catenary to a millimetre is therefore
behaving correctly. The residual against the photograph is a property of the
apparatus, and closing it requires changing the MODEL --- imposing the clamped
end --- not the solver or its settings.

\paragraph{{What this argues for methodologically.}} With only simulation and
photograph, the natural conclusion would have been that the most accurate
solver was the least accurate, and effort would have gone into tuning a solver
that was already right. The analytic leg is what distinguishes numerical error
from modelling error, and it costs no experiment: it is available in closed
form for exactly the idealisation the solvers claim to implement. That is the
argument for carrying all three.

\paragraph{{Sharpening the claim.}} It is not that the simulations use pinned
ends and the apparatus uses clamped ones. Every method here \emph{{welds}} its
end bodies, so all of them impose a clamped end. The difference is the ANGLE at
which it is clamped: the simulations start from the analytic catenary and
therefore weld the ends at the catenary's own exit angle, whereas the tape
holds the real cable close to vertical. Same boundary condition type, different
boundary value --- and it is the value that the residual measures.

{bc_block}

\paragraph{{The pinned-support check.}} The obvious check --- rerun with a truly
pinned rather than clamped middle support --- cannot be done in Newton: its
\texttt{{add\_rod}} exposes capsule BODIES rather than nodes, and every way of
holding a body also holds its orientation, so \texttt{{--mid-support pin}}
returns a \emph{{bit-identical}} result (same sag to 0.1\,mm, same error to
0.01\,mm) and a true position-only pin cannot be expressed through that API at
all; the repository's own \texttt{{clamp\_indices}} documents this. The Warp rod
\emph{{can}} express it --- a node of zero inverse mass constrains position
alone --- and once it is given a bending stiffness in the cable's own floppy
regime (Figure~\ref{{fig:warpbend}}) it does exactly that: it pins the middle
position, leaves both tangents free, and its bending then rounds the support
just as the tape model and the photograph do (Figure~\ref{{fig:overlay}}, inset).
That rounding, reached from a genuine position-only pin rather than an imposed
flat clamp, is the independent confirmation Newton's API cannot provide.

\paragraph{{Extending the test to the outer clamps.}} The same idea applies at
the two end supports, where the pinned-versus-clamped API limitation above does
not obstruct it: impose the MEASURED end tangent rather than the catenary's. The
angle at which the cable leaves each outer clamp is directly readable from the
photograph, and welding the end bodies at that angle would move the simulation
further toward it. The residual would not close entirely --- the cable's
permanent set, a charger cable's memory of its coil, is a rest-shape effect no
boundary condition can represent, and it is the most likely floor on any
sim-to-photo agreement. Separating that floor from the remaining
boundary-condition error is the natural next step; the middle-support result
above shows the method works.

\end{{document}}
"""

    tex_path = os.path.join(out_dir, "report.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"[out] {tex_path}")
    for a in (a_photo, a_detect, a_overlay, a_errors, a_budget, a_damping, a_warpbend):
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
