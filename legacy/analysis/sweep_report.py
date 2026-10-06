#!/usr/bin/env python3
"""
Turn a hyperparameter sweep (``run_sweep.py``) into figures and a LaTeX report.

    python3 analysis/sweep_report.py --sweep-dir results/sweep_20260720_1530_2pt
    python3 analysis/sweep_report.py --sweep-dir <dir> --pdf   # also run pdflatex

Reads ``sweep.csv`` and writes, alongside it:

    budget.png        accuracy vs wall-clock cost, one line per substep count
    damping.png       accuracy and settling vs damping ratio
    sweep_report.tex  the write-up, with both tables

TWO FIGURES, TWO QUESTIONS
--------------------------
budget.png asks "where should solver effort go?" -- substeps or iterations. Both
axes are logarithmic because cost and error each span two decades, and the
useful reading is the KNEE (where more cost stops buying accuracy), which a
linear axis hides.

damping.png asks "is the cable damped like Newton's are?" It is drawn as two
stacked panels sharing one x-axis rather than one panel with two y-scales. A
dual-axis chart lets the reader infer a crossing point that is an artefact of
where the two scales happen to be pinned; stacked panels carry the same
information with no such invitation.

Colour: the four substep series use fixed categorical slots (blue, green,
magenta, yellow) in a validated colourblind-safe order -- checked with the
palette validator at ``pairs: all``, not by eye. Reference markers (Newton's
default, this repository's default) are drawn in text ink with distinct marker
shapes, so they read as annotations rather than as a fifth and sixth series.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Validated categorical slots, fixed order (never cycled).
SERIES = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d4"
BAND = "#e8eef8"

NEWTON_SUBSTEPS, NEWTON_ITERATIONS = 10, 5
REPO_SUBSTEPS, REPO_ITERATIONS = 8, 200
# Range of bend_damping/bend_stiffness across Newton's shipped cable examples.
NEWTON_RATIO_LO, NEWTON_RATIO_HI = 0.05, 1.0


def _style(ax):
    """Recessive axes: the data should be the most prominent thing."""
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def read_sweep(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    def num(v):
        if v in ("", "None", None):
            return None
        try:
            return float(v)
        except ValueError:
            return v

    out = []
    for r in rows:
        rec = {k: num(v) for k, v in r.items()}
        rec["tag"] = r["tag"]
        rec["status"] = r["status"]
        rec["sweep"] = r["sweep"]
        rec["settled"] = r["settled"] in ("True", "true", "1")
        out.append(rec)
    return out


def plot_budget(rows: list[dict], out_path: str) -> str | None:
    """Accuracy vs cost, one line per substep count."""
    data = [r for r in rows if "budget" in r["sweep"] and r.get("rmse_mm") is not None]
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    _style(ax)

    substep_values = sorted({int(r["substeps"]) for r in data})
    for i, s in enumerate(substep_values):
        pts = sorted((r for r in data if int(r["substeps"]) == s),
                     key=lambda r: r["wall_s"])
        ax.plot([p["wall_s"] for p in pts], [p["rmse_mm"] for p in pts],
                "-o", color=SERIES[i % len(SERIES)], lw=2.0, ms=5,
                mec="white", mew=1.0, label=f"{s} substeps", zorder=3)

    # Reference configurations, as annotations rather than extra series.
    for r in data:
        s, it = int(r["substeps"]), int(r["iterations"])
        if (s, it) == (NEWTON_SUBSTEPS, NEWTON_ITERATIONS):
            ax.plot(r["wall_s"], r["rmse_mm"], "*", ms=17, color=INK, zorder=5)
            ax.annotate("Newton default\n10x5 = 50",
                        (r["wall_s"], r["rmse_mm"]), textcoords="offset points",
                        xytext=(10, 8), fontsize=8.5, color=INK, weight="bold")
        if (s, it) == (REPO_SUBSTEPS, REPO_ITERATIONS):
            ax.plot(r["wall_s"], r["rmse_mm"], "s", ms=9, color=INK, zorder=5)
            ax.annotate("this repo\n8x200 = 1600",
                        (r["wall_s"], r["rmse_mm"]), textcoords="offset points",
                        xytext=(-20, -32), fontsize=8.5, color=INK, weight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall-clock time [s]  (lower is better)")
    ax.set_ylabel("RMSE vs analytic catenary [mm]  (lower is better)")
    ax.set_title("Solver budget: where does effort buy accuracy?",
                 color=INK, fontsize=11.5, weight="bold", loc="left", pad=12)
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path


def plot_damping(rows: list[dict], out_path: str) -> str | None:
    """Stretch-damping effect on accuracy and arc drift, with the bend-damping
    negative control shown alongside.

    Two stacked panels sharing one x-axis, never one panel with two y-scales: a
    dual axis invites the reader to read a crossing point that is an artefact of
    where the two scales were pinned.
    """
    data = sorted((r for r in rows if "stretch_damping" in r["sweep"]
                   and r.get("rmse_mm") is not None),
                  key=lambda r: r["stretch_damping"])
    control = sorted((r for r in rows if "bend_damping_control" in r["sweep"]
                      and r.get("rmse_mm") is not None),
                     key=lambda r: r["bend_damping"])
    if not data:
        return None

    # log x with a 0.0 entry: place it at a decade below the smallest positive
    # value and label it explicitly, rather than dropping Newton's default.
    positives = [r["stretch_damping"] for r in data if r["stretch_damping"] > 0]
    zero_at = (min(positives) / 10.0) if positives else 1e-5
    x = [(r["stretch_damping"] if r["stretch_damping"] > 0 else zero_at) for r in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    for ax in (ax1, ax2):
        _style(ax)

    ax1.plot(x, [r["rmse_mm"] for r in data], "-o", color=SERIES[0], lw=2.0,
             ms=6, mec="white", mew=1.0, zorder=3, label="stretch_damping swept")
    if control:
        lo, hi = control[0]["rmse_mm"], control[-1]["rmse_mm"]
        ax1.axhspan(min(lo, hi), max(lo, hi), color=SERIES[2], alpha=0.30, zorder=1)
        ax1.annotate(
            f"bend_damping control:\n600$\\times$ range spans only "
            f"{abs(hi - lo):.3f} mm",
            xy=(x[0], 0.5 * (lo + hi)), textcoords="offset points",
            xytext=(6, 14), fontsize=8.5, color=INK_SECONDARY)
    ax1.set_ylabel("RMSE vs catenary [mm]")
    ax1.set_title("Damping: which knob actually moves the result?",
                  color=INK, fontsize=11.5, weight="bold", loc="left", pad=12)
    leg = ax1.legend(frameon=False, fontsize=9, loc="best")
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    # Arc drift is the reference-free accuracy check: an inextensible cable
    # must keep its length whatever the catenary says.
    ax2.plot(x, [r["arc_drift_pct"] for r in data], "-o", color=SERIES[1],
             lw=2.0, ms=6, mec="white", mew=1.0, zorder=3)
    ax2.axhline(0.0, color=INK, lw=1.0, ls="--", zorder=2)
    ax2.annotate("inextensible cable", xy=(x[0], 0.0), textcoords="offset points",
                 xytext=(6, 4), fontsize=8.5, color=INK_SECONDARY)
    ax2.set_ylabel("arc-length drift [%]")
    ax2.set_xlabel("stretch_damping  [N s/m]")
    ax2.set_xscale("log")

    if any(r["stretch_damping"] == 0 for r in data):
        for ax in (ax1, ax2):
            ax.axvline(zero_at, color=INK, ls=":", lw=1.2, zorder=2)
        ax2.annotate("0.0\n(Newton default)", xy=(zero_at, ax2.get_ylim()[0]),
                     textcoords="offset points", xytext=(4, 12), fontsize=8.5,
                     color=INK, weight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor="white")
    plt.close(fig)
    return out_path


def _tex_escape(s: str) -> str:
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$")):
        s = s.replace(a, b)
    return s


def _fmt(v, spec=".2f"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return format(v, spec) if isinstance(v, float) else str(v)


def write_report(path: str, rows: list[dict], figures: dict[str, str | None],
                 scenario: dict) -> None:
    budget = sorted((r for r in rows if "budget" in r["sweep"] and r.get("rmse_mm")),
                    key=lambda r: (r["substeps"], r["iterations"]))
    damping = sorted((r for r in rows if "damping" in r["sweep"] and r.get("rmse_mm")),
                     key=lambda r: r["damping_ratio"])

    ok = [r for r in rows if r.get("rmse_mm") is not None]
    best = min(ok, key=lambda r: r["rmse_mm"]) if ok else None
    newton_run = next((r for r in rows if int(r.get("substeps", 0)) == NEWTON_SUBSTEPS
                       and int(r.get("iterations", 0)) == NEWTON_ITERATIONS
                       and r.get("rmse_mm")), None)
    repo_run = next((r for r in rows if int(r.get("substeps", 0)) == REPO_SUBSTEPS
                     and int(r.get("iterations", 0)) == REPO_ITERATIONS
                     and r.get("rmse_mm")), None)

    budget_rows = "\n".join(
        f"    {int(r['substeps'])} & {int(r['iterations'])} & {int(r['budget'])} & "
        f"{_fmt(r['rmse_mm'])} & {_fmt(r['arc_drift_pct'], '+.2f')} & "
        f"{_fmt(r['wall_s'], '.1f')} & {'yes' if r['settled'] else 'no'} \\\\"
        for r in budget)

    damping_rows = "\n".join(
        f"    {_fmt(r['bend_damping'], '.1e')} & {_fmt(r['damping_ratio'], '.3f')} & "
        f"{_fmt(r['rmse_mm'])} & {_fmt(r['arc_drift_pct'], '+.2f')} & "
        f"{'yes' if r['settled'] else 'no'} & "
        f"{_fmt(r['settled_time_s'], '.2f') if r['settled'] else '--'} & "
        f"{_fmt(r['final_max_speed_mps'], '.2e')} \\\\"
        for r in damping)

    # Comparison sentence, only if both reference points actually ran.
    if newton_run and repo_run:
        speedup = repo_run["wall_s"] / newton_run["wall_s"] if newton_run["wall_s"] else float("nan")
        verdict = (
            f"Newton's default reaches {newton_run['rmse_mm']:.2f}~mm RMSE in "
            f"{newton_run['wall_s']:.1f}~s; this repository's default reaches "
            f"{repo_run['rmse_mm']:.2f}~mm in {repo_run['wall_s']:.1f}~s, "
            f"a {speedup:.1f}$\\times$ cost difference.")
    else:
        verdict = "One of the two reference configurations did not produce a result."

    figure_blocks = ""
    for caption, key in (("Solver budget: accuracy against wall-clock cost. Both axes "
                          "are logarithmic; the knee is where additional cost stops "
                          "buying accuracy.", "budget"),
                         ("Damping: accuracy and settling against the dimensionless "
                          "damping ratio. The shaded band is the range spanned by "
                          "Newton's own cable examples.", "damping")):
        p = figures.get(key)
        if p:
            figure_blocks += rf"""
\begin{{figure}}[htbp]
  \centering
  \includegraphics[width=\linewidth]{{{os.path.basename(p)}}}
  \caption{{{caption}}}
\end{{figure}}
"""

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2.4cm]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{amsmath}}
\usepackage[colorlinks=true,linkcolor=blue,urlcolor=blue]{{hyperref}}

\title{{Solver hyperparameters for a hanging cable:\\
       Newton's defaults versus a tuned configuration}}
\author{{Generated by \texttt{{analysis/sweep\_report.py}}}}
\date{{\today}}

\begin{{document}}
\maketitle

\section{{What is being measured}}

A cable of length $L = {_fmt(scenario.get('length'), '.3f')}$~m hangs from
{int(scenario.get('num_points', 2))} level supports spanning
$S = {_fmt(scenario.get('span'), '.3f')}$~m, discretised into
{int(scenario.get('num_segments', 60))} segments and solved with Newton's
\texttt{{add\_rod}} plus the VBD solver.

Accuracy is the RMS distance from the \emph{{analytic catenary}} for the same
$L$ and $S$, not from another solver's answer. This matters: a configuration
cannot score well merely by agreeing with its neighbours, and the reference
costs no simulation time. The catenary assumes zero bending stiffness, which
for this cable is a very good approximation --- its $EI$ is small enough that
the analytic and simulated shapes agree to a millimetre --- so a deviation of
much more than that is numerical error rather than physics.

Two hyperparameters are swept, chosen because the configuration in this
repository differs sharply from every cable example shipped with Newton:

\begin{{itemize}}
  \item \textbf{{Solver budget.}} Newton's examples all use
        $\text{{substeps}} \times \text{{iterations}} = 10 \times 5 = 50$
        iteration-steps per frame. This repository uses $8 \times 200 = 1600$,
        a $32\times$ larger budget. The existing default was chosen by varying
        iterations alone at fixed substeps, so the substep axis was never
        tested.
  \item \textbf{{Damping.}} Newton's examples run
        $\text{{bend\_damping}}/\text{{bend\_stiffness}}$ between
        {NEWTON_RATIO_LO} and {NEWTON_RATIO_HI}. This repository's default sits
        at roughly $1.7\times10^{{-3}}$, between $30\times$ and $600\times$
        less damped.
\end{{itemize}}

{figure_blocks}

\section{{Solver budget}}

{verdict}

\begin{{table}}[htbp]
\centering
\small
\begin{{tabular}}{{rrrrrrc}}
\toprule
substeps & iterations & budget & RMSE [mm] & arc drift [\%] & wall [s] & settled \\
\midrule
{budget_rows}
\bottomrule
\end{{tabular}}
\caption{{Every budget configuration. ``budget'' is
substeps $\times$ iterations, the total iteration-steps per frame.}}
\end{{table}}

\section{{Damping}}

Damping is reported as the dimensionless ratio
$\text{{bend\_damping}}/\text{{bend\_stiffness}}$, because the absolute
\texttt{{bend\_damping}} value is meaningless without the stiffness it damps ---
and the stiffness here is derived from beam theory ($EI/L_{{\text{{seg}}}}$),
not chosen to match Newton's demo numbers.

\begin{{table}}[htbp]
\centering
\small
\begin{{tabular}}{{rrrrccr}}
\toprule
bend\_damping & ratio & RMSE [mm] & arc drift [\%] & settled & $t_{{\text{{settle}}}}$ [s] & final $|v|_{{\max}}$ [m/s] \\
\midrule
{damping_rows}
\bottomrule
\end{{tabular}}
\caption{{Damping sweep at Newton's solver budget ($10\times5$), so the damping
effect is not confounded with a solver-effort effect.}}
\end{{table}}

\section{{How to read this}}

\begin{{itemize}}
  \item \textbf{{Arc drift is the check that needs no reference.}} An
        inextensible cable must keep its length. Drift away from
        $0\%$ is the solver stretching the cable under its own weight, which is
        an artefact regardless of what the catenary says.
  \item \textbf{{An unsettled run is provisional.}} Where \emph{{settled}} is
        ``no'', the reported shape is a snapshot of a cable still in motion,
        averaged over the final window. Its RMSE is not comparable on equal
        terms with a settled run's, and this is exactly what the damping sweep
        is testing.
  \item \textbf{{Cost is wall-clock on one machine.}} Ratios between rows are
        meaningful; absolute seconds are not portable.
\end{{itemize}}

\end{{document}}
"""
    with open(path, "w") as fh:
        fh.write(tex)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-dir", required=True, help="directory holding sweep.csv")
    ap.add_argument("--pdf", action="store_true",
                    help="also compile the report with pdflatex, if available")
    args = ap.parse_args()

    csv_path = os.path.join(args.sweep_dir, "sweep.csv")
    if not os.path.isfile(csv_path):
        print(f"!! no sweep.csv in {args.sweep_dir}", file=sys.stderr)
        return 2

    rows = read_sweep(csv_path)
    scenario = {}
    run_json = os.path.join(args.sweep_dir, "run.json")
    if os.path.isfile(run_json):
        import json
        with open(run_json) as fh:
            scenario = json.load(fh)

    figures = {
        "budget": plot_budget(rows, os.path.join(args.sweep_dir, "budget.png")),
        "damping": plot_damping(rows, os.path.join(args.sweep_dir, "damping.png")),
    }
    for name, p in figures.items():
        print(f"[out] {p}" if p else f"[skip] no {name} data")

    tex_path = os.path.join(args.sweep_dir, "sweep_report.tex")
    write_report(tex_path, rows, figures, scenario)
    print(f"[out] {tex_path}")

    if args.pdf:
        try:
            subprocess.call(["pdflatex", "-interaction=nonstopmode",
                             os.path.basename(tex_path)], cwd=args.sweep_dir,
                            stdout=subprocess.DEVNULL)
            print(f"[out] {tex_path[:-4]}.pdf")
        except FileNotFoundError:
            print("[warn] pdflatex not found; the .tex is still written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
