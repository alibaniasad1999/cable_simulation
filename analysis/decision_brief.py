#!/usr/bin/env python3
"""
A one-page DECISION BRIEF for choosing a cable-simulation approach.

    python3 analysis/decision_brief.py --run results/<run> --pdf

Unlike make_thesis_report.py (the full technical write-up), this is meant for a
busy reader who has to DECIDE which method to adopt for a larger project. It is
deliberately short: a plain-language comparison table whose decisive column is
INTEGRATION -- what each approach is compatible with -- one figure, and a
recommendation. Accuracy numbers are pulled from the run so they stay honest;
the integration notes are editorial and live here.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess


# Plain-language, project-facing description of each method. The 'fit' line is
# the one a decision hinges on: what wider ecosystem the approach plugs into.
PROFILE = {
    "physx_capsule": {
        "name": "PhysX capsule chain",
        "engine": "PhysX 5 (native to Isaac Sim / Omniverse)",
        "fit": "Drops straight into an Isaac Sim / Omniverse project: USD-native, "
               "works with Isaac Lab, the ROS 2 bridge, sensors, and collision "
               "against robots and the environment. Real-time.",
        "best": "the cable must live in, and interact with, an Isaac Sim scene.",
        "watch": "rigid-capsule approximation; not differentiable.",
    },
    "newton_cable": {
        "name": "Newton cable (add_rod)",
        "engine": "Newton — NVIDIA's new GPU, differentiable physics engine",
        "fit": "Native to the Newton / Warp stack: differentiable (gradients for "
               "control, learning, system identification) and GPU-batched for "
               "large-scale training. Newton is the forward-looking successor in "
               "the Isaac roadmap.",
        "best": "high-fidelity cable physics, differentiability, or GPU-batched "
                "learning are needed.",
        "watch": "newer, evolving API; heavier per-step cost.",
    },
    "warp_rod": {
        "name": "Warp XPBD rod",
        "engine": "Pure Warp — no physics engine",
        "fit": "A single self-contained Warp component with no engine dependency: "
               "differentiable, portable, easy to embed in a custom pipeline. You "
               "own the modelling (stiffness, resolution, boundary conditions).",
        "best": "a lightweight or differentiable rod is wanted without a full "
                "engine.",
        "watch": "bending stiffness and mesh resolution must be calibrated; "
                 "crudest bending model.",
    },
}
ORDER = ["physx_capsule", "newton_cable", "warp_rod"]


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("&", r"\&"),
                 ("%", r"\%"), ("#", r"\#"), ("$", r"\$"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    return s


def fnum(v, nd=0):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "--"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pdf", action="store_true")
    args = ap.parse_args()
    out_dir = args.out or os.path.join(args.run, "brief")
    os.makedirs(out_dir, exist_ok=True)

    metrics = {}
    with open(os.path.join(args.run, "metrics.csv")) as fh:
        for r in csv.DictReader(fh):
            metrics[r["method"]] = r

    overlay = None
    src = os.path.join(args.run, "overlay.png")
    if os.path.isfile(src):
        overlay = "overlay.png"
        shutil.copyfile(src, os.path.join(out_dir, overlay))

    # A simple accuracy bar chart (error against the real cable), the number that
    # matters for choosing a model of a real object.
    bar = _accuracy_bar(metrics, out_dir)

    rows = ""
    for m in ORDER:
        if m not in metrics or m not in PROFILE:
            continue
        p = PROFILE[m]
        r = metrics[m]
        rows += (
            f"\\textbf{{{esc(p['name'])}}} & "
            f"{fnum(r.get('rmse_vs_real_mm'))}~mm & "
            f"{fnum(r.get('wall_s'),0)}~s & "
            f"{esc(p['fit'])} \\\\[2pt]\n"
            f"\\multicolumn{{4}}{{p{{\\dimexpr\\linewidth-2\\tabcolsep}}}}{{"
            f"\\footnotesize\\itshape Best when {esc(p['best'])} "
            f"Watch: {esc(p['watch'])}}} \\\\\n\\midrule\n")

    fig_overlay = ("" if not overlay else
                   f"\\begin{{figure}}[h]\\centering\n"
                   f"\\includegraphics[width=0.86\\linewidth]{{{overlay}}}\n"
                   f"\\caption{{Every approach, the exact mathematics (dashed), and "
                   f"the real cable (dotted) on one axis. The methods bracket the "
                   f"real cable; the differences are boundary-condition modelling, "
                   f"not solver bugs.}}\\end{{figure}}")
    fig_bar = ("" if not bar else
               f"\\begin{{figure}}[h]\\centering\n"
               f"\\includegraphics[width=0.66\\linewidth]{{{bar}}}\n"
               f"\\caption{{Error against the photographed cable "
               f"(lower is closer to reality).}}\\end{{figure}}")

    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=2cm]{{geometry}}
\usepackage{{booktabs,graphicx,array,xcolor}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{5pt}}
\begin{{document}}

\begin{{center}}
{{\Large\bfseries Cable simulation: which approach to adopt}}\\[2pt]
{{\normalsize A decision brief --- compatibility first}}
\end{{center}}

\textbf{{The question.}} A cable must be simulated as part of a larger system.
Several methods reproduce its hanging shape to within a few centimetres of a real
cable, so raw accuracy is \emph{{not}} the deciding factor. What decides it is
\textbf{{what each approach is compatible with}} --- which existing stack it plugs
into with the least friction --- because the cable is one component of a much
bigger project.

\textbf{{One thing to know about accuracy.}} The method closest to the textbook
mathematics is the \emph{{furthest}} from the real cable, and vice-versa. This is
not a solver being wrong: the real cable is held with tape that fixes its angle,
which the ideal curve does not model. Every method here is behaving correctly;
they differ in \emph{{which}} real-world detail they capture. So choose on
integration, and treat all of them as accurate enough to a few centimetres.

\vspace{{4pt}}
\renewcommand{{\arraystretch}}{{1.25}}
\begin{{tabular}}{{@{{}}p{{3.1cm}} c c p{{7.9cm}}@{{}}}}
\toprule
\textbf{{Approach}} & \textbf{{vs real}} & \textbf{{speed}} & \textbf{{Integrates with}} \\
\midrule
{rows}\bottomrule
\end{{tabular}}

{fig_bar}

{fig_overlay}

\textbf{{Recommendation.}}
\begin{{itemize}}
\setlength{{\itemsep}}{{1pt}}
\item If the wider project lives in \textbf{{Isaac Sim / Omniverse}} and the cable
      must interact with robots or the scene: \textbf{{PhysX capsule chain}}.
      Most compatible with what is already there, closest to the real cable, and
      real-time.
\item If the project needs \textbf{{differentiability or GPU-batched learning}},
      or is adopting \textbf{{Newton}}: \textbf{{Newton cable}}. Highest fidelity
      to the physics and future-facing, at a higher compute cost.
\item If a \textbf{{lightweight, embeddable, engine-free}} component is wanted:
      \textbf{{Warp rod}}. Fewest dependencies and differentiable, but you
      calibrate its stiffness and resolution.
\end{{itemize}}

\textbf{{Ruled out on this build.}} A Newton \emph{{USD-engine}} cable cannot be
fixed at both ends (an articulation must be a tree, a doubly-clamped cable is a
loop), and PhysX \emph{{volumetric FEM}} builds but its state cannot be read back.
Both fail with a clear diagnosis rather than silently. The three above are the
live options.

\end{{document}}
"""
    tex_path = os.path.join(out_dir, "decision_brief.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)
    print(f"[out] {tex_path}")

    if args.pdf:
        for _ in range(2):
            subprocess.call(["pdflatex", "-interaction=nonstopmode", "decision_brief.tex"],
                            cwd=out_dir, stdout=subprocess.DEVNULL)
        pdf = os.path.join(out_dir, "decision_brief.pdf")
        if os.path.isfile(pdf):
            print(f"[out] {pdf}")
    return 0


def _accuracy_bar(metrics, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    names, vals, colors = [], [], []
    palette = {"physx_capsule": "#D55E00", "newton_cable": "#0072B2",
               "warp_rod": "#009E73"}
    for m in ORDER:
        r = metrics.get(m)
        if not r or not r.get("rmse_vs_real_mm"):
            continue
        names.append(PROFILE[m]["name"].replace(" ", "\n", 1))
        vals.append(float(r["rmse_vs_real_mm"]))
        colors.append(palette.get(m, "0.5"))
    if not vals:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(range(len(vals)), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f} mm", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("error vs real cable [mm]")
    ax.set_ylim(0, max(vals) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = "accuracy.png"
    fig.savefig(os.path.join(out_dir, p), dpi=160)
    plt.close(fig)
    return p


if __name__ == "__main__":
    main()
