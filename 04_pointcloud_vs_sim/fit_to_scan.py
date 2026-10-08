"""Find the cable properties that make the simulation match the scan: a parallel search.

    # from the repository root (Ubuntu / Linux: many cores, or an NVIDIA GPU)
    python 04_pointcloud_vs_sim/fit_to_scan.py --config configs/ethernet_cat6.json --workers 16
    python 04_pointcloud_vs_sim/fit_to_scan.py --config configs/ethernet_cat6.json --workers 8 --device cuda:0
    python 04_pointcloud_vs_sim/fit_to_scan.py --config configs/ethernet_cat6.json --params EI   # stiffness only

What is searched (everything else comes from the JSON and stays fixed):

    EI      bending rigidity [N m^2], searched in log scale. GJ keeps the JSON's GJ/EI ratio.
    curl    the cable's natural curl: rest curvature kappa [1/m] (coil radius 1/kappa)
            and the side it curls to, phi [deg]. kappa = 0 is a straight cable.

The score of a candidate is the RMS distance [mm] from the scanned centreline to the
settled simulated cable (`--part all`, default) or only its hanging part
(`--part hanging`, the part that does not depend on friction or history).

Mass and EI cannot be told apart from a still cable: only EI / weight sets the
shape. So weigh the cable and put mass_per_length_kg_m in the JSON first; the
fitted EI is only right for that mass.

Search:
  1. coarse grid, all candidates in parallel (EI x curl x direction);
  2. pattern search from the best: try one step up and down in every parameter,
     in parallel; move to the best, or halve the steps; stop when the steps are small.
Every run is kept in <output_dir>_fit/runs/ and every score in evaluations.csv. A
second call with the same --out continues where the first stopped (finished runs
are reused).

Outputs (<output_dir>_fit/):
    evaluations.csv     every candidate tried, its scores, settled or not
    best.json           the best parameters and scores
    best_config.json    the JSON with the best values filled in: run it with the viewer
    progress.png        score vs EI for every candidate
    best/compare/       overlay and error plots of the best run vs the scan

This is a FIT: the scan is used to choose the values. To know how good the model
really is, run best_config.json on a second scan in a different pose without
refitting (only change scan.centerline_csv), and look at that error.
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin" and os.environ.pop("DYLD_LIBRARY_PATH", None):
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import contextlib
import csv
import importlib.util
import json
import multiprocessing as mp
import shutil
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "03_franka_holds_cable" / "ethernet_scene.py"
COMPARE = REPO / "04_pointcloud_vs_sim" / "compare_to_scan.py"
G = 9.81

# Reference palette (same as compare_to_scan.py).
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e1e0d9", "#fcfcfb"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Parameters
# =============================================================================
class Space:
    """The searched parameters, as a vector x: [log10 EI, (kappa, phi)]."""

    def __init__(self, names, ei0, ei_span, kappa_max):
        self.names = names
        self.curl = "curl" in names
        self.lo = [np.log10(ei0 / ei_span)] + ([0.0, -np.inf] if self.curl else [])
        self.hi = [np.log10(ei0 * ei_span)] + ([kappa_max, np.inf] if self.curl else [])

    def clip(self, x):
        x = np.array(x, float)
        x = np.clip(x, self.lo, self.hi)
        if self.curl:
            x[2] = x[2] % 360.0
            if x[1] <= 1e-9:  # no curl: the direction means nothing
                x[1], x[2] = 0.0, 0.0
        return x

    def key(self, x):
        x = self.clip(x)
        return tuple(round(float(v), d) for v, d in zip(x, (4, 3, 1)))

    def describe(self, x):
        x = self.clip(x)
        d = {"EI_Nm2": float(10 ** x[0])}
        if self.curl:
            d["rest_curvature_per_m"] = float(x[1])
            d["rest_curl_direction_deg"] = float(x[2])
        return d

    def label(self, x):
        d = self.describe(x)
        s = f"EI{d['EI_Nm2']:.4g}"
        if self.curl:
            s += f"_k{d['rest_curvature_per_m']:.3g}_p{d['rest_curl_direction_deg']:.0f}"
        return s.replace("+", "")


# =============================================================================
# One evaluation (runs in a worker process)
# =============================================================================
_W = {}


def _init_worker(config_path, device, runs_dir):
    """Load the scene and comparison code and the scan once per worker."""
    import warp as wp
    wp.config.quiet = True
    _W["es"] = es = load_module("ethernet_scene", SCENE)
    _W["cs"] = load_module("compare_to_scan", COMPARE)
    cfg = es.load_config(config_path)
    cfg["robot"]["enabled"] = False  # visual only; skipping it saves the IK search in every run
    if device:
        cfg["sim"]["device"] = device
    _W["cfg"] = cfg
    _W["runs_dir"] = Path(runs_dir)
    with open(os.devnull, "w") as quiet, contextlib.redirect_stdout(quiet):
        _W["scan"] = es.load_scan(cfg)
    _W["ratio_gj"] = cfg["cable"]["twist_rigidity_GJ_Nm2"] / cfg["cable"]["bend_rigidity_EI_Nm2"]


def evaluate(params, label):
    """Simulate one candidate and score it against the scan."""
    es, cs = _W["es"], _W["cs"]
    cfg = json.loads(json.dumps(_W["cfg"]))
    cab = cfg["cable"]
    cab["bend_rigidity_EI_Nm2"] = params["EI_Nm2"]
    cab["twist_rigidity_GJ_Nm2"] = params["EI_Nm2"] * _W["ratio_gj"]
    cab["rest_curvature_per_m"] = params.get("rest_curvature_per_m", cab.get("rest_curvature_per_m", 0.0))
    cab["rest_curl_direction_deg"] = params.get("rest_curl_direction_deg", cab.get("rest_curl_direction_deg", 0.0))
    out = _W["runs_dir"] / label
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = dict(params)
    try:
        with open(out / "log.txt", "w") as log, contextlib.redirect_stdout(log):
            es.run(cfg, _W["scan"], 1.0, out)
        meta, sim_s, sim_X, init_X, _ = cs.load_run(out)
        scan_X, scan_s, scan_sup = cs.load_scan(meta["scan"], meta["gripper_end"], meta["scan_to_world"])
        row, _ = cs.compare_one(scan_X, scan_s, scan_sup, meta, sim_s, sim_X, init_X)
        result.update({k: row[k] for k in ("shape_rms_mm", "shape_hanging_rms_mm", "shape_lying_rms_mm",
                                           "shape_max_mm", "arc_rms_mm", "moved_max_mm", "settled")})
    except BaseException as e:  # a blown-up run is a bad candidate, not a crash of the search
        result.update({"shape_rms_mm": float("inf"), "error": f"{type(e).__name__}: {e}"[:200]})
    result["wall_s"] = round(time.time() - t0, 1)
    result["run"] = label
    return result


# =============================================================================
# The search (main process)
# =============================================================================
FIELDS = ["run", "EI_Nm2", "rest_curvature_per_m", "rest_curl_direction_deg", "score_mm", "shape_rms_mm",
          "shape_hanging_rms_mm", "shape_lying_rms_mm", "shape_max_mm", "arc_rms_mm", "moved_max_mm", "settled",
          "wall_s", "stage", "error"]


class Search:
    def __init__(self, space, part, out_dir, pool):
        self.space, self.part, self.out, self.pool = space, part, out_dir, pool
        self.done = {}  # key -> result
        self.csv_path = out_dir / "evaluations.csv"
        if self.csv_path.exists():  # resume
            with open(self.csv_path) as f:
                for r in csv.DictReader(f):
                    x = [np.log10(float(r["EI_Nm2"]))]
                    if space.curl:
                        x += [float(r["rest_curvature_per_m"] or 0.0), float(r["rest_curl_direction_deg"] or 0.0)]
                    r = {k: (float(v) if k not in ("run", "stage", "error", "settled") and v not in ("", None) else v)
                         for k, v in r.items()}
                    r["settled"] = r.get("settled") in ("True", "1", True)
                    self.done[space.key(x)] = r
            print(f"resuming: {len(self.done)} runs already in {self.csv_path}")

    def score(self, r):
        v = r.get("shape_hanging_rms_mm") if self.part == "hanging" else r.get("shape_rms_mm")
        if v is None or v == "" or not np.isfinite(float(v)):
            v = r.get("shape_rms_mm", float("inf"))
        v = float(v) if v not in ("", None) else float("inf")
        return v if r.get("settled", False) in (True, "True") else v + 1.0  # unsettled: penalised a little

    def run_batch(self, xs, stage):
        """Evaluate candidates in parallel (skipping ones already done); return (x, score) for all."""
        todo, seen = [], set()
        for x in xs:
            k = self.space.key(x)
            if k not in self.done and k not in seen:
                seen.add(k)
                todo.append(self.space.clip(x))
        if todo:
            print(f"[{stage}] {len(todo)} runs", flush=True)
            futures = {self.pool.submit(evaluate, self.space.describe(x), self.space.label(x)): x for x in todo}
            pending = set(futures)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    x = futures[fut]
                    r = fut.result()
                    r["stage"] = stage
                    r["score_mm"] = self.score(r)
                    self.done[self.space.key(x)] = r
                    self.append_csv(r)
                    d = self.space.describe(x)
                    curl = (f"  curl {d['rest_curvature_per_m']:5.2f}/m @ {d['rest_curl_direction_deg']:5.1f} deg"
                            if self.space.curl else "")
                    print(f"   EI {d['EI_Nm2']:.3e}{curl}  ->  {r['score_mm']:7.2f} mm"
                          f"{'' if r.get('settled') else '  (not settled)'}{'  ' + r['error'] if r.get('error') else ''}"
                          f"   [{r['wall_s']:.0f} s]", flush=True)
        return [(self.space.clip(x), self.done[self.space.key(x)]["score_mm"]) for x in xs]

    def append_csv(self, r):
        new = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: r.get(k, "") for k in FIELDS})

    def best(self):
        k = min(self.done, key=lambda k: float(self.done[k]["score_mm"]))
        return np.array(k, float), self.done[k]


def coarse_grid(space, ei0, n_ei, ei_grid_span, kappas, n_dir):
    eis = np.geomspace(ei0 / ei_grid_span, ei0 * ei_grid_span, n_ei)
    xs = []
    for ei in eis:
        if not space.curl:
            xs.append([np.log10(ei)])
            continue
        for k in kappas:
            dirs = [0.0] if k <= 0 else np.arange(n_dir) * 360.0 / n_dir
            xs += [[np.log10(ei), k, p] for p in dirs]
    return xs


def pattern_search(search, x0, steps, min_steps, max_rounds):
    x, fx = x0.copy(), search.done[search.space.key(x0)]["score_mm"]
    steps = np.array(steps, float)
    for rnd in range(1, max_rounds + 1):
        if np.all(steps <= min_steps):
            break
        cands = []
        for i in range(len(x)):
            if steps[i] <= min_steps[i]:
                continue
            for sign in (1.0, -1.0):
                y = x.copy()
                y[i] += sign * steps[i]
                cands.append(y)
        if search.space.curl and x[1] > 0:  # also the diagonal of strength and direction
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    y = x.copy()
                    y[1] += s1 * steps[1]
                    y[2] += s2 * steps[2]
                    cands.append(y)
        results = search.run_batch(cands, f"refine {rnd}")
        y, fy = min(results, key=lambda t: t[1])
        if fy < fx - 0.05:  # better by more than the noise of one run
            x, fx = y, fy
        else:
            steps = steps / 2.0
        d = search.space.describe(x)
        print(f"   round {rnd}: best {fx:.2f} mm at EI {d['EI_Nm2']:.4g}"
              + (f", curl {d['rest_curvature_per_m']:.2f}/m @ {d['rest_curl_direction_deg']:.0f} deg"
                 if search.space.curl else "") + f"; steps {np.round(steps, 3).tolist()}", flush=True)
    return x, fx


# =============================================================================
# Outputs
# =============================================================================
def write_best_config(config_path, best_params, out_path, output_dir):
    """The original JSON (comments kept) with the fitted values filled in."""
    raw = json.loads(Path(config_path).read_text())
    cab = raw["cable"]
    ratio = cab["twist_rigidity_GJ_Nm2"] / cab["bend_rigidity_EI_Nm2"]
    cab["bend_rigidity_EI_Nm2"] = float(f"{best_params['EI_Nm2']:.5g}")
    cab["twist_rigidity_GJ_Nm2"] = float(f"{best_params['EI_Nm2'] * ratio:.5g}")
    if "rest_curvature_per_m" in best_params:
        cab["rest_curvature_per_m"] = round(best_params["rest_curvature_per_m"], 4)
        cab["rest_curl_direction_deg"] = round(best_params["rest_curl_direction_deg"], 2)
    raw["output_dir"] = str(output_dir)
    raw["_fitted"] = f"values filled in by 04_pointcloud_vs_sim/fit_to_scan.py from {config_path}"
    out_path.write_text(json.dumps(raw, indent=2) + "\n")


def plot_progress(path, rows, best_run, curl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = [r for r in rows if np.isfinite(float(r["score_mm"]))]
    if not ok:
        return
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ei = np.array([float(r["EI_Nm2"]) for r in ok])
    sc = np.array([float(r["score_mm"]) for r in ok])
    if curl:
        k = np.array([float(r.get("rest_curvature_per_m") or 0.0) for r in ok])
        edges = np.unique(np.quantile(k, [0, 0.25, 0.5, 0.75, 1.0])) if np.ptp(k) > 0 else np.array([k[0], k[0]])
        bins = np.clip(np.searchsorted(edges, k, side="right") - 1, 0, len(BLUE_RAMP) - 1)
        for b in sorted(set(bins.tolist())):
            m = bins == b
            lo, hi = k[m].min(), k[m].max()
            ax.scatter(ei[m], sc[m], s=40, color=BLUE_RAMP[min(b + 1, len(BLUE_RAMP) - 1)], edgecolor=SURFACE,
                       linewidth=1.0, label=f"curl {lo:.1f}–{hi:.1f} /m" if hi > lo else f"curl {lo:.1f} /m", zorder=3)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    else:
        order = np.argsort(ei)
        ax.plot(ei[order], sc[order], color=BLUE_RAMP[2], linewidth=2, marker="o", markersize=8, zorder=3)
    b = next(r for r in ok if r["run"] == best_run)
    ax.scatter([float(b["EI_Nm2"])], [float(b["score_mm"])], s=140, facecolor="none", edgecolor=INK, linewidth=2, zorder=4)
    ax.annotate(f"best: {float(b['score_mm']):.1f} mm", (float(b["EI_Nm2"]), float(b["score_mm"])),
                textcoords="offset points", xytext=(10, 8), color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(0, min(np.percentile(sc, 95) * 1.2, sc.max() * 1.05))
    ax.set_xlabel("bending rigidity EI [N m²]", color=INK_2)
    ax.set_ylabel("RMS distance to the scan [mm]", color=INK_2)
    ax.set_title(f"Fit: every candidate ({len(ok)} runs)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--params", default="EI,curl", choices=["EI", "EI,curl"], help="what to fit")
    ap.add_argument("--part", default="all", choices=["all", "hanging"], help="which part of the cable scores")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                    help="simulations at the same time (CPU cores, or a few per GPU)")
    ap.add_argument("--device", default=None, help="Warp device for the runs, e.g. cpu or cuda:0 (default: JSON)")
    ap.add_argument("--out", default=None, help="output folder (default <output_dir>_fit)")
    ap.add_argument("--grid-ei", type=int, default=7, help="EI values in the coarse grid")
    ap.add_argument("--ei-span", type=float, default=8.0, help="coarse grid covers EI/span .. EI*span")
    ap.add_argument("--grid-curl", default="0,3,6,9", help="curl strengths [1/m] in the coarse grid")
    ap.add_argument("--grid-dir", type=int, default=6, help="curl directions in the coarse grid")
    ap.add_argument("--max-rounds", type=int, default=20, help="pattern-search rounds at most")
    args = ap.parse_args()

    es = load_module("ethernet_scene", SCENE)
    cfg = es.load_config(args.config)
    cab = cfg["cable"]
    ei0 = float(cab["bend_rigidity_EI_Nm2"])
    names = args.params.split(",")
    space = Space(names, ei0, ei_span=30.0, kappa_max=25.0)
    out = Path(args.out) if args.out else Path(str(es.repo_path(cfg["output_dir"])) + "_fit")
    (out / "runs").mkdir(parents=True, exist_ok=True)
    kappas = [float(k) for k in args.grid_curl.split(",")]

    w = cab["mass_per_length_kg_m"] * G
    print(f"fitting {', '.join(names)} to {cfg['scan']['centerline_csv']} ({args.part} of the cable), "
          f"{args.workers} workers, device {args.device or cfg['sim'].get('device') or 'default'}")
    print(f"mass per metre {1000 * cab['mass_per_length_kg_m']:.1f} g (fixed: weigh it), start EI {ei0:.3g} N m^2")

    t0 = time.time()
    ctx = mp.get_context("spawn")  # never fork a process that may hold CUDA / Warp state
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(str(Path(args.config).resolve()), args.device, str(out / "runs"))) as pool:
        search = Search(space, args.part, out, pool)
        search.run_batch(coarse_grid(space, ei0, args.grid_ei, args.ei_span, kappas, args.grid_dir), "coarse grid")
        x0, _ = search.best()
        # Steps: a third of a decade in EI first; curl 1.5 /m and 30 deg.
        steps = [0.15] + ([1.5, 30.0] if space.curl else [])
        min_steps = [0.01] + ([0.1, 3.0] if space.curl else [])
        pattern_search(search, x0, steps, np.array(min_steps), args.max_rounds)
        xb, rb = search.best()

    best = space.describe(xb)
    best.update({k: rb.get(k) for k in ("score_mm", "shape_rms_mm", "shape_hanging_rms_mm", "shape_lying_rms_mm",
                                        "shape_max_mm", "moved_max_mm", "settled", "run")})
    ell = (best["EI_Nm2"] / w) ** (1.0 / 3.0)
    best.update({"part_scored": args.part, "mass_per_length_kg_m": cab["mass_per_length_kg_m"],
                 "EI_over_weight_m3": best["EI_Nm2"] / w, "gravity_bending_length_m": ell,
                 "runs": len(search.done), "wall_min": round((time.time() - t0) / 60, 1)})
    (out / "best.json").write_text(json.dumps(best, indent=2, default=float) + "\n")
    write_best_config(args.config, best, out / "best_config.json", out / "best_run")
    plot_progress(out / "progress.png", list(search.done.values()), rb["run"], space.curl)

    # Overlay and error plots of the best run, with the usual comparison script.
    best_dir = out / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(out / "runs" / rb["run"], best_dir / rb["run"])
    subprocess.run([sys.executable, str(COMPARE), "--config", args.config, "--runs", str(best_dir)],
                   stdout=subprocess.DEVNULL, check=False)

    print("\n=== best ===")
    print(f"EI    {best['EI_Nm2']:.4g} N m^2   (EI / weight = {best['EI_over_weight_m3']:.3g} m^3, "
          f"gravity-bending length {1000 * ell:.0f} mm)")
    if space.curl:
        k = best["rest_curvature_per_m"]
        print(f"curl  {k:.2f} 1/m" + (f" (coil radius {1000 / k:.0f} mm) towards {best['rest_curl_direction_deg']:.0f} deg"
                                      if k > 0 else " (straight cable fits best)"))
    print(f"RMS distance to the scan: {best['shape_rms_mm']:.2f} mm (hanging {best['shape_hanging_rms_mm']:.2f}, "
          f"lying {best['shape_lying_rms_mm']:.2f}), max {best['shape_max_mm']:.1f} mm, "
          f"{'settled' if best['settled'] else 'NOT settled'}")
    print(f"{best['runs']} runs in {best['wall_min']} min. Results: {out}/")
    print(f"watch it:  python 03_franka_holds_cable/ethernet_scene.py --config {out / 'best_config.json'} --viewer gl")
    print("Differences below ~2-3 mm are within the pipeline's own noise. To test the fit, run best_config.json on a "
          "second scan (another pose) without refitting.")


if __name__ == "__main__":
    main()
