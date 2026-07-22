# Cable-hang study — mathematics, a real cable, and four solvers

Predict the shape of a cable hanging from 2 or 3 equal-height supports, and
compare, on one axis, three independent answers:

- **MATH** — the governing differential equation, integrated numerically;
- **REAL** — the cable in a photograph, measured by computer vision;
- **SIMULATION** — several physics solvers (Newton, PhysX, Warp).

The three-way structure is the whole point: any two can agree for uninteresting
reasons, and only the third lets a disagreement be blamed on *numerical* error
(vs MATH) as opposed to *modelling* error (vs REAL).

---

## Do it all with one command

Everything about a cable — its photo, support positions, material, solver
settings — lives in **one config file**. To run the complete study for a cable,
point the driver at its file:

```bash
~/isaacsim/python.sh run_experiment.py --config experiments/apple_cable_3pt.yaml
```

That single command:

1. **measures** the cable's length and middle-split from the photograph,
2. **integrates** the governing ODE and checks it against the closed form,
3. **runs** every requested solver on the identical scenario,
4. **compares** all of them against the math and the photo, writing
   `metrics.csv`, `overlay.png`, `errors.png`, and a `report.tex`.

Then build the full write-up (adds the hyperparameter sweep, the Warp
bending-stiffness study, validity, and conclusions):

```bash
# sensitivity figure the report embeds
~/isaacsim/python.sh analysis/warp_bend_sensitivity.py \
    --scenario results/<run>/run.json --out results/<run>/warp_bend.png

# assemble the PDF
python3 analysis/make_thesis_report.py --run results/<run> \
    --cross-check results/<the 2-point run> \
    --sweep results/<a sweep dir> --pdf
```

For a reader who has to **decide** rather than review, there is a separate
two-page brief — comparison table, the real-cable measurement, one overlay:

```bash
~/isaacsim/python.sh analysis/decision_brief.py --run results/<run> --pdf
```

**A new cable needs no code — just a new config file.** Copy
`experiments/apple_cable_3pt.yaml`, change the photo path and support positions,
and run it.

---

## The config file (the only thing you edit)

```yaml
name: apple_cable_3pt
supports:
  x_m: [0.0, 0.15, 0.35]   # support positions from the left clamp [m]
  height_m: 1.0            # all supports share a height
cable:
  radius_m: 0.002          # material stand-ins; the shape barely depends on them
  youngs_modulus_pa: 80.0e6
  length_m: null           # null -> MEASURED from the photo (recommended)
  mid_fraction: null       # null -> MEASURED from the photo
image:
  path: images/three_point.jpg
  sat_max: 70              # HSV thresholds separating cable from background
  val_min: 150
simulation:
  methods: [newton_cable, newton_cable_tape, physx_capsule, warp_rod]
  segments: 150            # discretisation; 150 renders smoothly
  substeps: 4              # Newton solver budget (from the sweep)
  iterations: 200
  mid_support: clamp       # middle-support model: clamp | pin | (tape via the _tape method)
  tape_halfwidth_m: 0.012  # flat-tape plateau half-width [m]
  warp_bend_length_m: 0.02 # Warp bending length l=(EI/w)^(1/3) [m]; null -> material EI
```

A field left `null` is **measured from the photograph** — deliberately, so the
length the simulations are scored against comes from the same image, not from a
tape measure or the packaging.

---

## The problem, and why it is exactly solvable

A cable of length `L` hangs from `N ∈ {2,3}` supports, **all at height `H`**,
across a span `S`:

```
   ●─────╮             ╭────●        all at z = H
          ╰───────────╯
```

Equal heights make it exactly solvable. A perfectly flexible, inextensible cable
is a **catenary**, `z(x) = a·cosh((x−x₀)/a) + c`, with `a = H/w` fixed by the
arc-length constraint. Three equal-height supports give **two independent
sub-catenaries**. So there is a closed form to measure against, with no fitted
parameters — and the code integrates the governing ODE `z'' = √(1+z'²)/a` from
scratch and confirms it reproduces the closed form to `7×10⁻³ µm`.

---

## Methods

| key | model | engine | status |
|---|---|---|---|
| `newton_cable` | Newton Cosserat rod (`add_rod` + VBD). Middle support a **point pin** → kinks. | Newton (Warp) | ✅ most accurate vs math |
| `newton_cable_tape` | Same solver, middle support a **flat tape** → rounds the top like the real cable. | Newton (Warp) | ✅ best real-match among Newton |
| `physx_capsule` | PhysX rigid capsule chain, D6 joints + bending springs. | Isaac Sim (PhysX) | ✅ best real-match overall |
| `warp_rod` | XPBD elastic rod in pure Warp. True position-only pin. | Warp only | ✅ needs a bending length |
| `newton_engine` | Newton via USD articulation. | Isaac Sim | ❌ USD can't hold both ends |
| `physx_fem` | PhysX volumetric deformable. | Isaac Sim (PhysX) | ⚠️ no state readback on this build |

The middle support **dominates** the 3-support result and is a modelling choice,
not solver quality:

- **point pin** — two catenaries meet at opposite slopes → an unavoidable
  upward **kink**. Nearest the math.
- **tape** (`newton_cable_tape`) — a short flat plateau, so the cable leaves
  horizontally and bending **rounds** the top, as a strip of tape does. Nearest
  the photo among Newton runs.
- **true pin** (Warp) — position held, both tangents free; bending rounds it
  smoothly. Newton's `add_rod` cannot express this (it holds bodies, hence
  orientation), which is why the tape model exists.

---

## Other ways to run

```bash
# direct CLI, no config file (older driver) — 2 supports, all methods
python3 run_benchmark.py --points 2

# one method standalone, with a GUI viewport
~/isaacsim/python.sh methods/hang_newton_cable.py \
    --length 0.9 --span 0.35 --points 3 --mid-support tape --gui

# hyperparameter sweep (substeps x iterations, damping) + its report
~/isaacsim/python.sh run_sweep.py --out results/sweep
python3 analysis/sweep_report.py --sweep results/sweep

# measure a cable from a photo only (no simulation)
~/isaacsim/python.sh image_utils/measure_length.py \
    --image images/three_point.jpg --supports 0 0.15 0.35 \
    --figure detection.png --profile-csv real.csv
```

---

## Outputs

```
results/<run>/
  config.resolved.yaml   the experiment as run, measurements filled in
  provenance.json        which inputs were measured vs assumed
  run.json               the shared scenario handed to every solver
  real_profile.csv       the cable extracted from the photo
  detection.png          CHECK THIS — segmentation + fitted profile
  metrics.csv            THE comparison table, one row per method
  overlay.png            every method + catenary + photo on one axis
  errors.png             height error vs x
  warp_bend.png          Warp bending-stiffness sensitivity
  report.tex / report.pdf
  <method>/profile.csv, meta.json, ...
```

`metrics.csv` key columns: `rmse_mm` (vs the analytic catenary),
`rmse_vs_real_mm` (vs the photo), `arc_drift_pct` (reference-free: an
inextensible cable must keep its length), `settled`, `wall_s`.

---

## Current results (apple charger cable, 3 supports, 150 segments)

`L ≈ 0.893 m` measured from the photo, span `0.35 m`, `L/S = 2.55` (very slack).

| method | vs MATH (catenary) | vs REAL (photo) |
|---|---|---|
| `newton_cable` (point pin) | **0.9 mm** | 75 mm |
| `newton_cable_tape` | 21 mm | **65 mm** |
| `physx_capsule` | 74 mm | **33 mm** |
| `warp_rod` (drape-consistent EI) | 45 mm | 51 mm |

**The central result:** the solver truest to the mathematics (Newton point pin)
is *furthest* from the real cable, and PhysX is the reverse. That inversion is a
**boundary-condition** effect — the taped clamps fix the cable's angle, which the
ideal catenary does not — not a solver ranking. Imposing the real (flat-tape)
condition moves Newton from 75 → 65 mm toward the photo, confirming it.

Two independent photos of the same cable measure its length to **0.32 mm**
(0.036 %) of each other — the vision pipeline is trustworthy.

---

## Two solver sensitivities worth knowing

**Newton budget.** Accuracy is a function of `substeps × iterations`; cost is
not (each substep has fixed overhead). Fewer substeps with more iterations is
cheaper at equal accuracy. Damping the stretch constraint is *harmful*; damping
bending does nothing. Defaults `4 × 200` come from `run_sweep.py`.

**Warp bending stiffness.** Warp is the only method with a true position-only
pin, so its shape depends on the bending stiffness — the one input not measured
here. The material stand-in (`ℓ = (EI/w)^{1/3} ≈ 150 mm`) is too stiff: at a
coarse mesh the slack rod even **buckles** into an arch (a discretisation
artefact the finer mesh removes; see `analysis/warp_bend_sensitivity.py`). The
cable's visible ~2 cm rounding scale implies `ℓ ≈ 20 mm`, where it drapes
correctly. This is read from the rounding, **not** fitted to the catenary.

---

## Known limitations on this build (Isaac Sim 6.0)

- **`newton_engine` cannot represent a both-ends-fixed cable.** A Newton USD
  articulation must be a tree; a cable clamped at both ends is a closed loop.
  Excluding the second anchor makes the engine silently ignore it. Use
  `newton_cable`, which drives the same solver through `add_rod`.
- **`physx_fem` builds and simulates but its state cannot be read back.** The
  deformable tensor view is created with a `None` backend, so nodal positions
  cannot be exported on this build. Reading the TetMesh from USD/Fabric is the
  likely workaround; not yet implemented.

Both methods fail fast with this diagnosis rather than producing empty results.

---

## Layout

```
run_experiment.py          config-driven driver (measure → math → sim → compare)
run_benchmark.py           direct-CLI driver (no config file)
run_sweep.py               hyperparameter sweep
experiment.py              the config schema (CableSpec/SupportSpec/...)
experiments/*.yaml         one file per cable — this is what you edit
methods/
  catenary.py              analytic solution
  cable_ode.py             the governing ODE, integrated + verified
  hang_common.py           scenario, settling, output protocol, tape shape
  hang_newton_cable.py     Newton rod (point pin / pin / tape middle support)
  hang_physx_capsule.py    PhysX capsule chain
  hang_warp.py             Warp XPBD rod (--bend-length)
  hang_newton_engine.py    Newton via USD  (blocked — see above)
  hang_physx_fem.py        PhysX deformable (no readback — see above)
analysis/
  compare.py               metrics.csv, overlay, errors, report.tex
  make_thesis_report.py    the full PDF write-up
  decision_brief.py        2-page "which one do we adopt?" brief
  warp_bend_sensitivity.py Warp stiffness study
  sweep_report.py          hyperparameter-sweep report
image_utils/
  measure_length.py        photo → length + profile (non-interactive)
  extract_cable_profile.py full interactive pipeline (SAM, clicked scale)
results/                   per-run outputs
```

## Requirements

- **Isaac Sim 6.0+** at `~/isaacsim` (or set `ISAAC_SIM_PATH`). Newton and Warp
  ship inside it. Run the simulation scripts with `~/isaacsim/python.sh` — the
  system Python has an incompatible NumPy/matplotlib.
- `pdflatex` for the PDF reports (the `.tex` is written regardless).
- `image_utils/extract_cable_profile.py` additionally needs
  `ultralytics opencv-contrib-python scipy` (the non-interactive
  `measure_length.py` needs only what Isaac Sim already provides).
```
