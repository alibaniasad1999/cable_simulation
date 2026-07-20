# Cable hang benchmark — Isaac Sim 6, Newton, PhysX, Warp

Compare how different simulators predict the shape of a cable hanging from 2 or
3 equal-height supports, against an **exact analytic solution** and against a
**real cable measured from a photograph**.

Everything is driven by one command:

```bash
python3 run_benchmark.py --length 1.0 --span 0.8 --points 2
```

which writes a CSV table of per-method errors, comparison figures, and a
LaTeX report.

---

## The problem

A cable of length `L` hangs from `N ∈ {2, 3}` supports, **all at the same
height** `H`, evenly spaced across a horizontal span `S`:

```
   2 supports:   x = 0                    x = S
   3 supports:   x = 0        x = S/2     x = S
                 ●─────╮             ╭────●          all at z = H
                        ╰───────────╯
```

Equal support heights are not a simplification for convenience — they are what
makes the problem **exactly solvable**. A perfectly flexible, inextensible cable
between equal-height supports is a catenary,

$$ z(x) = H + a\left[\cosh\frac{x - S/2}{a} - \cosh\frac{S}{2a}\right],
\qquad L = 2a\sinh\frac{S}{2a} $$

and with three equal-height supports the cable separates into **two independent
sub-catenaries** of span `S/2` and length `L/2`. So both scenarios have a closed
form to measure against, with no fitted parameters.

---

## Methods

| key | model | runs under | status |
|---|---|---|---|
| `newton_cable` | Newton Cosserat rod, `ModelBuilder.add_rod` + `SolverVBD`. Real bend stiffness `EI`. | Newton standalone (no Kit) | ✅ |
| `newton_engine` | Isaac Sim 6 native Newton engine via UsdPhysics spherical-joint chain. `EI = 0`. | Isaac Sim | ❌ blocked |
| `physx_capsule` | PhysX rigid capsule chain, D6 joints with beam-theory bending springs. | Isaac Sim (PhysX) | ✅ |
| `physx_fem` | PhysX volumetric deformable (3-D continuum). | Isaac Sim (PhysX) | ⚠️ no readback |
| `warp_rod` | XPBD elastic rod written directly in Warp. | Warp only, CPU or GPU | ✅ |
| *(reference)* | analytic catenary | numpy | ✅ |

Three methods produce results on this build; two are blocked by Isaac Sim 6.0.0-rc.59
limitations documented in **Known limitations** below.

`warp_rod` exists to be the method that owes nothing to Isaac Sim or Newton — if
it agrees with them, the agreement is not an artefact of a shared code path.

### All methods solve provably the same problem

The scenario is written **once** to `run.json` and passed to every method with
`--scenario`. Each is initialised **on the analytic catenary**, so what the
benchmark measures is how far a solver *drifts from equilibrium*, not how well
it recovers from an arbitrary transient. (`physx_fem` is the exception — PhysX
auto-attachment requires a straight rest mesh, so it starts straight and is
dragged into place. Its transient is genuinely larger.)

---

## Quick start

```bash
# one scenario, all methods
python3 run_benchmark.py --points 2

# both scenarios
python3 run_benchmark.py --all

# just the Newton rod, 3 supports, longer cable
python3 run_benchmark.py --points 3 --length 1.2 --span 0.8 --methods newton_cable

# see the commands without running them
python3 run_benchmark.py --dry-run
```

Inputs:

| flag | meaning | default |
|---|---|---|
| `--length` | cable length `L` [m] | 1.0 |
| `--span` | distance between the outer supports `S` [m] | 0.8 |
| `--height` | support height `H` [m] (shared by all) | 1.0 |
| `--points` | number of supports, 2 or 3 | 2 |
| `--segments` | discretisation | 60 |
| `--max-time` | sim-time cap [s] | 8 |

`L` must exceed `S`, otherwise the cable is taut and there is no sag to compare.

---

## Outputs

```
results/<run>/
  run.json          the shared scenario
  metrics.csv       ← THE comparison table, one row per method
  overlay.png       every method + the catenary on one axis
  errors.png        height error vs x
  report.tex        standalone LaTeX report
  <method>/
    profile.csv     settled shape, columns x_m,z_m
    trajectory.csv  wide time series: t, n000_x…, n000_z…
    meta.json       timings, convergence, solver settings
    profile.png     that method vs the catenary
    run.log
```

`metrics.csv` columns:

| column | meaning |
|---|---|
| `rmse_mm`, `max_err_mm` | height error against the catenary |
| `sag_mm`, `sag_err_mm` | lowest point, and its error |
| `arc_drift_pct` | **change in the cable's own length** — reference-free |
| `wall_s`, `settled` | cost, and whether motion actually decayed |
| `rmse_vs_real_mm` | error against the photographed cable, when `--real` is given |

### How to read them

- **`arc_drift_pct` is the honest accuracy check.** It needs no reference
  shape. A solver that stretches the cable 2 % has failed regardless of how
  catenary-like its profile looks.
- **Matching the catenary is not automatically a win.** A spherical-joint chain
  has `EI = 0`, exactly like the analytic catenary. It agrees because it shares
  the idealisation, not because it is more faithful to a real cable.
- **An unsettled run is provisional.** Its profile is a time-average over the
  final window, not a true equilibrium.

---

## Where the catenary stops being a valid reference

The catenary is the `EI → 0` limit. Bending competes with gravity over the
**elasto-gravitational length**

$$ \ell = (EI/w)^{1/3} $$

and the catenary is only a good description when `ℓ` is small compared with the
span the cable must curve over. For the default cable (`r = 2 mm`, `E = 80 MPa`)
`ℓ ≈ 150 mm`, which is **15 % of a 1 m cable** — not negligible.

Measured consequence, sweeping the Warp rod's bend compliance on the 3-support
case (everything else fixed):

| bend compliance | RMSE vs catenary |
|---|---|
| `5.8e-4` (physical `EI`) | 129.7 mm |
| `5.8e-2` | 29.9 mm |
| `5.8e0` (nearly floppy) | **5.0 mm** |

The rod converges to the catenary exactly as it is made floppier, which is what
theory demands. So the large 3-support residual is **real physics, not solver
error**: the middle support forces a slope *kink*, a rod with real bending
stiffness refuses to kink, and it buckles into an arch instead. `report.tex`
prints `ℓ/S` and warns when the reference is being used outside its range.

For the 2-support case bending is a mild perturbation and the catenary is a
sound reference (`newton_cable` reaches ~1.5 mm RMSE).

---

## Comparing against a real cable

`image_utils/extract_cable_profile.py` measures a real cable's shape from a
photo (SAM segmentation, metric scale from a measured in-plane reference) and
writes the same `x_m,z_m` CSV the simulations produce.

```bash
# photo -> profile.csv
cd image_utils && ./run_image.sh media/IMG_0501.JPG

# fold it into the comparison
python3 analysis/compare.py --run results/<run> \
    --real image_utils/results/IMG_0501/profile.csv
```

See `image_utils/README.md` for the photography checklist — the shot must be
square-on, the cable must hang in one plane, and the scale reference must lie
in that same plane.

> The static shape identifies the cable's **length and span**, not its
> stiffness: a stiff and a floppy cable of the same length hang almost
> identically. Identifying `EI` needs a **dynamic** experiment (film it
> oscillating, measure the frequency). `extract_cable_profile.py video` produces
> the per-frame profile for that.

---

## Known limitations on this build

Measured on **Isaac Sim 6.0.0-rc.59**, each verified by bisection rather than
inferred. These are the reasons the earlier `hang_benchmark` produced empty
result directories.

**`newton_engine` cannot run this scenario at all.** There is no working
configuration for a both-ends-fixed cable through Newton's USD/engine path:

1. Bodies and joints must sit under a prim with `UsdPhysics.ArticulationRootAPI`
   — loose joints are rejected outright.
2. An articulation must be a **tree**. A cable held at both ends is a closed
   loop: with one world weld the simulation view builds, with two it fails.
3. Marking the second anchor `excludeFromArticulation` lets the view build, but
   the engine then **silently ignores** that constraint — the cable hangs from
   one end and drops to `min_z = 0.18 m` instead of `0.73 m`.
4. Anchoring via `physics:kinematicEnabled` also fails: that attribute on a body
   inside an articulation is itself what breaks the view.

The method fails fast with this diagnosis rather than crashing opaquely.
`newton_cable` drives the **same Newton solver** through `ModelBuilder.add_rod`
and supports both boundary conditions natively — use it.

**`physx_fem` builds and simulates but its state cannot be read back.** Isaac
Sim 6 replaced the whole deformable API:

- `deformableUtils.add_physx_deformable_body` and `add_deformable_body_material`
  are gone. A volume deformable is now an explicit **hierarchy** —
  `create_auto_volume_deformable_hierarchy` with a root Xform (which must
  already exist and be `Imageable` but not a `Gprim`) holding a simulation and a
  collision tet mesh cooked from the source triangle mesh.
- `PhysxSchema.PhysxAutoAttachmentAPI` is replaced by
  `create_auto_deformable_attachment`, which binds the deformable **root**, not
  the source mesh.
- `isaacsim.core.prims.DeformablePrim` is removed; the replacement in
  `isaacsim.core.experimental.prims` exposes `get_nodal_positions()`.

All of that now works — the body cooks, the material binds, the attachments
succeed. But the deformable tensor view is created with a **`None` backend**, so
`get_nodal_positions()` fails and there are no nodal positions to export. Same
class of gap as the Newton engine's tensors view. Reading the simulation
TetMesh points straight from USD/Fabric is the likely workaround; **not yet
implemented**. The method raises with this diagnosis rather than failing
silently.

Other build-specific findings, documented in `methods/hang_newton_engine.py`:

- Assigning `XPBDSolverConfig` breaks state readback entirely (only the default
  MuJoCo solver has a working tensors backend). **The original `hang_newton.py`
  requested XPBD, so it never reached the point of writing a CSV.**
- `isaacsim.physics.newton.tensors` must be enabled explicitly.
- The older `timeline.play()` + `simulation_app.update()` route does not step
  physics headlessly at all — a free-falling body stays exactly at its initial
  height. This is a *dangerous* failure for this benchmark: a frozen cable sits
  precisely on the catenary it was initialised with and scores a perfect
  **0.0 mm RMSE**. Every method asserts that the cable actually moved.
- `physics_dt`, `rendering_dt` and `cfg.physics_frequency` are all ignored; the
  engine advances a fixed `1/499.6 s` per `world.step()`.
- `cfg.armature` defaults to `0.1`, an artificial joint inertia that completely
  dominates capsule links of ~0.5 g.

---

## Measured results

2 supports, `L = 1.0 m`, `S = 0.8 m`, 60 segments, RTX 3080. Catenary sag
265.4 mm.

| method | RMSE | max err | sag err | arc drift | wall |
|---|---|---|---|---|---|
| `physx_capsule` | **0.69 mm** | 1.17 mm | −0.7 mm | **+0.034 %** | 26 s |
| `newton_cable` | 1.29 mm | 6.87 mm | +1.8 mm | +0.215 % | 144 s |
| `warp_rod` | 2.15 mm | 4.91 mm | +1.0 mm | +0.043 % | 4 s |
| `newton_engine` | — | — | — | — | blocked (see below) |

Everything agrees with the closed-form catenary to within a few millimetres on
a 1 m cable, and with each other. The PhysX capsule chain is both the most
accurate and cheap; the Newton rod pays for its `EI` term in solver iterations;
the Warp rod is by far the fastest and its residual is a visible left-right
asymmetry from the sequential Gauss–Seidel sweep (see `errors.png`).

---

### 3 supports — read this before comparing the numbers

3 supports, same cable. Catenary sag 132.7 mm.

| method | middle support | RMSE | arc drift | wall |
|---|---|---|---|---|
| `newton_cable` | **clamp** (2 capsules held at the mid node) | 2.1 mm | +1.44 % | 142 s |
| `physx_capsule` | **pin** (translations locked, rotations free) | 26.4 mm | −0.15 % | 25 s |
| `warp_rod` | **pin** (one node, inverse mass 0) | 38.5 mm | −0.68 % | 4 s |

> ⚠️ **This spread is a boundary-condition difference, not a solver ranking.**
> Two equal-height catenaries meet at the middle support with *opposite* slopes,
> so the exact solution has a kink there. A **clamp** imposes that kink by
> construction — `newton_cable` reproduces the reference partly because it is
> being told the answer. A **pin** lets the cable pivot, so bending stiffness
> resists the kink and pushes the low points outward, which is what the other
> two show in `overlay.png`.
>
> `newton_cable --mid-support pin` holds only one capsule instead of two, and
> **measurably changes nothing** (RMSE 2.1 mm, arc +1.44 % either way): holding
> one capsule's orientation already pins that tangent. This is an `add_rod` API
> limitation — it exposes capsule *bodies*, not nodes, and every way of holding
> a body also holds its orientation, so a true pinned interior support cannot be
> expressed without a loop-closing joint to the world. The Warp rod can express
> one (a node with inverse mass 0 constrains position only) and lands at 39 mm.
>
> Note also that `newton_cable`'s arc drift jumps from +0.22 % (2 supports) to
> **+1.44 %** (3 supports): the imposed kink is bought by stretching the cable.
> That is the cost showing up in the reference-free metric.
>
> Which model is physically right depends on your rig — a cable *draped over* a
> hook is a pin; one gripped in a fixture is a clamp. The 2-support case has no
> such ambiguity: all three methods hold the ends identically and agree to
> within ~2 mm.

---

## Solver convergence matters more than you would expect

`newton_cable`'s VBD iteration count directly controls how nearly inextensible
the cable is (2-support case, `L = 1 m`, `S = 0.8 m`):

| iterations | arc drift | RMSE | wall |
|---|---|---|---|
| 20 | +1.85 % | 9.9 mm | 6 s |
| 100 | +0.40 % | 2.4 mm | 29 s |
| **200** | **+0.22 %** | **1.5 mm** | 44 s |
| 400 | +0.12 % | 2.0 mm | 114 s |

Below ~100 iterations the cable visibly stretches under its own weight — a
solver artefact, not a property of the model. 200 is the default.

---

## Layout

```
run_benchmark.py         driver: one scenario -> every method -> analysis
methods/
  cable_config.py        the physical cable, one source of truth
  catenary.py            analytic solution (+ CLI: python3 catenary.py --help)
  hang_common.py         scenario, settling, output protocol
  isaac_env.py           locate Isaac Sim, expose its Warp/Newton without Kit
  hang_newton_cable.py   Newton Cosserat rod + VBD
  hang_newton_engine.py  Isaac Sim Newton engine via USD  (blocked, see above)
  hang_physx_capsule.py  PhysX capsule chain + D6
  hang_physx_fem.py      PhysX volumetric deformable
  hang_warp.py           Warp XPBD rod
analysis/compare.py      metrics.csv, figures, report.tex
image_utils/             photo -> real cable profile (SAM segmentation)
results/                 per-run outputs
```

## Requirements

- Isaac Sim 6.0+ at `~/isaacsim`, or set `ISAAC_SIM_PATH`. Newton 1.2.0 and
  Warp ship inside it — nothing extra to install for the simulations.
- `image_utils` additionally needs `ultralytics opencv-contrib-python scipy
  matplotlib imageio`.
- `pdflatex` for `compare.py --pdf` (otherwise `report.tex` is still written).
