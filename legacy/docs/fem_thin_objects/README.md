# Why Isaac Sim's voxel FEM works for a sphere but not for a cable

Short version: **Isaac Sim (PhysX) fills a deformable body with equal-sized
cubes. A sphere is the same size in every direction, so the cubes fit it well.
A cable is 223 times longer than it is thick, so cubes sized for its length are
bigger than its whole cross-section — and "just add more voxels" does not fix
it.**

The one-page version for showing someone else is
[fem_sphere_vs_cable.pdf](fem_sphere_vs_cable.pdf) (source:
[fem_sphere_vs_cable.tex](fem_sphere_vs_cable.tex)).

---

## 1. How the FEM mesh is built

A PhysX volumetric deformable has two meshes:

- a **collision mesh** that follows the surface, and
- a **simulation mesh** made of cubic voxels (each cube is split into
  tetrahedra). This is the one the physics is solved on.

You control the simulation mesh with one number, the **hexahedral resolution
`N`**: the number of voxels along the body's *longest* dimension. Because the
voxels are cubes, that fixes the voxel size everywhere:

```
h = L_max / N
```

So the number of voxels across the *thinnest* dimension `d` is

```
n_across = N * d / L_max
```

Everything below follows from that one line.

## 2. Sphere: good

A sphere has `d / L_max = 1`.

| | |
|---|---|
| Resolution | `N = 10` |
| Voxels across, any direction | 10 |
| Total voxels | about 520 |
| Longest chain of elements | about 10 |

Every voxel is useful, the elements are well shaped, and the solver only has to
pass forces across about 10 elements. What you care about for a sphere (how it
squashes and bounces) depends on its bulk volume, which is well resolved.

## 3. Cable: bad

Our cable (from [cable_config.py](../../methods/cable_config.py) and the
3-point experiment): length `L = 893 mm`, diameter `d = 4 mm`, so
`L / d = 223`.

With `N = 130` (the cap in
[hang_physx_fem.py](../../methods/hang_physx_fem.py)):

```
h        = 893 / 130 = 6.9 mm      <- one voxel is bigger than the cable
n_across = 130 / 223 = 0.58
```

The cable sits inside a single row of 6.9 mm cubes. The solver simulates that
square bar, not the 4 mm round cable. Bending stiffness goes as the fourth
power of thickness, so this matters a lot:

```
I_bar   = h^4 / 12      = 1.9e-10 m^4
I_cable = pi * r^4 / 4  = 1.3e-11 m^4      -> about 15x too stiff in bending
```

For a hanging cable, bending and weight are the whole story, and bending is set
entirely by the dimension the mesh cannot see.

## 4. Why "make more voxels" does not fix it

To get a modest 4 voxels across the cable you need

```
N = 4 * L / d = 4 * 893 / 4 = 893      (h = 1 mm, 893 * 4^2 = about 14,000 voxels)
```

Three things go wrong:

1. **Every voxel across costs 223 along.** You cannot refine only the
   cross-section; the voxels are cubes.
2. **The solver cannot converge along the chain.** PhysX solves iteratively,
   and force travels roughly one element per iteration. A chain 893 elements
   long needs hundreds of iterations per step (the sphere needs about 10). With
   too few, the cable stretches and sags like a rubber band.
3. **The system is very stiff.** In an 80 MPa cable an elastic wave crosses a
   1 mm element in about 5 microseconds, roughly 760 times shorter than the
   1/240 s physics step. Small, stiff elements with a large step are exactly
   where this kind of solver struggles.

## 5. The workaround in our script, and what it costs

[hang_physx_fem.py](../../methods/hang_physx_fem.py) avoids the problem by
making the cable **fat**: simulation radius 12 mm instead of 2 mm (6 times),
which gives 3 voxels across at `N = 112`. To keep the fat rod behaving like the
thin one, the material is rescaled with `q = r_real / r_sim = 1/6`:

```
E_sim   = E   * q^4  = 62 kPa       keeps bending stiffness EI
rho_sim = rho * q^2  = 66 kg/m^3    keeps mass per length
```

What this gets right: bending stiffness, torsional stiffness, weight per length.

What it gets wrong:

- **Stretch stiffness `EA` is 36 times too soft** (1005 N becomes 28 N), since
  `EA` scales as `q^2`, not 1.
- **The shape is 6 times too thick** (24 mm instead of 4 mm), so anything
  involving contact — clips, connectors, routing through gaps, a gripper — is
  wrong.
- The rest shape must be straight (auto-attachment fails on a curved mesh), so
  the cable has to be dragged into place rather than started in its hanging
  shape.

Separately, on Isaac Sim 6.0 the FEM body builds and simulates but its node
positions cannot be read back, so this method currently produces no profile at
all (see the main [README](../../README.md)).

## 6. What about a thick cable (2 to 3 cm)?

Take the same length (893 mm) and the same stand-in material, but a diameter of
20 to 30 mm. This has **not been run** (Isaac Sim is not on this machine, and
the FEM readback is broken on Isaac Sim 6.0); it is the same calculation as
above.

| | thin cable, 4 mm | thick cable, 20 mm | thick cable, 30 mm |
|---|---|---|---|
| `L / d` | 223 | 45 | 30 |
| voxel size at `N = 130` | 6.9 mm | 6.9 mm | 6.9 mm |
| voxels across (`d / h`) | 0.58 | 2.9 | 4.4 |
| `N` for 4 voxels across (`4 L / d`) | 893 | 179 | 120 |
| rescaling of `E`, `rho` needed | yes | no | no |
| wave crossing one voxel vs 1/240 s step | 760x shorter | 110x shorter | 110x shorter |

**Verdict: workable, but coarse.**

- The cross-section is actually meshed: 3 to 4 voxels across at the existing
  cap, and a 30 mm cable reaches the 4-across target at `N = 120`.
- No fat-rod trick is needed, so stretch stiffness and contact shape are
  correct. The script's 24 mm "fat rod" is exactly a cable in this range; the
  difference is that here 24 mm would be the real size.
- The element chain is 130 to 180 long instead of 893. The script already uses
  80 solver iterations for a chain of 112.
- It is still stiffer numerically than the sphere, and 3 voxels across is a
  coarse mesh for bending, so expect some stiffness error. How much is not
  quantified here.

One thing changes physically: with `E = 80 MPa` a cable this thick is stiff.
Its bending length `(EI / w)^(1/3)` is 0.44 m (20 mm) to 0.58 m (30 mm),
comparable to the 0.89 m length, so it hangs like a bent beam rather than a
catenary. That is the regime where a volumetric model is useful. A real 2 to
3 cm cable is probably much softer than 80 MPa; that value is the stand-in for
the thin charger cable.

## 7. What to use instead

A thin cable is a one-dimensional object, so model it as one. In a rod model the
cross-section is not meshed; it enters through the numbers `EI`, `EA` and `GJ`.
Cost grows only with length, and the real 4 mm geometry is kept.

This repository already has three such models, all matching the photographed
cable to within 33–75 mm RMS:

| method | model |
|---|---|
| `physx_capsule` | rigid capsule chain with joints (native Isaac Sim) |
| `newton_cable` | Cosserat rod (Newton) |
| `warp_rod` | XPBD elastic rod (Warp) |

Keep volumetric FEM for compact soft bodies: spheres, blocks, soft gripper pads.

## 8. Where every number comes from

**Inputs** (read from the repository, not chosen by me):

| number | value | source |
|---|---|---|
| cable length `L` | 893 mm (0.8932 m) | measured from the photo; `length_m` in [config.resolved.yaml](../../results/apple_cable_3pt_150/config.resolved.yaml) |
| cable radius `r` | 2 mm (`d = 4 mm`) | [cable_config.py](../../methods/cable_config.py), stand-in |
| Young's modulus `E` | 80 MPa | [cable_config.py](../../methods/cable_config.py), stand-in |
| density `rho` | 2390 kg/m^3 | [cable_config.py](../../methods/cable_config.py), stand-in |
| resolution cap | 130 | `min(..., 130)` in [hang_physx_fem.py](../../methods/hang_physx_fem.py) |
| fat radius `r_sim` | 12 mm | `--sim-radius` default in [hang_physx_fem.py](../../methods/hang_physx_fem.py) |
| physics step | 1/240 s | `--physics-dt` default in [hang_physx_fem.py](../../methods/hang_physx_fem.py) |
| 33–75 mm RMS | 32.8, 51.2, 65.3, 75.1 | `rmse_vs_real_mm` in [metrics.csv](../../results/apple_cable_3pt_150/metrics.csv) |

**Illustrative choices** (mine, to make the comparison concrete): sphere
resolution `N = 10`, and "4 voxels across" as the target for the cable.

**Derived** (each from the inputs above):

| number | calculation |
|---|---|
| aspect ratio 223 | `L / d = 893 / 4` |
| voxel size 6.9 mm | `h = L / N = 893 / 130` |
| 0.58 voxels across | `d / h = 4 / 6.9` |
| bending 15x too stiff | `(h^4 / 12) / (pi r^4 / 4) = 1.9e-10 / 1.3e-11` with `h = 6.87 mm`, `r = 2 mm` |
| sphere, about 520 voxels | `pi / 6 * N^3 = 0.524 * 1000` |
| `N = 893` for 4 across | `4 * L / d` |
| about 14,000 voxels | `893 * 4 * 4` |
| thick cable `L / d` 45, 30 | `893 / 20`, `893 / 30` |
| thick cable 2.9, 4.4 voxels across | `20 / 6.87`, `30 / 6.87` |
| thick cable `N` 179, 120 | `ceil(4 * 893 / 20)`, `ceil(4 * 893 / 30)` |
| thick cable 110 times | `6.87 mm / 183 m/s = 37.5e-6 s`; `(1/240) / 37.5e-6` |
| thick cable bending length 0.44, 0.58 m | `(EI / w)^(1/3)`, `EI = E pi r^4 / 4` (0.63, 3.18 N m^2), `w = rho pi r^2 g` (7.4, 16.6 N/m) |
| wave speed 183 m/s | `sqrt(E / rho) = sqrt(80e6 / 2390)` |
| 5.5 microseconds | `1 mm / 183 m/s` |
| 760 times | `(1/240 s) / 5.5e-6 s` |
| fat rod `N = 112` | `ceil(3 L / (2 r_sim)) = ceil(3 * 0.893 / 0.024)`, the formula in the script |
| 3 voxels across fat rod | `24 mm / (893 / 112 mm)` |
| `q = 1/6` | `r / r_sim = 2 / 12` |
| `E_sim = 62 kPa` | `80e6 / 6^4` |
| `rho_sim = 66 kg/m^3` | `2390 / 6^2` |
| `EA` 1005 N to 28 N (36x) | `E pi r^2 = 80e6 * pi * 0.002^2`; fat rod `62e3 * pi * 0.012^2` |

**Not a number from anything:** "hundreds of iterations" and "about 10" are a
scaling argument (iterations needed grow with the length of the element chain),
not measurements.

## 9. Caveats

- `N = 130` is the cap used in our script, not a limit I have verified in
  PhysX itself. The argument does not depend on it: at any `N`, the cable gets
  `N / 223` voxels across.
- The 15x figure assumes each voxel is simulated as a full solid cube, which is
  how the PhysX voxel mesh behaves as far as I know; it has not been measured
  here because of the readback problem above.
- Material values (`E = 80 MPa`, density 2390 kg/m^3) are the stand-ins used
  across the benchmark, not measurements of this cable.

## Build the PDF

```bash
cd docs/fem_thin_objects
pdflatex fem_sphere_vs_cable.tex
```
