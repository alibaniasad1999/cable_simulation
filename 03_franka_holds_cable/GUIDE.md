# Build it yourself: the Franka holding the scanned Ethernet cable

> Study notes for writing your own versions of
> [`ethernet_scene.py`](ethernet_scene.py) (simulation) and
> [`../04_pointcloud_vs_sim/compare_to_scan.py`](../04_pointcloud_vs_sim/compare_to_scan.py)
> (comparison). Each stage gives the idea, the Newton/numpy calls to look up,
> and the **numbers you should get**, so you can check yourself. The finished
> scripts are in the repo. Look at them only when stuck.

**Input:** the tube-fit result of your scan (`results/Ethernet_tube_fit/centerline.csv`
and `summary.json`) and one JSON file with the cable's properties.
**Output:** the simulated cable's centreline, and how far (in mm) it is from the
scanned one, for several bending stiffnesses.

**The one idea behind everything:** start the simulated cable *exactly where the
real one is*, hold it *the way the gripper holds it*, and let gravity act. If
the model were perfect, nothing would move. **How far it moves is the
disagreement, and the stiffness that moves it least is the cable's stiffness.**

Newton version used for every number below: `newton 1.6.1`, `warp-lang 1.18.0`,
CPU. Work in **metres** (the tube fit's CSV is in metres). Every check uses the
reference JSON as committed (`EI = 5e-3`, `EA = 200`, `kGA = 20`, 20 substeps ×
20 iterations, plug end fixed), pointed at the test scan of Stage 0.

---

## Stage 0. A test scan with a known shape

Before touching your real scan, make a fake one. Then every check below has an
exact answer, and you can debug without wondering whether the scan is the problem.

Write a small script that builds this curve (all in metres, `r = 0.0036`) and saves
it **in the tube-fit format**:

```
hanging   t ∈ [0, 1], 200 points:
          x = 0.45 + 0.12 t^1.5
          y = 0
          z = max( 0.35 − 0.35 (1 − (1 − t)²) + r t ,  r )
lying     x from 0.57 to 0.72, y = 0, z = r, 400 points (drop the first, it repeats)
loop      θ from −π/2 to 1.5π + 0.6, 400 points:
          x = 0.72 + 0.07 cos θ,   y = 0.07 + 0.07 sin θ,
          z = r + 2r·lift(θ)      the second pass CROSSES OVER the first:
          lift rises linearly 0 → 1 over θ ∈ [1.5π − 0.45, 1.5π],
          then falls 1 → 0 over θ ∈ [1.5π, 1.5π + 0.5]
```

Why the lift: where a real cable crosses itself, one strand **lies on** the other
(centres one diameter apart). My first test scan let the strands pass *through*
each other at the same height. Self-contact then fought that overlap forever, and
the simulation never settled. Physically impossible input gives nonsense, so build
your test data as carefully as your code.

Stack the three parts, resample to **151 nodes at equal arc length**, and write:

- `centerline.csv` with header `s_m,x_m,y_m,z_m,tx,ty,tz,supported`
  (tangents from `np.gradient`, normalised; `supported` all 1),
- `summary.json` with `{"radius_mm": 3.6, "ply_units": "mm"}`.

**Check:** length 1020.9 mm along the 151 nodes. Plotted from the top, a straight
line from the gripper into a loop of 70 mm radius, the loop crossing over its start
6.8 mm higher.

---

## Stage 1. The JSON config

Every physical number goes in one file, never in the code
([`configs/ethernet_cat6.json`](../configs/ethernet_cat6.json) is the reference).
Sections: `scan`, `cable`, `contact`, `robot`, `sim`, `sweep`, `output_dir`.

- JSON has no comments. Use keys starting with `_` as comments, and **strip them
  when loading** (a small recursive dict comprehension), so a typo in a comment
  key can never reach the code.
- Resolve relative paths against the **repository root** (`Path(__file__).resolve().parents[1]`),
  not the current directory, so the script works from anywhere.

**Check:** `load_config(...)["cable"]` has no key starting with `_`.

---

## Stage 2. Load and orient the scan

- Read the CSV with `np.genfromtxt(path, delimiter=",", names=True)`. Columns are
  then `data["x_m"]` etc.
- **Which way is up? (2b below.)** Find it first, and turn the nodes so up is +z:
  `X_world = X @ R.T`. Everything after this uses the turned nodes.
- **Which end is the gripper?** The cable hangs from the gripper and its other end
  lies on the table, so the gripper end is the **higher** one. Compare `z` of the
  first and last node. If the last is higher, reverse the arrays. After this, node 0
  is always the gripper end. Allow `"start"` / `"end"` in the JSON to override.
- Arc length `s` = cumulative sum of segment lengths, starting at 0.
- Radius: from the JSON, or `summary["radius_mm"] / 1000`.
- **Table height:** the lowest centreline point lies on the table, one radius above it:

```
ground_z = min z over supported nodes − r
```

**Check (test scan):** 151 nodes, 1019.5 mm, gripper at the start of the CSV,
`ground_z = 0.0000`. Write the same scan reversed and check you still get "gripper
at the end" and the same numbers.
**Check (your scan):** 892.7 mm, `r = 3.60` mm.

### 2b. Which way is up

A scanner's frame is almost never z-up (scanning apps often use y-up, a depth
camera has y pointing **down**). Assume z-up on such a scan and the part lying on
the table becomes a tall vertical loop, the gripper end lands at table height, and
the robot is pushed into the table. That really happened with your Ethernet scan.

Measure "up" from the cable itself. Like the tube fit measuring the radius, this
never looks at the simulation. The idea: **the table is the plane that the most
cable rests on, with all of the cable above it.** With the true up, every node lying
on the table sits at exactly the same, lowest height. Tilt the direction even a
little and only a few stay at the bottom.

1. **Coarse:** for ~20 000 directions spread over the sphere (Fibonacci sphere),
   count the nodes within **3 mm** of the lowest node. Keep only directions where
   the cable hangs at least 5 cm above that and an **end** is the highest point (the
   gripper). Take the directions with (nearly) the most nodes at the bottom.
   Tie-break: the cable should never climb back up once it leaves the gripper, so
   prefer the direction with the least climb:

   ```
   rise(d) = Σ max(0, h[k+1] − h[k]),   h = X·d, walked from the higher end
   ```

   Vectorise: `H = X @ D.T` is (nodes × directions).
2. **Fine:** within 6° of that, on a 0.1° grid, count nodes within **0.5 mm** of the
   bottom. 3 mm is too loose for a stiff cable whose loop arches off the table: a
   slightly tilted plane can graze more of the arch than the real table.
3. **Polish:** fit a plane (SVD) through the resting nodes, keep only those within
   1 mm, fit again. Use its normal if the nodes span an **area** (second singular
   value > 0.2 × first) and it's within 5° of the estimate.

Up or down comes out for free: with `−up` the "bottom" is the gripper end, a single
node, so it never wins step 1. (My first version scored only the climb. A cable
"hanging upward" from the floor is monotone too, so the −y-up case came out exactly
180° wrong.)

**The limit:** if the cable rests on the table along a **line only** (a stiff loop
arching up, touching down in one straight stretch), the tilt *about that line* is
not determined by the cable: it can be off by a degree or two. Detect that case
(the resting nodes' second singular value is small) and say so. The real fix is
the table itself, e.g. its normal from the scan in CloudCompare, given as `scan.up`.

Then build the rotation that takes `up` to `+z` (Rodrigues:
`R = I + K + K²/(1 + up·z)`, `K` = cross-product matrix of `up × z`; for `up = −z`
use a half turn about x). Keep `R` in `meta.json` so the comparison and the
`.ply` output can use it.

**Check:** take the test scan, turn it into several frames yourself (y-up, a
camera's −y-up, tilted 10° and 25°, a few random rotations, plus an offset). The
detected up must match the true one to ~0.01° in every frame. Then try a settled
**stiff** cable (Stage 12's truth): it touches the table along a line, and you'll
get ~1.4° and the line NOTE. That's the limit above, not a bug.

---

## Stage 3. Plan the simulated cable

The scan only shows the cable *between* the fingers and the plug. The simulated
cable needs three parts:

```
   grip (inside the fingers)      scanned stretch            plug
 |=========|-------------------------------------------|========|
 s = −grip  s = 0                                   s = L_scan
```

Arc length `s` is measured from the **first scanned node**, so the grip has
negative `s`. This convention makes sim and scan directly comparable later.

**Segment length.** Take `h₀` from the JSON (10 mm). Make the grip a whole number of
segments, and let the plug absorb the rounding:

```
n_grip = max(1, round(grip / h₀))          h = grip / n_grip
n_seg  = n_grip + max(2, round((L_scan + plug_len) / h))
plug   = n_seg·h − grip − L_scan             (≈ plug_len, within h/2)
n_plug = round(plug / h)
```

**End directions.** Use the tube fit's own tangents: `tx, ty, tz` of the first and
last node (turned with `R`, and negated if you reversed the cable). They come from
its spline, so they're exact. Only if a CSV has no tangents, average over ~2 cm:
`t₀ = unit(X[k] − X[0])`.

This matters more than it looks: the grip direction is the boundary condition.
1° at the grip moves the hanging part ~6 mm at 35 cm. With the 2 cm average, a
stiff cable (which curves within those 2 cm) failed the self-test: the true EI
moved 6.5 mm away from its own equilibrium.

**Starting centreline** (`init = "scan"`): a dense polyline

```
[ X[0] − grip·t₀ ,  X[0], X[1], …, X[−1] ,  X[−1] + plug·t₁ ]
```

resampled to `n_seg + 1` points at equal arc length (`np.interp` on cumulative
length, per coordinate).

**Check (test scan):** `h = 10.0` mm, 108 segments, 2 grip segments, 4 plug
segments, plug 39.1 mm, total length 1.080 m.
**Check (your scan):** 95 segments, plug ≈ 37 mm.

---

### 3b. Make crossings physically possible

Where the scanned cable crosses itself, one strand **lies on** the other: their
centres are one diameter (2r) apart. The tube fit can't see the touching sides and
may put the centrelines closer. Start the simulation like that and self-contact
pushes the strands apart **for as long as it runs**: the cable jumps by several mm
again and again and never settles. It's worst with the plug end fixed (4c),
because the free end can no longer slide them apart.

Fix the start shape, on the scan nodes, before planning:

1. Closest distance between every pair of polyline segments more than 6r apart
   along the cable (closer pairs are neighbours, which always "touch"). This is the
   classic segment–segment distance (Ericson, *Real-Time Collision Detection* §5.1.9).
   Vectorise it over all pairs.
2. For each pair closer than `2r + 0.3 mm`:
   - **crossing** (the gap is mostly vertical, or zero): lift the **upper** strand
     (higher closest point; a tie goes to the later one) by exactly what's
     missing: `δ = −d·|n_z| + √((2r+m)² − d²(1−n_z²))`;
   - **side by side** (gap mostly horizontal): push both apart sideways by half.
3. Apply each as a smooth Gaussian bump along the cable (σ = 2 cm). Take the
   per-node **maximum** over overlapping pairs, not the sum: one crossing involves
   several segment pairs. Never move nodes within 2 cm of the ends (the grip and
   plug boundary conditions).
4. Repeat until nothing overlaps, then let the plug take back the extra length the
   bump added, so the total stays `n_seg·h` (no pre-stretch).

Print what you changed. It's a correction to the data, so say so.

**Check:** on the Stage 0 test scan *without* the lift (strands at the same height),
the fix reports ~7.5 mm near s ≈ 510–590 mm. With the plug fixed, the run then
settles at t ≈ 12 s. Without the fix it was still jumping 43 mm at 20 s.

---

## Stage 4. Build the cable in Newton

Look up: `newton.Rod`, `newton.Rod.create_straight`, `ModelBuilder.add_rod`
(read the docstrings with `help(...)`. They're precise.)

### 4a. Stiffness from section rigidities

Create the rod **straight** (Stage 6 explains why), with the physical rigidities:

```
Rod.create_straight(start, direction, length = n_seg·h, segment_count = n_seg, radius = r,
                    stretch_rigidity = EA, shear_rigidity = kGA,
                    bend_rigidity   = EI, twist_rigidity = GJ)
```

Newton then derives the **per-joint** stiffness itself: rigidity / segment length.
Direct `stretch_stiffness=` / `bend_stiffness=` arguments to `add_rod` are per
joint and *not* scaled, so they'd change meaning when you change `h`. Prefer
rigidities.

**Check:** after `add_rod`, look at `builder.joint_target_ke` for the first rod
joint's 4 DOFs (stretch, shear, bend, twist). With the reference JSON:
`[2.0e4, 2.0e3, 0.5, 0.385]`, i.e. `EA/h, kGA/h, EI/h, GJ/h`.

### 4b. Mass: the capsule-cap trap

Newton's mass = shape volume × density. Each segment is a **capsule**: a cylinder
of length `h` **plus two half-spheres**. Using `density = μ / (π r²)` gives ~50 %
too much mass at `h = 10 mm`. Correct it:

```
V_capsule = π r² h + (4/3) π r³
density   = μ · h / V_capsule          (μ = mass per metre)
```

Pass it in `ModelBuilder.ShapeConfig(density=..., ke=..., kd=..., mu=..., margin=0, gap=...)`.

**Check:** sum `model.body_mass` over the free cable bodies (after zeroing the grip,
4c) = `μ·(L_scan + plug) + plug_mass` with a free plug end: 51.7 g on the test
scan (74.6 g without the correction). With the plug end fixed (4c), the plug
segments are kinematic too and the free cable is 45.9 g.

### 4c. Damping, grip, plug

- `bend_damping = τ · EI / h`, `twist_damping = τ · GJ / h` (τ = 2 ms),
  `stretch_damping = 0`. Damping changes how fast it settles, not where.
- **Grip:** the first `n_grip` bodies get zero `body_mass`, `body_inv_mass`,
  `body_inertia`, `body_inv_inertia`. VBD treats `inv_mass = 0` as kinematic and
  never moves them. This is the boundary condition.
- **Plug, free:** spread `plug_mass` over the last `n_plug` bodies. For each one,
  multiply mass **and inertia** by `f = (m + Δm)/m` and set the inverses to match.
  (`builder.body_inertia[b]` is a `wp.mat33`. Go through numpy:
  `np.array(I).reshape(3, 3)`.)
- **Plug, fixed** (`fix_plug_end`): make the plug bodies zero-mass instead, like
  the grip. They stay exactly where they start, which is only the scanned place with
  `init = "scan"`. With `drop`, leave the plug free. Holding the far end stops the
  lying part from sliding away, so the comparison is about the shape in between.

### 4d. Table and finalize

```
builder.add_ground_plane(height=ground_z, cfg=ShapeConfig(ke, kd, mu, ...))
builder.color()                # VBD refuses to run without colouring
model = builder.finalize()
```

Call `SolverVBD.register_custom_attributes(builder)` right after creating the
builder, before adding anything.

### 4e. Check that it bends like its EI (the "jelly" trap)

**Don't trust the stiffness you typed: measure it.** Build a separate tiny scene:
a straight piece clamped horizontally (2 zero-mass segments), 20 cm sticking out,
gravity on, no table. Let it settle and compare the tip drop with beam theory:

```
bending   δ_b = w L⁴ / (8 EI)        w = μ g
shear     δ_s = w L² / (2 kGA)
```

Keep `δ_b` under ~15 % of `L` (small deflection). With EI = 5e-3: δ_b = 17.7 mm.
Use exactly the scene's segment length, rigidities, damping, substeps and
iterations. `check_stiffness.py` does this.

What I found (10 mm segments, EI = 5e-3, 20 cm):

| EA [N] | kGA [N] | substeps × iterations | sim / theory |
|---|---|---|---|
| 2e4 | 7e3 | 10 × 20 | **10.8×** too soft. This was the "jelly". |
| 2e4 | 7e3 | 10 × 400 | 9.2× (more iterations don't help) |
| 200 | 7e3 | 10 × 50 | unstable (tip went **up**) |
| 2e4 | 7 | 10 × 50 | 1.51× |
| 200 | 20 | 10 × 20 | 1.27× |
| 200 | 20 | 5 × 100 | 1.18× |
| 200 | 20 | 10 × 50 | 1.07× |
| **200** | **20** | **20 × 20** | **1.06×** (cheapest within ~6 %) |

So: **in VBD, stretch and shear rigidities far above the bending one make bending
far too soft**, and this gets worse with shorter segments (20× at 20 mm segments,
79× at 10 mm with EI = 0.2). Real Cat6 has `EA ≈ 2e5 N`, but 200 N already keeps
it to ~0.1 % stretch under its own weight, so keep EA and kGA moderate. Small time
steps help more than iterations. The last ~6 % appears at every setting: it's the
discrete model, not convergence.

**Check:** with the reference JSON, `check_stiffness.py` reports `sim/theory 1.06`
at EI 5e-3, and 1.06 at EI 2e-2.

---

### 4f. Natural curl (rest curvature)

Ethernet cable remembers the coil it was wound in. A model whose rest shape is
straight can never hold a loop the way it does, whatever EI you choose. So allow a
curled rest shape: constant curvature `κ` [1/m] (coil radius `1/κ`) about a fixed
axis of the cable's own frame, `a = (cos φ, sin φ, 0)`. `φ` says to which side it
curls.

Build the rod in that rest shape, with explicit frames:

```
R_i      = rotation by κ·h·(i + ½) about a           (Rodrigues)
x_{i+1}  = x_i + h · R_i ẑ                            (segment along its frame's +Z)
Rod(points = x, quaternions = quat(R_i), radius, rigidities...)
```

Consecutive frames then differ by the same rotation `κh` about `a`. That's
constant curvature with no twist, and VBD stores it as the rest bend. The start pose
(Stage 6) is unchanged: the cable still starts on the scan. Only what it *wants* to
be changes. `φ` is relative to the start frames (Newton's parallel transport from
the gripper), so it's a fit parameter, not something you measure with a protractor.

**Check:** gravity off, first 2 segments clamped, start straight, `κ = 10 /m`. It must
curl into a flat circle of radius 100 mm (I get 100.1 mm), and `φ = 90°` must turn the
circle's plane by 90°.

### 4g. The hidden twist (needed as soon as κ > 0)

A centreline scan shows the cable's *shape* but not how it is **twisted** about its
own axis. For a straight-rest cable that doesn't matter. For a curled one it decides
where the curl points, everywhere along the cable. Starting with Newton's
parallel-transport frames means assuming *zero* twist, which is arbitrary, and it
made my fit test fail badly: the exact true values scored **28.5 mm** while a wrong
combination scored 3.6 mm.

The scanned cable is at rest, so its twist is the one with the **least energy for
that shape**. Find it before starting. Keep the positions and turn each segment by a
roll angle `θ_i` about its own axis (the grip segments stay at 0):

```
E(θ) = Σ_i  EI/h · |R(−θ_i) b_i − k|²  +  GJ/h · (θ_{i+1} − θ_i)²

b_i = bend between segments i and i+1 in the parallel-transport frames
      (rotation vector of q_i⁻¹ q_{i+1}, its x-y part)
k   = κ h (cos φ, sin φ)       the rest bend
```

Minimise with L-BFGS (`scipy.optimize.minimize(..., jac=True)`), using the analytic
gradient `∂/∂θ_i: EI/h (d_x l_y − d_y l_x) − GJ/h (θ_{i+1} − θ_i) + GJ/h (θ_i − θ_{i−1})`,
where `l = R(−θ_i) b_i` and `d = l − k`. Then start from `q_i ⊗ (rotation θ_i about z)`.

**Checks:**
- Give it the rest circle itself as the start curve: the bend afterwards must equal
  the rest bend (I get 2e-6 rad, against 0.08 rad per joint).
- Gradient against finite differences: ~1e-7 relative.
- A cable at equilibrium (simulate, restart from its own result, let it settle again):
  restarting from that, with the true values, must stay put (2.0 mm RMS, 4 mm moved).

**The limit:** this assumes the twist is at equilibrium. The part lying on the table
keeps whatever twist friction held when it came to rest (history again), and a
**fixed** plug end holds its roll. So with real data, expect the fitted curl and EI
to be *effective* values that make the shape match, not necessarily the cable's
true ones. Test them on a second scan.

---

## Stage 5. The Franka: kinematic, placed with IK

The robot is only there to *look* right. The cable's boundary condition is the grip
from 4c. So: place it with IK, then make every link zero-mass.

### 5a. Where the base goes

If the JSON doesn't give `base_xyz_m`: put the base on the table, on the side of the
grip **away from the cable**, facing the grip:

```
away = (mean of scan nodes − grip centre), with z set to 0, normalised
base = grip_centre − d · away,   base_z = ground_z
yaw  = atan2(away_y, away_x)          (the robot faces the grip)
```

`grip_centre = X[0] − ½·grip·t₀`. Base transform:
`wp.transform(base, wp.quat_from_axis_angle(wp.vec3(0, 0, 1), yaw))`.
Don't fix `d`: try `d = 0.4, 0.5, 0.6, 0.7` m (see 5d).

### 5b. The TCP target frame: how the fingers hold the cable

The fingers close along the hand's **y** axis. The hand's **z** axis points from the
wrist out through the fingertips. There are two ways to hold a cable end, and
they put the hand in very different places:

```
across (default, like the reference example)       along
the pads pinch the cable from the sides            the cable comes out of the fingertips
x = t₀                                             z = t₀
z = the most downward direction across the cable:  y = unit(up × z)   (horizontal)
    z = unit(−up − ((−up)·x) x)                    x = y × z
    (cable vertical → use the horizontal
     direction from the robot base instead)
y = z × x
```

`R = [x y z]` (columns) → quaternion. Write the matrix → quaternion conversion
yourself (the standard "trace" method). **Newton/Warp order is `(x, y, z, w)`.**
Test it on a 90° rotation about z before trusting it.

Which one is right is a fact of your real setup: look at the gripper in your scan
or photo. It's only visual: the cable's grip (4c) is the same either way.

A **flip** (x → −x, y → −y, i.e. 180° about z) is the same grasp, because the
fingers are symmetric, but it gives IK a different arm pose. Try both.

### 5c. IK on a Franka-only model

Look up: `newton.ik.IKSolver`, `IKObjectivePosition`, `IKObjectiveRotation`,
`IKObjectiveJointLimit`.

- Build a **second** builder with only the Franka, **base at the origin**. Then
  one model serves every base candidate: express the target in the base frame
  instead, `p_b = Rz(−yaw)(p − base)`, `R_b = Rz(−yaw) R`.
- Asset: `newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf"`.
  `add_urdf(..., floating=False, enable_self_collisions=False)`.
- The URDF already has a TCP body, **`fr3_hand_tcp`**. Target it with zero offset
  (no need for the reference example's `0.107` offset on `fr3_hand`).
- Seed from the home pose `[0, −0.785, 0, −2.356, 0, 1.571, 0.785, 0.04, 0.04]`,
  `ik_solver.step(q, q, iterations=300)`. Joint arrays are 2-D: `(n_problems, n_coords)`.
- Fingers: set the last two coordinates to `r`, so the fingers touch the cable.
- **Verify** with `newton.eval_fk` on the IK model: compare `body_q` of
  `fr3_hand_tcp` with the target.

### 5d. IK doesn't know about the table: check it yourself

IK only places the **TCP**. It will happily put the wrist or the hand *through the
table* and still report 0 mm error. So check every solution against the table
with the robot's real geometry:

- For every shape of the IK model (`shape_body`, `shape_transform`, `shape_type`,
  `shape_scale`): take its points in the body frame. A `MESH` gives
  `shape_source[i].vertices × scale`, a `BOX` its 8 corners `±scale`. Then move
  them by `shape_transform`.
- After `eval_fk`, transform them by each body's pose and take the lowest `z`.
- **Leave out `base` and `fr3_link0`.** They stand on the table, so they always read ~0.
- Clearance = `base_z + lowest z − ground_z`.

Then search: every base distance × both flips (8 poses). Keep those that reach
(< 5 mm, < 2°), and of those the one with the **most clearance**. Warn if even the
best one is below the table.

This search only places the visual robot. It never changes the cable.

Then in the main builder: `add_urdf` with the chosen base, write the joint angles
into `builder.joint_q[:9]` (and `joint_target_q`), and zero the mass of every
Franka body (as in 4c). In a sweep, do the search **once**: the grip is the
same for every stiffness.

**Check (test scan, `across`):** IK 0.0 mm / 0.0°, 2 of 8 poses reach, lowest
point +141 mm above the table (`fr3_link1`).
**Check the trap:** a grip 15 cm above the table with the cable leaving **upward
at 45°**, grasp `along`: every pose that reaches puts `fr3_link6` ~77 mm under the
table, and IK still says 0.0 mm. With `across`: +130 mm.

---

## Stage 6. The start pose: the key trick

You want the cable to **start** on the scanned curve but **want to be straight**
(a real cable's rest shape). Newton separates the two:

- **Rest shape** = the pose at `finalize()`. `SolverVBD` computes each rod joint's
  rest bend and twist from `model.body_q` when the solver is created. Your rod was
  created straight (or curled, 4f), so that's what it wants to be.
- **Start pose** = the `State`. Overwrite `state_0.body_q` and `state_1.body_q`
  for the cable bodies after `finalize()`, and set their `body_qd` to zero.

Segment poses for `body_frame_origin="com"`:

```
position_i = ½ (node_i + node_{i+1})
rotation_i = newton.Rod(nodes).quaternions[i]     (local +Z along the segment)
```

Letting `newton.Rod` compute the frames gives you **parallel transport**: frames
that follow the curve without twisting around it. Your own "rotate +Z onto the
direction" would add random twist, which the twist stiffness would then fight.

Order of calls that is safe:

```
state = model.state()
newton.eval_fk(model, model.joint_q, model.joint_qd, state)   # places the Franka;
                                                              # never touches rod bodies
overwrite the cable bodies' body_q / body_qd in BOTH states
create SolverVBD                                              # its first step takes the
                                                              # velocity history from the state
```

Read the poses back into nodes (you'll need this for output too):

```
z_i     = local +Z of quaternion i            (rotate (0,0,1) by q)
start_i = p_i − ½h z_i,   end_i = p_i + ½h z_i
node_0 = start_0,   node_k = ½(end_{k−1} + start_k),   node_N = end_{N−1}
```

**Check:** before any step, the nodes read back equal the planned nodes to within
a few µm (3 µm on the test scan). It isn't exactly zero: on a bend, a straight
segment (chord) is slightly shorter than the arc it replaces (9.975 mm instead of
10 mm in the 70 mm loop). Each segment's +Z must point along its chord to within
~0.03°.
**Check that the rest shape is straight:** run the test scan. The 70 mm loop on the
table must partly open up as the cable relaxes. If it stays perfectly still, your
rest shape is the curve (you created the rod on the curve instead of straight).

---

## Stage 7. Contacts

Look up: `newton.CollisionPipeline(model, broad_phase="explicit", shape_pairs_filtered=...)`,
`model.shape_contact_pairs`, `model.shape_body`.

Start from `model.shape_contact_pairs` (Newton already removed neighbours joined by
a rod joint) and keep:

- cable ↔ table,
- cable ↔ cable **more than 3 segments apart** (the scanned cable crosses itself;
  closer pairs just touch through the bends),
- **nothing with the Franka.** It's kinematic, and its fingers overlap the clamped
  cable on purpose.

Pass the result as a `wp.array` of `wp.vec2i`.

**Check (test scan, 108 segments):** 5568 pairs = all pairs 108·107/2 = 5778,
minus 107 + 106 + 105 close pairs, plus 108 cable–table pairs.

---

## Stage 8. Solve and settle

```
solver = SolverVBD(model, iterations=20, friction_epsilon=1e-4,
                   rigid_compliant_alm=True, rigid_contact_history=False)
dt = (1/60) / 20                     (20 substeps: see 4e)
each substep:  state_0.clear_forces(); pipeline.collide(state_0, contacts)
               solver.step(state_0, state_1, control, contacts, dt); swap
```

Two traps that cost me a failed test. Learn them here rather than the hard way:

1. **`friction_epsilon`.** VBD smooths friction below this sliding speed. With
   Newton's default `1e-2` m/s, a cable lying on a table **creeps ~1 mm/s forever**.
   Your answer then depends on how long you wait. `1e-4` makes it stick. More
   iterations or more damping do **not** fix it.
2. **Don't detect "settled" from speed.** At rest, VBD body velocities still show
   1–5 mm/s of iteration noise, jumping between random segments. Instead, every
   0.5 s read the nodes and compare with 0.5 s ago:

```
drift = max_i ‖nodes_now[i] − nodes_before[i]‖
settled when drift < 0.1 mm   (and t ≥ 1 s)
```

Also stop on NaN ("blew up": more substeps, or softer contact `ke`).

**Check (test scan, init = scan, max_time_s 20):** drift per 0.5 s goes roughly
`117 → 22 → 4.8 → 1.9 → 1.9 → 1.3 → 0.7 → 0.6 → 0.03 mm` and it settles at
**t = 4.5 s** (the stiff cable springs the hand-drawn loop open first). Stretch
(arc length / planned length − 1): **+0.006 %**.

A third trap: if two strands of the **start** shape overlap (closer than one
diameter), self-contact pushes them apart and you get bursts of motion that may
never settle. Check the start shape for it, and print where.
With `friction_epsilon = 1e-2` instead, it never gets below ~1 mm per 0.5 s:
see it once, so you recognise it.

---

## Stage 9. Outputs

Per run, in `<output_dir>/<init>/bend_x<scale>/`:

- `sim_centerline.csv`: `s_m, x_m, y_m, z_m, part` (0 grip, 1 scanned, 2 plug),
  same `s` convention as Stage 3. Also write the **starting** centreline the same
  way, so the comparison can say how far it moved.
- `meta.json`: every number used (segments, `h`, EI, mass, IK error, ground z,
  settled, times, stretch, which end was the gripper). The comparison reads it, so
  the two scripts never disagree about the setup.
- `sim_centerline.ply`: the final centreline turned **back into the scan's frame**
  (`nodes_world @ R`) and the scan PLY's units (`summary["ply_units"]`), so it
  overlays the scan in CloudCompare. An ASCII PLY is ten lines of `f.write`.
- All CSVs stay in the turned (world) frame. Say so in `meta.json`.
- `as_scan/centerline.csv` + `summary.json`: the scanned stretch only
  (`0 ≤ s ≤ L_scan`), in the tube-fit format, for the self-test (Stage 12).

---

## Stage 10. Compare with the scan (no Newton needed)

Load the scan again, turn it with `scan_to_world` from `meta.json`, and orient it
with the `gripper_end` from `meta.json`. Then scan and simulation are in the same
frame. Use only nodes with `supported = 1`.

### 10a. Point-to-polyline distance

For a point `p` and a segment `[a, b]`:

```
t = clamp( (p − a)·(b − a) / ‖b − a‖², 0, 1 )
d = ‖p − (a + t (b − a))‖
```

Distance to the polyline = minimum over its segments. Vectorise it with numpy
broadcasting in chunks of ~256 points (`(chunk, n_segments, 3)` arrays).

### 10b. The numbers

| name | definition | why |
|---|---|---|
| shape | each scan node → nearest point of the sim curve | main number. Doesn't care if the cable slid along itself. |
| hanging / lying | `shape` split by `z < ground_z + r + 3 mm` | hanging = stiffness + weight; lying = friction + history |
| arc | `‖sim(s) − scan(s)‖` at the same `s` (`np.interp` per coordinate) | also sees sliding and length errors |
| plug end | arc distance at `s = L_scan` | one intuitive number |
| moved | `max ‖final − start‖` over all sim nodes | how far reality is from an equilibrium of the model |
| Hausdorff | max of (scan→sim, sim→scan) | worst case. Report, don't optimise. |

Report mean, RMS, 95th percentile and max, in **mm**.

---

## Stage 11. Plots

- **Overlay:** three panels (top x-y, side x-z, side y-z), equal aspect, scan in
  black on top, sim runs in one blue ramp (light = soft, dark = stiff), the table
  as a thin line, the gripper as a dot. Leave out the stub inside the fingers
  (`s < 0`), or it looks like a spike.
- **Error along the cable:** `shape` vs `s`, with the lying stretch shaded.
- **Sweep:** RMS shape error vs EI on a log axis, three lines (all / hanging /
  lying) with different markers, best EI marked. Put ticks at the swept values.

---

## Stage 12. Self-test: find a stiffness you know

The step that proves the whole chain works.

1. Run the test scan (Stage 0) with the reference JSON and `max_time_s` 20 so it
   settles. Its `as_scan/` folder is now a "real" cable whose stiffness you
   **know** (EI = 5e-3).
2. Point the JSON's `scan` at that folder, set `"up": "z"` (it's already in the
   gravity frame, and this test is about stiffness, not Stage 2b), and run
   `EI × [0.2, 0.5, 1, 2, 4]`.
3. Compare.

**Check:**

| EI | 1e-3 | 2.5e-3 | **5e-3 (true)** | 1e-2 | 2e-2 |
|---|---|---|---|---|---|
| shape RMS [mm] | 11.1 | 3.4 | **1.3** | 4.9 | 10.8 |
| shape max [mm] | 24.0 | 7.1 | **2.3** | 8.4 | 20.5 |
| moved [mm] | 24.1 | 7.1 | **2.5** | 8.5 | 21.3 |

The true value must win, clearly on both sides. About 2.5 mm of `moved` remains
even for the true EI (re-meshing, and the grip and plug rebuilt from the exported
shape). That's the pipeline's noise floor: differences below ~3 mm mean nothing.
If the true EI doesn't win, don't run your real scan yet. Check Stages 3, 4e and 8
first.

This test failed three times before it passed, and each failure found a real bug:
friction creep (Stage 8), the 10× too-soft bending (4e), and the averaged grip
direction (Stage 3). Run it after every change to the solver settings.

---

## Stage 13. Fit the cable to the scan

Once the self-test passes, let a search choose the properties. It's a fit: the scan
picks the values, so test the result on a second scan afterwards.

**What to search:** `log10 EI` (GJ keeps its ratio to EI), and the curl `κ ≥ 0`,
`φ ∈ [0°, 360°)` (periodic; with `κ = 0`, `φ` means nothing, so set it to 0). Keep
the mass fixed at the weighed value: from a still cable only *EI ÷ weight* can be
found, because doubling both gives the same shape.

**Score:** RMS of the Stage 10 *shape* distance, over the whole cable or only its
hanging part. Add 1 mm to runs that didn't settle, and give `inf` to runs that blew
up (catch the exception: a bad candidate must not stop the search).

**Search:**
1. **Coarse grid**, all in parallel: EI geometric from EI₀/8 to EI₀·8 × κ ∈ {0, 3, 6, 9}
   × 6 directions.
2. **Pattern search** from the best: for each parameter try ± one step (plus the
   κ–φ diagonals), all in parallel. Move if the best is better by more than 0.05 mm
   (one run's noise), else halve the steps. Start steps: 0.15 decades, 1.5 /m, 30°;
   stop below 0.01, 0.1, 3°.

**Parallel:** a `ProcessPoolExecutor` with the **`spawn`** start method. Never `fork`
a process that may hold CUDA or Warp state. Each worker loads the scene module and
the scan once (initializer). Robot off: it's only visual. Save every result to a CSV
as it arrives, keyed by the rounded parameters, so a rerun skips what's done.

**Check (known answer):** make a truth with the reference JSON, but EI = 3e-3,
κ = 6 /m, φ = 60°. Settle it, restart it once from its own result (so its twist is at
equilibrium, 4g), and fit that, starting from EI = 5e-3, κ = 0, with grid directions
0/90/180/270°. After one refinement round I get EI 3.5e-3, κ 6.5, φ 60° at 2.05 mm,
where the truth itself scores 2.0 mm. Full table: `04_pointcloud_vs_sim/README.md`
(*Fitting*).

**Smoke test** (does the script run end to end?): 1 s simulations,
`--grid-ei 2 --grid-curl 0,5 --grid-dir 2 --max-rounds 1`. That's 16 runs in ~2 min on
4 cores, and all outputs appear. Run it again: it must say `resuming: 16 runs`.

---

## When something goes wrong

| symptom | likely cause | fix |
|---|---|---|
| `body_color_groups is empty` error | forgot `builder.color()` | call it before `finalize()` |
| cable ~50 % too heavy | capsule caps | Stage 4b density correction |
| cable never settles, drifts ~1 mm/s on the table | friction smoothing | `friction_epsilon = 1e-4` |
| "settled" never triggers though it looks still | speed-based criterion | shape-drift criterion (Stage 8) |
| loop on the table stays perfectly still at any EI | rest shape = scanned curve | create the rod straight; only the *state* follows the scan |
| cable twists/corkscrews at the start | your own frames add twist | frames from `newton.Rod(nodes).quaternions` |
| red scan line stands up / hand on the floor / cable collapses at once | scan not z-up | Stage 2b; or set `scan.up` |
| clamped end jumps away from the fingers | robot–cable contact pairs | exclude all Franka pairs |
| result changes when you change segment length | per-joint stiffness passed directly | use rigidities on the `Rod` |
| robot in a strange pose / IK error of cm | base out of reach | set `robot.base_xyz_m`. The cable is unaffected. |
| hand or wrist under the table, IK says 0 mm | IK ignores the table; or the wrong `grasp` | 5d clearance search; match `robot.grasp` to your scan |
| hand rotated 90° from expected | quaternion order | `(x, y, z, w)` in Newton/Warp |
| NaN | contact too stiff for the time step | more substeps or lower `contact.ke` |
| cable floppy like jelly, sags far more than its EI | EA / kGA far above bending; too few substeps | Stage 4e: `check_stiffness.py`; EA ~200, kGA ~20, 20 substeps |
| bursts of motion that never settle (self-contact on) | strands overlap in the start shape | crossing fix, Stage 3b |
| no EI makes the loop / hanging part match; error the same everywhere | the cable's natural curl | rest curvature, Stage 4f, and fit it (Stage 13) |
| true EI doesn't win the self-test; hanging part off everywhere | grip direction estimated by averaging | use the tube fit's tangents (Stage 3) |
| up tilted by 1–2° on a stiff cable | it rests on the table along a line only | Stage 2b limit: set `scan.up` from the table |

---

## Look up

- `help(newton.Rod)`, `help(newton.ModelBuilder.add_rod)`: rigidities vs per-joint stiffness, `body_frame_origin`
- `help(newton.solvers.SolverVBD)`: `iterations`, `friction_epsilon`, `rigid_compliant_alm`
- `help(newton.eval_fk)`: note "ROD body transforms are not changed"
- `newton.ik`: `IKSolver`, the objectives, `IKJacobianType`
- `newton/examples/cable/example_cable_twist.py`: kinematic first segment, the same zero-mass trick
- parallel transport frames on a curve; quaternion from rotation matrix
- capsule volume; point-to-segment distance; Hausdorff distance

---

## Is the stiffness right? Look at the hanging part

The first Ethernet run used `EI = 1e-3`, and, as Stage 4e found later, the solver
made it bend ~10× more easily than even that. In the viewer the simulated cable (blue)
dropped **straight down** from the fingers, while the scan (red) **bows out
sideways** before reaching the table. A cable can only hold that bow if it is
stiff enough. So the hanging part tells you at a glance that the simulation is
too soft. The lying part can't tell you: friction holds it wherever it started.

Two honest ways to get `EI`, from least to most fitting:

1. **Measure it, then compare (validation, no fitting).** Table-edge test with
   your cable: clamp it so `L` sticks out horizontally, measure the tip drop `δ`,
   `EI = w L⁴ / (8 δ)` with `w` = weight per metre × 9.81 (~0.44 N/m). Keep
   `δ < 0.15 L` (small deflection), and repeat for `L` = 10, 15, 20 cm: the three
   values must agree. Put the result in the JSON and run once. The `hanging` error
   then says how good the *model* is.
2. **Sweep and pick (identification).** `--sweep` runs 5 complete simulations
   (1e-3 to 2e-2) and the comparison reports which one is closest. That uses the
   scan to *choose* `EI`, so you then need a second scan, in another pose, to test it.

If even the stiffest value can't make the hanging part bow like the scan, the
bow isn't elasticity. Solid-copper Ethernet keeps a **permanent curl** from the
spool (plastic memory), and a straight rest shape can never reproduce that. Lay
the cable loose on the table: if it doesn't lie straight, its rest shape isn't
straight either.

