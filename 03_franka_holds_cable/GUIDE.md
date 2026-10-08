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
CPU. Work in **metres** (the tube fit's CSV is in metres).

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
          x = 0.72 + 0.07 cos θ,   y = 0.07 + 0.07 sin θ,   z = r
```

Stack the three parts, resample to **151 nodes at equal arc length**, and write:

- `centerline.csv` with header `s_m,x_m,y_m,z_m,tx,ty,tz,supported`
  (tangents from `np.gradient`, normalised; `supported` all 1),
- `summary.json` with `{"radius_mm": 3.6, "ply_units": "mm"}`.

**Check:** length 1019.5 mm, 151 nodes. Plotted from the top, a straight line
from the gripper into a loop of 70 mm radius.

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
never looks at the simulation:

1. **The cable never climbs back up.** Hanging from the gripper and lying on the
   table, its height only goes *down* along the cable (except a few mm where one
   strand crosses another). For a direction `d`, walk from the higher end and add
   up every increase in height `X·d`:

   ```
   rise(d) = Σ max(0, h[k+1] − h[k]),   h = X·d, walked from the higher end
   ```

   Evaluate it for ~20 000 directions spread over the sphere (Fibonacci sphere),
   take the smallest, then refine within ±2.5° at 0.2° steps. Vectorise:
   `H = X @ D.T` is (nodes × directions).
2. **Up or down?** `d` and `−d` score the same: a cable "hanging upward" from the
   floor is monotone too. The table tells them apart: the stretch lying on it is
   **many nodes at the same, lowest height**, while the gripper is a single high
   point. Count nodes within 5 mm of the bottom and of the top, and flip `d` if the
   top has more.
3. **Polish with the table plane:** fit a plane (SVD) through the nodes within 5 mm
   of the bottom, keep only those within 1 mm of it, and fit again (this drops the
   nodes where the cable lifts off the table). Use the plane's normal if the nodes
   spread in two directions (a straight line has no plane) and it's within 5° of `d`.

Then build the rotation that takes `up` to `+z` (Rodrigues:
`R = I + K + K²/(1 + up·z)`, `K` = cross-product matrix of `up × z`; for `up = −z`
use a half turn about x). Keep `R` in `meta.json` so the comparison and the
`.ply` output can use it.

**Check:** take the test scan, turn it into several frames yourself (y-up, a
camera's −y-up, tilted 10° and 25°, a few random rotations, plus an offset). The
detected up must match the true one to ~0.01°, and the turned scan must have a
height span of 346 mm every time. Before the plane polish you'll see ~0.14°.
Without step 2, the −y-up case comes out exactly 180° wrong.

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

**End directions.** One segment's direction is noisy. Average over ~2 cm:
`t₀ = unit(X[k] − X[0])` with `k` the first node past 2 cm, and the same at the
plug end (`t₁`).

**Starting centreline** (`init = "scan"`): a dense polyline

```
[ X[0] − grip·t₀ ,  X[0], X[1], …, X[−1] ,  X[−1] + plug·t₁ ]
```

resampled to `n_seg + 1` points at equal arc length (`np.interp` on cumulative
length, per coordinate).

**Check (test scan):** `h = 10.0` mm, 108 segments, 2 grip segments, 4 plug
segments, plug 40.5 mm, total length 1.080 m.
**Check (your scan):** 95 segments, plug ≈ 37 mm.

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
`[2.0e6, 7.0e5, 0.1, 0.077]`, i.e. `EA/h, kGA/h, EI/h, GJ/h`.

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
4c) = `μ·(L_scan + plug) + plug_mass`. Test scan: 51.7 g. Without the
correction you'd get 74.6 g.

### 4c. Damping, grip, plug

- `bend_damping = τ · EI / h`, `twist_damping = τ · GJ / h` (τ = 2 ms),
  `stretch_damping = 0`. Damping changes how fast it settles, not where.
- **Grip:** the first `n_grip` bodies get zero `body_mass`, `body_inv_mass`,
  `body_inertia`, `body_inv_inertia`. VBD treats `inv_mass = 0` as kinematic and
  never moves them. This is the boundary condition.
- **Plug:** spread `plug_mass` over the last `n_plug` bodies. For each one,
  multiply mass **and inertia** by `f = (m + Δm)/m` and set the inverses to match.
  (`builder.body_inertia[b]` is a `wp.mat33`. Go through numpy:
  `np.array(I).reshape(3, 3)`.)

### 4d. Table and finalize

```
builder.add_ground_plane(height=ground_z, cfg=ShapeConfig(ke, kd, mu, ...))
builder.color()                # VBD refuses to run without colouring
model = builder.finalize()
```

Call `SolverVBD.register_custom_attributes(builder)` right after creating the
builder, before adding anything.

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
  created straight, so its rest shape is straight.
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
dt = (1/60) / 10
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

**Check (test scan, init = scan):** drift per 0.5 s goes roughly
`75 → 18 → 13 → 10 → 7 → 5 → 3.5 → 2.3 → 1.5 → 0.9 → 0.6 → 0.4 → 0.3 → 0.05 mm`
and it settles at **t ≈ 7 s** (~95 s wall on a 4-core CPU). Stretch
(arc length / planned length − 1): **+0.07 %**.
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

1. Run the test scan with `EI = 1e-3` and a long `max_time_s` (20 s) so it settles.
   Its `as_scan/` folder is now a "real" cable whose stiffness you **know**.
2. Point the JSON's `scan` at that folder and run `EI × [0.25, 0.5, 1, 2, 4]`.
3. Compare.

**Check:**

| EI scale | 0.25× | 0.5× | **1×** | 2× | 4× |
|---|---|---|---|---|---|
| shape RMS [mm] | 0.7 | 0.4 | **0.4** | 1.2 | 4.4 |
| shape max [mm] | 2.5 | 1.3 | **0.9** | 3.7 | 13.0 |
| moved [mm] | 2.4 | 1.5 | **1.0** | 3.6 | 13.0 |

The true value must win. About 1 mm of `moved` remains even for the true EI
(re-meshing). That's your pipeline's noise floor. If 1× doesn't win, don't run
your real scan yet: look at Stage 8 first.

The curve is sharp on the stiff side and shallow on the soft side, because this
test cable leaves the gripper almost straight down. That pose limits EI from
above much better than from below.

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

---

## Look up

- `help(newton.Rod)`, `help(newton.ModelBuilder.add_rod)`: rigidities vs per-joint stiffness, `body_frame_origin`
- `help(newton.solvers.SolverVBD)`: `iterations`, `friction_epsilon`, `rigid_compliant_alm`
- `help(newton.eval_fk)`: note "ROD body transforms are not changed"
- `newton.ik`: `IKSolver`, the objectives, `IKJacobianType`
- `newton/examples/cable/example_cable_twist.py`: kinematic first segment, the same zero-mass trick
- parallel transport frames on a curve; quaternion from rotation matrix
- capsule volume; point-to-segment distance; Hausdorff distance
