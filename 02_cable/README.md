# Part 2 — The cable alone in Newton

> **Goal of this part:** create a cable, choose its **size**, **number of
> segments**, **mass** and **elasticity**, fix one end, and read its shape
> back as numbers. No robot yet.
>
> Reference: [`reference/example_franka_cable_ik_pick_place.py`](../reference/example_franka_cable_ik_pick_place.py)
> (`L231–L267`) and the legacy file `legacy/methods/hang_newton_cable.py`
> (fixing ends, reading nodes). Newton does the physics. Your job is to give it
> the right inputs and check the outputs.

---

## 2.1 What Newton builds when you "add a cable"

`builder.add_rod(...)` creates a **chain of capsule rigid bodies**: one capsule per
segment. Neighbours are connected by **cable joints**, each with a stretch spring
and a bend/twist spring. Solved by **`SolverVBD`**.

```
 node0    node1    node2           nodeN
   ●━━━━━━━━●━━━━━━━━●━━━ ... ━━━━━━●
   └ body0 ┘└ body1 ┘                   N segments = N capsule bodies, N+1 nodes
```

So the cable is just **bodies** in `state.body_q`, like robot links. You read
and move them the same way.

---

## 2.2 Creating the cable: two calls

**Step 1 — the geometry** (`L243–L250`):

```python
rod = newton.Rod.create_straight(
    start=wp.vec3(x0, y0, z0),       # first node, world frame [m]
    direction=wp.vec3(1, 0, 0),      # unit vector the cable points along
    length=0.38,                     # total length [m]
    segment_count=19,                # number of capsules
    twist_total=0.0,                 # built-in twist over the whole length [rad]
    radius=0.005,                    # cable radius [m] (half the diameter)
)
```

The example centres it: `start = CENTER − ½·length·direction`.

**Step 2 — add it to the model with material properties** (`L235–L262`):

```python
cable_cfg = newton.ModelBuilder.ShapeConfig(
    density=100.0,      # kg/m³ → sets the mass
    ke=1.0e3,           # contact stiffness
    kd=1.0e-1,          # contact damping
    mu=1.0,             # friction
    margin=0.0,
    gap=0.01,           # contact detection distance [m]
)
builder.add_rod(
    rod=rod,
    body_frame_origin="com",          # each body's frame sits at its capsule centre
    cfg=cable_cfg,
    stretch_stiffness=1.0e2,
    stretch_damping=1.0e-1,
    bend_stiffness=4.0e-4,
    bend_damping=2.0e-3 * 4.0e-4,
    label="vbd_cable",
)
```

Before `add_rod`, call `SolverVBD.register_custom_attributes(builder)` (`L161`).
Before `finalize()`, call **`builder.color()`** (`L194`). VBD updates bodies in
parallel groups ("colours") and needs this.

Remember the body/joint/shape index ranges, as for the Franka (`L232–L265`):

```python
cable_body_start = builder.body_count
... add_rod ...
cable_bodies = list(range(cable_body_start, builder.body_count))
```

Also read the docstrings in your Newton version: `help(newton.Rod.create_straight)`,
`help(newton.ModelBuilder.add_rod)`. Look for extra options (e.g. separate
twist stiffness, curved rods, closed loops).

---

## 2.3 Choosing the SIZE: `radius` and `length`

- **`length`**: the real cable's length (or, for a held cable, the **free
  length** from the fingers to the tip). Measure it with a tape.
- **`radius`**: half of the diameter measured with calipers. The radius does
  three things in Newton:
  1. **collision thickness**: the cable's contact surface (with floor, fingers);
  2. **mass**, through volume × density (§2.5);
  3. what you **see** in the viewer.
  
  For a thin cable gripped by fingers, an accurate radius matters: the fingers
  close onto it (the example's gripper comments `L55–L61` are all about this).

---

## 2.4 Choosing the NUMBER OF SEGMENTS

`segment_count = N` → segment length `l = length / N`. The example uses
19 segments of 2 cm, with radius 5 mm (so `l = 4r`).

| consideration | rule of thumb |
|---|---|
| **Shape resolution** | the smallest bend you want to see (e.g. the curve near the gripper, a few cm) should span ≥ 3–5 segments |
| **Capsule proportions** | keep `l ≳ 2r`. Shorter segments are basically spheres that overlap their neighbours heavily. |
| **Gripping by contact** | ≥ 1–2 segments under a finger pad (~2 cm), so the fingers have something to pinch. 2 cm segments work in the example. |
| **Cost** | each iteration costs ∝ N, and longer chains need **more iterations** to converge. Doubling N costs more than 2× in practice. |
| **Mac (CPU)** | 20–60 segments is comfortable |

**How to decide for real (do this!):** the convergence test.
1. Simulate your test case (e.g. cable hanging from a fixed point, §2.8) with N.
2. Repeat with 2N (and more iterations if needed).
3. Compare the final shapes (tip position, max distance between curves).
4. If they differ by less than your tolerance (say 1–2 mm), N is enough. Otherwise
   double again.

**Important:** when you change N, check whether the stiffness values
(`stretch_stiffness`, `bend_stiffness`) you pass are **per joint** or
**per unit length** in your Newton version (read the `add_rod` docstring). If
they're per joint, the same numbers with more segments give a *different
cable*. The legacy code (Newton 1.2) scaled them as `E·A/l` and `E·I/l` per
segment (see `rod_stiffness` in `legacy/methods/cable_config.py`). The
convergence test above reveals it too: if the shape keeps changing with N, the
stiffness isn't being scaled.

---

## 2.5 Choosing the MASS: `density`

Newton computes each capsule's mass = volume × `density`. So:

```
density = (mass per metre) / (π r²)
```

Measure the real cable: weigh a known length on a kitchen scale.

| example | mass per metre | radius | density to use |
|---|---|---|---|
| the reference example | — | 5 mm | 100 kg/m³ (very light, 3 g total) |
| typical USB/charger cable | ~20 g/m | 2 mm | 0.020 / (π·0.002²) ≈ 1600 kg/m³ |
| your cable | weigh it | calipers | compute |

**Check it:** after `finalize()`, sum `model.body_mass` over `cable_bodies` and
compare with (mass per metre × length). Capsule end caps can make it slightly
different. Correct the density if needed.

The cable's mass matters a lot: **the shape of a held cable is a balance between
its weight and its bending stiffness**. Wrong mass = wrong shape.

---

## 2.6 Choosing the ELASTICITY: `stretch_stiffness` and `bend_stiffness`

| parameter | controls | what you see if too low | too high |
|---|---|---|---|
| `stretch_stiffness` | how much the cable **elongates** | cable visibly stretches like rubber when lifted | needs more VBD iterations, otherwise jitter |
| `bend_stiffness` | how **stiff** it is to bend | droops like wet string | sticks out like a rod |
| `stretch_damping` | how fast stretch oscillations die | bouncy | sluggish |
| `bend_damping` | how fast bending wiggles die | keeps swinging/wiggling | slow-motion movement |

**Stretch:** a real cable is practically **inextensible**. Choose
`stretch_stiffness` high enough that the length changes by **< 0.5 %** when the
cable hangs (measure it, §2.9), and no higher than needed (cost).

**Bend:** this is **the** parameter that makes your cable look like *your* cable.
The example's `4e-4` was tuned for IsaacLab's task, not measured. For your cable:
- **starting value**: measure the real cable's bending stiffness `EI` with a
  table-edge test (let a length `L` stick out horizontally over a table edge,
  measure the tip drop `δ`, then `EI ≈ w·L⁴ / (8·δ)` with `w` = weight per metre in
  N/m; use a short overhang so the drop is small). Convert to Newton's
  `bend_stiffness` following the docstring (e.g. `EI / l` if per joint);
- **final value**: identified by comparing with the point cloud (Part 4).

**Damping as a ratio:** the example writes `bend_damping = 2e-3 × bend_stiffness`.
Keeping that ratio when you change the stiffness keeps the "feel" the same. Damping
should **not** change the final resting shape, only how fast it's reached.
Verify that once.

**Twist:** `twist_total` builds initial twist into the cable. Leave it 0 unless
your real cable is twisted.

**Rest shape = creation shape.** `create_straight` makes a cable that *wants* to
be straight. Real cables often keep a curl from being coiled. Remember this when
comparing with reality.

---

## 2.7 The cable solver alone

```python
solver = SolverVBD(model, iterations=20, rigid_compliant_alm=True, rigid_contact_history=False)
```

(Settings from `L316–L321`.)

| knob | effect |
|---|---|
| `iterations` | more = cable closer to the true solution (less stretch); cost ∝ iterations |
| `substeps` | more = smaller time step = more accurate and stable; cost ∝ substeps |

The example uses 20 iterations × 10 substeps for *motion*. The legacy study,
for a static shape at high accuracy (150 segments), needed ~200 iterations × 4
substeps. **Accuracy depends on `substeps × iterations`**, and fewer substeps
with more iterations was cheaper. Run your own small test (§2.9).

Collisions for the cable alone:

```python
collision_pipeline = newton.CollisionPipeline(model)      # default: all pairs
contacts = collision_pipeline.contacts()
```

and in the loop call `collision_pipeline.collide(state_0, contacts)` before
`solver.step`. Self-collision of the cable is usually not needed (slow).
Check how to disable it in your version (collision group / filtered pairs, §3.4).

---

## 2.8 Fixing an end, and moving it

**Fix** (legacy method): make the first body immovable by giving it zero mass
and inertia, **after** `add_rod`, **before** `finalize()`:

```python
b = cable_bodies[0]
builder.body_mass[b] = 0.0
builder.body_inv_mass[b] = 0.0
builder.body_inertia[b] = wp.mat33(0.0)
builder.body_inv_inertia[b] = wp.mat33(0.0)
```

A zero-mass body is held in **position and orientation**, like a cable
clamped in a gripper. Fixing 2 bodies = a longer clamp (like a finger pad holding
2 cm of cable).

**Move it (kinematic attachment):** each substep, write the fixed body's pose
(and velocity) into the state before `solver.step`:

```
body_q[b]  = desired pose (position + quaternion)
body_qd[b] = (pose change) / sim_dt          ← consistent velocity, no "teleport"
```

That's the simplest way to attach the cable to a robot hand (Part 3, option K).
Interpolate the pose smoothly across substeps. A body that jumps injects
energy and makes the cable jitter.

---

## 2.9 Reading the cable back (your measurement)

```python
q = state_0.body_q.numpy()[cable_bodies]      # (N, 7): px, py, pz, qx, qy, qz, qw
```

With `body_frame_origin="com"`, `q[:, :3]` are the **segment midpoints**. To get
the N+1 **nodes**: each node is midpoint ± ½·l along the capsule's axis (which
local axis that is, +Z in the legacy code, is shown in the legacy
`read_nodes` function). Rotate that axis by the body quaternion.

Useful numbers to compute from the nodes:
- **arc length** = sum of distances between consecutive nodes → compare with
  `length` → **stretch %** (should be < 0.5 % at rest),
- **tip position** (last node),
- **settled?** Tip moved < 0.1 mm over the last 0.1 s, for 1 s,
- save `nodes` as `.npy` / `.csv` for Part 4.

---

## Tasks

1. **Cable on the floor.** Floor + straight cable lying on it (like the example's
   starting scene). It must lie still, not bounce or drift.
2. **Hanging from a fixed point.** Fix body 0 in mid-air, cable starting
   horizontal. Watch it fall and settle. Print stretch % and the tip position.
3. **Mass check.** Sum the cable body masses and compare with your target.
4. **Segments.** Task 2 with N = 10, 20, 40, 80. Plot the tip position vs N.
   Choose N (§2.4).
5. **Iterations.** Task 2 with 5, 10, 20, 50, 100 iterations. Plot stretch % vs
   iterations. Choose a value.
6. **Bend stiffness sweep.** Task 2 with `bend_stiffness` × 0.1, 1, 10, 100.
   Plot all final shapes in one figure. This shows you how much the parameter
   matters for a cable held at one end.
7. **Damping check.** Change `bend_damping` 10×. The final shape must not change.
8. **Your real cable.** Measure diameter, mass per metre and EI (table-edge test).
   Put them in a config file and simulate the same table-edge test in Newton.
   Does the tip drop match?
9. **Kinematic end.** Move the fixed end in a circle and watch the cable follow
   smoothly (no jitter).

## Check yourself

- What does `radius` change besides the look?
- How do you compute `density` from a weighed cable?
- How do you decide the number of segments, concretely?
- What symptom tells you there are too few VBD iterations?
- Why must damping not change the final shape?
- What does `body_frame_origin="com"` mean for reading positions?

## Common mistakes

- Forgetting `builder.color()`: VBD fails or behaves strangely.
- Forgetting `register_custom_attributes` before adding things.
- Changing segment count without understanding whether stiffness is per joint.
- Too few iterations: the cable stretches, and you misread it as "too soft".
- Reading `body_q` positions as nodes (they're midpoints with `"com"`).
- Comparing shapes before the cable has settled.
