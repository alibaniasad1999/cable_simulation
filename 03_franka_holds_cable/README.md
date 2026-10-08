# Part 3 — The Franka holds the cable

> **Goal of this part:** put Part 1 (robot) and Part 2 (cable) in one scene, and
> make the gripper hold the cable: first in the simplest robust way, then the
> full pick-and-place way the reference example does it.
>
> Reference: [`reference/example_franka_cable_ik_pick_place.py`](../reference/example_franka_cable_ik_pick_place.py).

---

## 3.1 The core problem: two solvers

- The **robot** is best simulated by `SolverMuJoCo` (articulations, PD joints).
- The **cable** is best simulated by `SolverVBD` (stiff rods).

They're different solvers, but the gripper and the cable must interact. There are
two ways to deal with it:

| | **Option K — kinematic attachment** | **Option C — coupled solvers (the example)** |
|---|---|---|
| idea | cable's first segment(s) are glued to the TCP; you set their pose every substep | gripper pinches the cable through real contact and friction |
| solvers | VBD for the cable only (+ optionally MuJoCo for the arm) | `SolverCoupledProxy` = MuJoCo + VBD, gripper bodies shared as "proxies" |
| grasp | exact and repeatable; you choose exactly where and at what angle | emerges from contact; can slip, rotate, depend on finger force |
| difficulty | easy | harder (contact tuning, finger force, coupling settings) |
| best for | **comparing with the point cloud** (Part 4): you control the boundary condition | pick-and-place, studying grasping/slipping |

**Recommendation:** build **K first**, use it for Part 4. Then build **C** to
understand and reproduce the example.

---

## 3.2 Scene assembly (common to both)

Order inside the builder:

```
builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
SolverMuJoCo.register_custom_attributes(builder)      # if using MuJoCo
SolverVBD.register_custom_attributes(builder)
1. Franka   (add_urdf, gains, gravcomp)               → franka_bodies / joints / shapes
2. Cable    (Rod.create_straight + add_rod)           → cable_bodies / joints / shapes
3. Floor    (add_ground_plane)                        → ground_shapes
builder.color()
model = builder.finalize()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)   (and state_1)
```

The Franka must come first (IK coordinates line up, Part 1 §1.3). Keep the
three index lists. Every later step uses them.

Where to create the cable:
- **K**: create it **already in the gripper**. Compute the TCP pose from FK of the
  start configuration, and start the rod at the TCP, pointing along your grasp
  direction.
- **C**: create it **lying on the table**, where the robot will pick it (the example:
  `CABLE_CENTER = (0.5, 0, 0.256)`, along x, `L47`).

---

## 3.3 Option K — kinematic attachment step by step

1. **Define the grasp.** Which cable body sits between the fingers
   (`grasp_body`: 0 = held at one end, `N//2` = held in the middle), and the
   transform `T_tcp_grasp`, i.e. how the cable is oriented between the fingers.
   For a Franka hand pointing down, the fingers close along the hand's y axis, so a
   cable between them runs along the hand's **x** axis (or along its z, if it
   points out of the hand). Put this in a config file. In Part 4 it must match the
   real photo.
2. **Clamp it.** Zero mass/inertia for `grasp_body` (and one neighbour for a
   longer clamp, like a 2 cm finger pad), as in Part 2 §2.8.
3. **Robot pose.** Either kinematic (IK/recorded joints → `joint_q` → `eval_fk`)
   or MuJoCo-driven. Get `T_world_tcp` = hand body pose × TCP offset
   (0.107 m along the hand z).
4. **Every substep**, before `solver.step`:
   `T_world_grasp = T_world_tcp · T_tcp_grasp` → write into `state.body_q[grasp_body]`,
   and a matching velocity into `body_qd`.
5. **Solver**: `SolverVBD` steps the cable. With a kinematic robot, nothing else
   needs stepping. Just call `eval_fk` when the joints change.
6. **Collisions**: cable–floor on. Cable–robot links: optional (needed if the cable
   drapes over the hand/arm). The clamped bodies are inside the fingers, so
   exclude cable–finger pairs or the fingers will push the cable out.

**Check:** hand still → cable hangs and settles. Rotate the hand 90° slowly →
the cable follows, no jitter, stretch < 0.5 %.

---

## 3.4 Option C — the example's coupled solver, piece by piece

### a) The coupled solver (`L293–L342`)

```python
solver = SolverCoupledProxy(
    model=model,
    entries=[
        SolverCoupled.Entry(name="mjc", solver=lambda v: SolverMuJoCo(model=v, ...),
                            bodies=franka_bodies, joints=franka_joints),
        SolverCoupled.Entry(name="vbd", solver=lambda v: SolverVBD(model=v, iterations=20, ...),
                            bodies=cable_bodies, joints=cable_joints),
    ],
    coupling=SolverCoupledProxy.Config(
        proxies=[SolverCoupledProxy.Proxy(
            source="mjc", destination="vbd",
            bodies=gripper_bodies,          # hand + fingers
            mass_scale=1.0,
            mode="lagged",                  # or "staggered"
            collision_pipeline=lambda m: newton.examples.create_collision_pipeline(m, broad_phase="explicit"),
            collide_interval=1,
        )],
        iterations=1,
    ),
)
```

Read it as: *"MuJoCo owns the Franka, VBD owns the cable. The gripper bodies are
copied into VBD as **proxies** (moving obstacles with mass), so the cable feels
the fingers."* Imported from `newton.solvers.experimental.coupled`. Note the word
**experimental**: this API may change.

| setting | meaning | try |
|---|---|---|
| `bodies` (proxy) | which robot bodies the cable can touch | hand + both fingers (`L270–L274`) |
| `mass_scale` | how heavy the proxies look to VBD | 1.0. Bigger = fingers less pushed by the cable. |
| `mode` | `lagged`: VBD sees the gripper from the previous substep (simple, stable). `staggered`: solvers take turns within a substep (tighter, costlier). | start `lagged` |
| proxy `iterations` | relaxation passes between the solvers per substep | 1, then 2–3 if the grasp is soft |
| `collide_interval` | how often (in substeps) gripper–cable contacts are recomputed | 1 |

### b) Collisions are split in two (`L118–L124`, `L344–L354`)

- The **main** pipeline handles only **robot↔floor and cable↔floor** pairs. The
  example builds this explicit pair list itself and passes
  `broad_phase="explicit", shape_pairs_filtered=pairs`.
- The **proxy** pipeline (inside the coupled solver) handles **gripper↔cable**.

So no pair is computed twice. Then `solver.prepare_contacts(contacts)` once,
and `collision_pipeline.collide(state_0, contacts)` every substep.

### c) Gripper material = cable material (`L173–L180`)

The finger shapes get the same contact `ke, kd` as the cable, and `mu = 1.0`
(high friction so the pinch holds). Only the gripper shapes are changed, not the
whole robot.

### d) The substep loop (`L476–L493`)

```
IK solve → copy to control.joint_target_q (fingers included)
for substeps:
    clear_forces → collide → solver.step → newton.eval_ik(model, state_1, state_1.joint_q, state_1.joint_qd) → swap
```

`eval_ik` recomputes the joint coordinates from the body poses after the step,
so `joint_q` stays consistent for the next MuJoCo step and for reading.

### e) The finger trap (`L55–L63`)

The fingers are solved in MuJoCo, which **doesn't contain the cable**. They only
"feel" it through the lagged proxy contact. If you command them fully closed with a
strong controller, they can squeeze **through** the thin cable. Knobs:
finger target width (`GRIP_CLOSE`, `GRIP_HOLD`), finger stiffness
(`GRIP_STIFFNESS = 1000`), force limit (`GRIP_FORCE`), contact `ke`, cable
radius, segment length (Part 2 §2.4). Tune them while watching the grasp
closely in the viewer.

### f) The motion: keyframes (`L410–L436`)

Approach above the cable centre → descend so the **TCP is at the cable's centre
height** (`grasp_z = cz`) → close → lift → move → lower → release → retract.
The test at the end (`L509–L524`) checks the grasped segment ended within 1 cm of the
target. Write a similar check for yours.

---

## 3.5 Many copies at once: worlds (useful for Part 4)

The example can simulate `world_count` independent copies in one model
(`L157–L171`, `L278–L288`):

```
template = ModelBuilder(...);   build ONE scene into template
builder  = ModelBuilder(...);   builder.replicate(template, world_count=W)
```

Body index of body `i` in world `w` = `w × bodies_per_world + i`. The example's
`_expand_world_indices` does this for every index list. IK also solves `W`
problems at once (`n_problems=world_count`).

**Why you care:** in Part 4 you'll run the same scene with many different
`bend_stiffness` values. With worlds, that's **one** simulation instead of many.
Per-world parameters can be set by editing the replicated builder's arrays
before `finalize()` (e.g. the cable joint stiffness of world `w`). Check which
arrays hold the joint stiffness in your version.

---

## 3.6 Viewer helpers

The coupled example uses `newton.examples.configure_coupled_view`,
`log_coupled_view` and `apply_coupled_viewer_forces` (they let you drag bodies with
the mouse across both solvers). For option K, use the plain viewer calls:
`viewer.begin_frame(t)`, `viewer.log_state(state)`, `viewer.log_contacts(...)`,
`viewer.end_frame()`. Set the camera with `viewer.set_camera(...)` (`L129–L133`).

---

## Tasks

**Option K**
1. Robot (kinematic) + cable created in the gripper, clamped at body 0. Hand still → cable settles.
2. Grasp in the middle (`grasp_body = N//2`) → two hanging halves.
3. Keyframes: rotate the hand from "pointing down" to "pointing sideways". Watch the cable.
4. Same with MuJoCo driving the robot instead of kinematic. Any difference?

**Option C**
5. Rebuild the example in stages: robot + cable on the table + floor, **no coupling**
   (two separate solvers stepping their own bodies). The fingers pass through the cable.
6. Add `SolverCoupledProxy`. The fingers now push the cable.
7. Add the split collision pipelines and gripper material.
8. Add the keyframes. Pick, lift, place. Add the end test (within 1 cm).
9. Break it on purpose: close the fingers harder, use fewer segments, a thinner
   radius, `staggered` vs `lagged`. Note what fails and why.

**Worlds**
10. Run Option K with `world_count = 4`, each world a different `bend_stiffness`.
    See all four shapes side by side.

## Check yourself

- Why does the example need two solvers, and what does a "proxy" do?
- Why is option K better for comparing against a real point cloud?
- Why does the main collision pipeline exclude gripper–cable pairs?
- What is `eval_ik` for in the substep loop?
- How do you find body `i` of world `w`?

## Common mistakes

- Franka not added first, so the IK coordinates are misaligned.
- Option K: cable–finger collisions left on, so the fingers push the clamped cable away.
- Option K: grasp pose written once per frame instead of every substep (jitter).
- Option C: fingers commanded fully shut with high force, so the cable gets squeezed through.
- Forgetting `prepare_contacts` or computing gripper–cable contacts twice.

---

## The Ethernet scene (working code)

[`ethernet_scene.py`](ethernet_scene.py) is Option K applied to your scanned
Ethernet cable. **To write it yourself from scratch, follow [GUIDE.md](GUIDE.md)**
(stages, Newton calls, and the numbers to expect at each step). Every number comes from [`configs/ethernet_cat6.json`](../configs/ethernet_cat6.json)
(the `_...` keys in that file explain each value).

```bash
python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json               # one run
python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --viewer gl   # watch it
python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --sweep       # all EI values
python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --bend-scale 2
```

Input: the tube-fit output `results/Ethernet_tube_fit/centerline.csv` (+ `summary.json`
for the radius). The scan is taken as **Z-up, in metres**.

### What the script builds, step by step

| step | JSON | Newton call / what happens |
|---|---|---|
| orient the scan | `scan.gripper_end` | node 0 = the gripper end (`auto`: the higher end; the other end lies on the table) |
| table | `scan.ground_z_m` | `add_ground_plane(height=...)`; `null` = lowest scanned centreline point − radius |
| cable length | `cable.grip_length_m`, `cable.plug_length_m` | grip (inside the fingers) + scanned length + plug. Arc length `s = 0` is the first scanned node. |
| segments | `sim.segment_length_m` | adjusted so the grip is a whole number of segments (10 mm → 2 grip segments) |
| elasticity | `cable.*_rigidity_*` | `newton.Rod.create_straight(..., stretch_rigidity=EA, shear_rigidity=kGA, bend_rigidity=EI, twist_rigidity=GJ)`. With rigidities, **Newton divides by the segment length itself** (`EI / h` per joint), so changing the segment length doesn't change the cable. |
| mass | `cable.mass_per_length_kg_m` | `ShapeConfig(density=...)`. Each capsule has two round end caps on top of its length, which adds ~50 % volume at 10 mm segments; the density is corrected so the mass per metre is exact (printed in `meta.json` as `mass_free_cable_kg`). |
| plug | `cable.plug_mass_kg` | extra mass (and inertia) on the plug segments |
| damping | `cable.bend_damping_time_s` | `bend_damping = τ · EI / h` (same idea as the example's `2e-3 × stiffness`) |
| grip | — | the grip segments get **zero mass** = kinematic: held at the scanned position and direction. **This is the boundary condition.** |
| Franka | `robot.*` | `add_urdf` (FR3 + hand), placed with `newton.ik` so `fr3_hand_tcp` sits on the grip, hand z along the cable, fingers closed to the cable radius. All links zero mass → kinematic. Only visual: it doesn't touch the cable. |
| contacts | `contact.*` | explicit pairs: cable–table and cable–cable (more than 3 segments apart, because the scanned cable crosses itself). No robot pairs. |
| solver | `sim.iterations`, `substeps`, `friction_epsilon` | `SolverVBD(iterations, friction_epsilon, rigid_compliant_alm=True)` |
| settle | `sim.settle_window_s`, `settle_tol_m` | runs until no node moved more than `settle_tol_m` (0.1 mm) over `settle_window_s` (0.5 s), or `max_time_s` |

### The key trick: rest shape straight, start pose curved

`init: "scan"` starts the cable **on the scanned curve** but keeps its **rest shape
straight**:

1. The rod is created straight (`Rod.create_straight`). VBD reads the rest
   bend/twist from `model.body_q`, the pose at `finalize()`, so the cable "wants"
   to be straight, like a real cable.
2. After `finalize()`, only the **state** (`state_0.body_q`, `state_1.body_q`) is
   overwritten with segment poses along the scan. The frames come from
   `newton.Rod(nodes).quaternions` (parallel transport, so no twist is added).
   VBD's first step takes its velocity history from this state, so there's no jump.
3. `newton.eval_fk` places the Franka but **never touches rod bodies**, so the
   order of these calls is safe.

So the run answers: *"starting exactly where the real cable is, where does this
cable model go?"* If the model were perfect, it would stay put (`moved ≈ 0`).

`init: "drop"` instead starts the free cable straight and horizontal at grip
height and lets it fall. The hanging part should end up the same. The part lying
on the table won't, because with friction it depends on *how* the cable came
down. That's why `scan` is the default for the comparison.

### Two solver settings that matter for a *static* comparison

Found while testing this scene, and worth knowing for any cable lying on a table:

- **`friction_epsilon`.** VBD smooths friction below this sliding speed. With
  Newton's default (`1e-2` m/s), a cable lying on the table never really stops: it
  creeps about 1 mm/s, forever, so the final shape depends on how long you wait.
  With `1e-4` it sticks: drift falls from ~1 mm to ~0.03 mm per 0.5 s. More
  iterations or more damping did **not** fix it. This setting did.
- **Settling by shape, not speed.** Even at rest, VBD body velocities carry
  ~1–5 mm/s of iteration noise (a few µm per 1/600 s step), on random segments.
  A speed threshold therefore never triggers. The script instead compares the
  centreline every 0.5 s and stops when nothing moved more than 0.1 mm.

### Outputs (`results/ethernet_cat6/<init>/bend_x<scale>/`)

| file | content |
|---|---|
| `sim_centerline.csv` | `s_m, x_m, y_m, z_m, part` (0 grip, 1 scanned stretch, 2 plug), final shape, scan frame |
| `init_centerline.csv` | same columns, the starting shape |
| `sim_centerline.ply` | the final centreline in the **scan PLY's units**: open it with the scan in CloudCompare |
| `meta.json` | every parameter used, IK error, mass, settle time, stretch % |
| `as_scan/` | the final shape in the tube-fit format: use it as a synthetic scan (below) |

### Self-test: can the pipeline find a stiffness it knows?

1. Run once with a known `EI` and a long `max_time_s` so it settles.
2. Point `scan.centerline_csv` / `summary_json` at that run's `as_scan/`.
3. Run `--sweep` and `compare_to_scan.py`.

The `EI` you started from must come out best, with `moved` ≈ 0 for it.
`04_pointcloud_vs_sim/README.md` shows the result of this test.

### When the Franka can't reach

With `robot.base_xyz_m: null`, the base is put on the table, `base_distance_m`
behind the grip, facing it. If IK can't reach, a warning prints the error. The
cable result is unaffected, because the robot is only visual. Set
`robot.base_xyz_m` (and `base_yaw_deg`) to where the real robot stands in the
scan frame.
