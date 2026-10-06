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
