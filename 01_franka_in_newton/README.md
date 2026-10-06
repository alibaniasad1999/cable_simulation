# Part 1 — The Franka alone in Newton

> **Goal of this part:** a scene with a floor and a Franka FR3 that you move to
> any hand pose with Newton's IK. No cable yet.
>
> Reference: [`reference/example_franka_cable_ik_pick_place.py`](../reference/example_franka_cable_ik_pick_place.py).
> Line numbers below (`L144`) point into that file. Read the lines, understand
> them, then write your own version from memory.
>
> Newton's API changes between releases. If a name here doesn't exist in your
> version, look at the same example inside your installed Newton
> (`python -m newton.examples franka_cable_ik_pick_place`) and its source.

---

## 1.0 Run the original first

```bash
python -m newton.examples franka_cable_ik_pick_place            # full example
python -m newton.examples franka_cable_ik_pick_place --help     # all options
```

On a Mac, Warp runs on the **CPU**, and CUDA graph capture is skipped
automatically (`L111`: `use_graph and device.is_cuda`). If it's very slow, try
fewer substeps (`--substeps`), or run it on a Linux/NVIDIA machine.

> Check that `SolverMuJoCo` works on your Mac (it needs the `mujoco` /
> `mujoco_warp` packages). If it doesn't, use the **kinematic robot**
> variant (§1.7). It needs no robot solver at all.

---

## 1.1 The skeleton every Newton example uses

Every Newton example is a class with three jobs, plus a small `main`:

```
class Example:
    __init__(viewer, args)   build model → solver → states → control → contacts
    step()                   advance one frame (= several substeps)
    render()                 viewer.begin_frame(t); viewer.log_state(...); viewer.end_frame()

main:
    parser = Example.create_parser()          # newton.examples.create_parser() + your args
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
```

See `L95–L138` (`__init__`), `L495` (`step`), `L504` (`render`), `L561` (main).
Copy this structure exactly. It gives you the viewer, `--device`, frame count
and headless options for free.

**Time:** `fps = 60`, `frame_dt = 1/60`, `sim_dt = frame_dt / substeps`
(`L99–L102`). One `step()` = `substeps` physics steps.

**The substep loop** (`L487–L493`). Learn its shape by heart:

```
for _ in range(substeps):
    state_0.clear_forces()
    collision_pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, sim_dt)
    state_0, state_1 = state_1, state_0          # swap: output becomes next input
```

The objects:

| object | made by | holds |
|---|---|---|
| `builder` | `newton.ModelBuilder(gravity=(0,0,-9.81))` | everything you add, before `finalize()` |
| `model` | `builder.finalize()` | frozen description (masses, joints, shapes) |
| `state_0`, `state_1` | `model.state()` | positions/velocities: `body_q`, `body_qd`, `joint_q`, `joint_qd` |
| `control` | `model.control()` | your commands, e.g. `control.joint_target_q` |
| `contacts` | `collision_pipeline.contacts()` | contact points found this substep |

Z is up (`gravity=(0, 0, -9.81)`). Units: metres, kg, seconds, radians.

---

## 1.2 Add the floor

`L183–L192`:

```python
plane_cfg = newton.ModelBuilder.ShapeConfig(ke=1e3, kd=1e-1, mu=1.0, margin=0.0, gap=0.01)
builder.add_ground_plane(height=surface_z, cfg=plane_cfg, label="cable_ground_plane")
```

| argument | meaning | how to choose |
|---|---|---|
| `height` | z of the floor/table surface | your table height. The example puts it at `cable_center_z − cable_radius` so the cable starts lying on it. |
| `ke` | contact stiffness [N/m] | higher = harder floor, less sinking. 1e3 is soft and matched to the cable. |
| `kd` | contact damping | removes bouncing |
| `mu` | friction coefficient | 1.0 = grippy (cable doesn't slide away) |
| `gap` | distance at which contacts are *detected* (before touching) | a few mm to 1 cm. Too small and fast objects tunnel through. |
| `margin` | extra thickness added to the shape | 0 unless objects sink |

If you want a table *and* a floor, add a box shape for the table
(`builder.add_shape_box(...)`, check the name in your version) and a ground
plane at z = 0.

---

## 1.3 Add the Franka

`L143–L155`:

```python
builder.add_urdf(
    newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
    xform=wp.transform(wp.vec3(0.0, 0.0, base_z), wp.quat_identity()),
    floating=False,
    enable_self_collisions=False,
    parse_visuals_as_colliders=False,
    force_show_colliders=False,
)
```

| argument | meaning |
|---|---|
| `download_asset("franka_emika_panda")` | downloads (once, then cached) Newton's robot asset folder and returns its path. `fr3_franka_hand.urdf` = FR3 arm + Franka hand. Open the folder and look at the files. |
| `xform` | where the robot **base** goes in the world: position + quaternion. Put the base on your table: `z = table height`. Rotate it with the quaternion if your real robot faces another direction. |
| `floating=False` | base bolted to the world. **Always** for a mounted arm. Otherwise it falls over. |
| `enable_self_collisions=False` | links don't collide with each other (faster; adjacent links overlap anyway) |
| `parse_visuals_as_colliders=False` | use the URDF's simple collision meshes, not the detailed visual ones |

**Order matters:** add the Franka **first** in the builder. Then its joints are
coordinates `0…8` (7 arm + 2 fingers), its bodies come first, and the IK model
(§1.5) lines up with it (`L360–L362`). Remember the index ranges before and after
adding (`L202–L221`):

```python
franka_body_start = builder.body_count
... add_urdf ...
franka_bodies = list(range(franka_body_start, builder.body_count))
```

Do the same for joints and shapes. You'll need these lists for collisions and
coupling in Part 3.

**Find bodies by name:** print `builder.body_label` once. The hand is `fr3_hand`
and the fingers contain `finger` (`L270–L274`, `L373`).

### Initial joint configuration

`L33–L44`, `L154–L155`:

```python
FRANKA_Q = [q1, ..., q7, finger1, finger2]     # radians, radians..., metres, metres
builder.joint_q[:9] = FRANKA_Q                  # where the robot starts
builder.joint_target_q[:9] = FRANKA_Q           # where the PD controllers pull it
```

Set **both**. If only `joint_q` is set, the controllers pull the robot to zero
on the first step and it whips around. Fingers: `0.04` = fully open (each
finger travels 0–0.04 m, so max opening 8 cm), `0.0` = closed.

A good "ready" pose: `[0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04]`.
The example's pose (`L34`) has the hand pointing down over the table.

After `finalize()`, compute the body poses from the joints once (`L135–L136`):

```python
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
```

Without this, the bodies sit at their default poses until the first step.

---

## 1.4 How the robot is driven: joint PD targets

The example simulates the arm with **`SolverMuJoCo`**. Each joint gets a PD
controller that pulls it toward `control.joint_target_q`. Settings (`L208–L217`):

```python
builder.joint_target_ke[:7] = [400.0] * 7    # arm stiffness (P gain)
builder.joint_target_kd[:7] = [80.0] * 7     # arm damping   (D gain)
builder.joint_target_ke[7:9] = [1000.0] * 2  # finger stiffness
builder.joint_target_kd[7:9] = [100.0] * 2
builder.joint_effort_limit[:4]  = [87.0] * 4   # max torque [N·m], real Franka joints 1-4
builder.joint_effort_limit[4:7] = [12.0] * 3   # real Franka joints 5-7
builder.joint_effort_limit[7:9] = [1500.0] * 2 # gripper force limit
builder.joint_armature[:7] = [1e-3] * 7        # small rotor inertia, stabilises the solver
```

| knob | effect if too low | effect if too high |
|---|---|---|
| `joint_target_ke` | arm lags behind IK, sags | jittery, needs more substeps |
| `joint_target_kd` | overshoot/oscillation | sluggish |
| `joint_effort_limit` | can't hold the pose | unrealistic forces |

**Gravity compensation** (`L223–L229`): the real Franka cancels gravity
internally. In MuJoCo you enable it per body:

```python
SolverMuJoCo.register_custom_attributes(builder)   # must be called before adding things (L160)
gravcomp = builder.custom_attributes["mujoco:gravcomp"]
gravcomp.values = gravcomp.values or {}
for b in franka_bodies:
    gravcomp.values[b] = 1.0
```

With it, the PD gains only track the target and don't fight gravity. Without
it, the arm droops below the IK target.

**The robot solver** (robot only, for this part):

```python
solver = SolverMuJoCo(model, solver="newton", integrator="implicitfast",
                      cone="elliptic", iterations=100, ls_iterations=20)
```

In Part 3 this same solver becomes one "entry" of the coupled solver. For now
use it directly.

---

## 1.5 Moving the hand: Newton's IK

You don't set 7 joint angles by hand. You say "hand here, pointing like this"
and IK finds the joints. `L359–L405`:

**(a) A separate Franka-only model for IK.** Build a second builder with only the
Franka (same `add_urdf` call, same base pose), and `finalize` it. IK then never
sees the cable's bodies. Because the Franka was added first in the main model,
the first `n_coords = ik_model.joint_coord_count` coordinates match.

**(b) Objectives.** IK minimises a sum of objectives:

```python
hand = index of "fr3_hand" in ik_model.body_label

pos_obj = ik.IKObjectivePosition(
    link_index=hand,
    link_offset=wp.vec3(0.0, 0.0, 0.107),      # TCP: 10.7 cm along hand z = between fingertips
    target_positions=target_pos_array)        # wp.array of wp.vec3, one per world

rot_obj = ik.IKObjectiveRotation(
    link_index=hand,
    link_offset_rotation=wp.quat_identity(),
    target_rotations=target_rot_array)        # wp.array of wp.vec4 (x, y, z, w)

limits_obj = ik.IKObjectiveJointLimit(
    joint_limit_lower=..., joint_limit_upper=..., weight=10.0)

ik_solver = ik.IKSolver(model=ik_model, n_problems=world_count,
                        objectives=[pos_obj, rot_obj, limits_obj],
                        lambda_initial=0.05,
                        jacobian_mode=ik.IKJacobianType.ANALYTIC)
```

- **`link_offset = 0.107`** is important: it makes the *point between the
  fingertips* (the tool centre point, TCP) go to your target, not the hand's
  flange. When you grasp a cable, the target is where the cable is.
- **Orientation as a quaternion `(x, y, z, w)`.** `GRIPPER_DOWN = (1, 0, 0, 0)`
  (`L53`) = 180° about world x, so the hand's z axis points **down**:
  a top-down grasp. Newton/Warp use `(x, y, z, w)`. MuJoCo, Isaac and many
  papers use `(w, x, y, z)`. Mixing them is the classic bug.
- `lambda_initial`: damping of the IK step (stability near singular poses).

**(c) Each frame: solve → copy into the PD targets** (`L478–L485`):

```python
ik_solver.step(ik_joint_q, ik_joint_q, iterations=24)   # in-place, warm-started
# put finger widths into the last two coords (kernel set_gripper_q, L67)
wp.copy(dest=control_joint_target_q[:, :n_coords], src=ik_joint_q)
```

`ik_joint_q` keeps the last solution, so each solve starts from the previous
one (*warm start*). It converges fast and the elbow doesn't flip.

**(d) Changing the target.** Write new values into `target_pos_array` /
`target_rot_array` before `simulate()`. The example does it with a tiny Warp
kernel (`set_task_targets`, `L73`). From plain Python you can also do
`arr.assign(...)` or create the array from numpy. That's fine on the CPU.

---

## 1.6 Scripting motion: keyframes

`L410–L462`: a table of `[duration, x, y, z, qx, qy, qz, qw, finger]` rows, and
`update_ik_targets()` linearly interpolates between rows by time. This is how
the example does approach → descend → grasp → lift → move → release.

For your project, keyframes are the simplest way to **reproduce a real robot
pose**: one row "go to the recorded hand pose and stay". The joystick (Part 5)
just replaces this function with "target += stick velocity × dt".

> Linear interpolation of a quaternion's 4 numbers is only OK for small
> rotation changes (the example keeps the orientation constant). For big
> rotations, use spherical interpolation (slerp) and renormalise.

---

## 1.7 Alternative: a kinematic robot (no robot physics)

If you only need the robot to **be at a pose** (Part 4: reproduce the real
capture), you don't need MuJoCo at all:

1. solve IK (or use the real recorded joint angles directly),
2. write them into `state.joint_q`,
3. `newton.eval_fk(model, state.joint_q, state.joint_qd, state)` → body poses.

The robot then follows exactly, with no gains to tune. The cable (Part 2) is then
attached to the hand directly instead of being grasped by contact. Part 3
explains both options.

---

## Tasks

Write your code in this folder, **without copying**. Look at the reference only when stuck.

1. **Skeleton + floor.** Example class, empty builder, ground plane, a falling box.
2. **Franka.** `add_urdf`, fixed base on the floor, initial `joint_q` and
   `joint_target_q`, `eval_fk`. Print all body labels and the joint count.
3. **Hold the pose.** `SolverMuJoCo` + PD gains + gravity compensation. The robot
   must stand still for 10 s. Then try turning gravcomp off and see the sag.
4. **IK to one target.** Separate IK model, three objectives, TCP offset. Send the
   TCP 10 cm forward, hand down. Print the TCP position from FK and check it's
   within 1–2 mm.
5. **Keyframes.** Approach → down → close fingers → up. No cable yet.
6. **Kinematic variant.** Same keyframes, without MuJoCo (§1.7).

## Check yourself

- Why must the Franka be the first thing added to the builder?
- What happens if you set `joint_q` but not `joint_target_q`?
- What does `link_offset=(0, 0, 0.107)` change?
- Write the quaternion for "hand pointing down" in `(x, y, z, w)`, and in `(w, x, y, z)`.
- When would you choose the kinematic robot over MuJoCo?

## Common mistakes

- `floating=True` (or forgetting it): the robot falls over.
- Forgetting `register_custom_attributes` **before** adding the robot, so `gravcomp` is missing.
- Quaternion order mixed up.
- IK target given for the hand flange instead of the TCP: the fingers end up 10.7 cm off.
- Forgetting `eval_fk` after `finalize()`.
