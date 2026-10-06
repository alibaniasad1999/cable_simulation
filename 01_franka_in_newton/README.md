# Part 1 — The Franka robot in Newton

> Study notes. Read the **Concepts** first, then do the **Tasks** in order, then
> try the **Exercises**. Answer the **Self-check** questions without looking
> back. If you can't, re-read that concept before moving to Part 2.

## What you'll be able to do at the end

- Explain how a physics engine advances time, and what Newton's `Model`,
  `State`, `Control` and solver each do.
- Load a robot from a URDF/MJCF file and know what is inside that file.
- Work fluently with poses: positions, quaternions, homogeneous transforms, frames.
- Compute forward kinematics (FK) and write your own inverse kinematics (IK).
- Choose between kinematic and dynamic robot control, and justify the choice.

---

## Concepts

### 1. What a physics engine actually does

A simulator stores the **state** of the world (positions `x`, velocities `v`) and
repeatedly computes the state a small time `Δt` later:

```
v(t+Δt) = v(t) + Δt · M⁻¹ · f(x, v)        (forces → accelerations)
x(t+Δt) = x(t) + Δt · v(t+Δt)              (velocities → positions)
```

This is *semi-implicit (symplectic) Euler*, the simplest stable-ish scheme.
Real solvers are more sophisticated, but every one of them is "state in → state
out, one `Δt` at a time".

Two numbers control everything:

- **frame rate**: how often you *look* at the simulation (render, read the
  joystick), typically 60 Hz;
- **substeps**: how many physics steps per frame. `Δt = (1/60) / substeps`.

Stiff things (a steel-like cable, a robot with high gains) need a small `Δt` or
an implicit solver. An explicit spring of stiffness `k` on a mass `m` is stable
only if roughly `Δt < 2/ω` with `ω = √(k/m)`. Remember this. It explains a lot
of "my simulation exploded" moments in Part 3.

### 2. Newton's architecture

Newton is written on top of **NVIDIA Warp**. Warp compiles Python functions
("kernels") to fast code for the CPU or an NVIDIA GPU. On a Mac, Warp runs on the
**CPU only**. Newton's objects:

| object | what it is | changes during simulation? |
|---|---|---|
| `ModelBuilder` | a "construction kit": you add bodies, joints, shapes, robots | — (used before) |
| `Model` | the frozen description: masses, joint types, shapes, gravity | no |
| `State` | positions and velocities of bodies/joints (`body_q`, `body_qd`, `joint_q`, `joint_qd`) | yes, every step |
| `Control` | what you command: joint targets, forces | yes, you write it |
| `Contacts` | the collision pairs found this step | yes |
| `Solver*` | the algorithm that maps `State_in → State_out` | — |

You keep **two** states and swap them each substep (the solver reads one, writes
the other). This "double buffering" avoids overwriting data that is still being
read.

Newton ships several solvers with different strengths. Examples are
`SolverMuJoCo` (robots, contacts), `SolverFeatherstone` (articulations in joint
coordinates), `SolverXPBD`, and `SolverVBD` (deformables and cables, Part 3).
Learning **which solver suits which problem** is part of this course.

> Newton's API is young and changes between versions. The examples that ship
> with *your installed version* are the ground truth for exact names. Run the
> example list, open the source of a robot example, and read it like a textbook.

### 3. Poses, frames and transforms

A **pose** = position `p ∈ ℝ³` + orientation. Orientation can be written as:

- a **rotation matrix** `R` (3×3, orthonormal, det = +1), easiest for math;
- a **quaternion** `q = (x, y, z, w)` with `|q| = 1`, compact, no singularities,
  what Newton/Warp store;
- **axis-angle** `θ·n̂`, best for *errors* and *small rotations*;
- Euler angles: avoid them except for printing.

**Homogeneous transform** — pack `R` and `p` in a 4×4 matrix:

```
        ┌ R  p ┐
T_A_B = │      │      maps a point written in frame B into frame A
        └ 0  1 ┘

T_A_C = T_A_B · T_B_C              (chain frames — read the subscripts like dominoes)
T_B_A = T_A_B⁻¹ = [Rᵀ, −Rᵀp; 0, 1]
```

Adopt a naming convention now and never break it: `T_world_hand`,
`T_base_camera`, `T_hand_grasp`. Half of all robotics bugs are a transform used
in the wrong direction. The subscript rule makes those bugs visible.

**Frames you will meet:** `world`, `base` (robot base, here = world), each
`link`, `hand` (the flange/gripper body), `TCP` (tool centre point, between the
fingertips), and in Part 4 `camera` and `cloud`.

### 4. Articulated robots

An **articulation** is a tree of rigid **links** connected by **joints**.
Each joint has degrees of freedom (DOF). Revolute = 1 rotation, prismatic = 1
translation, fixed = 0.

The Franka Emika Panda:
- 7 revolute arm joints (it's *redundant*: 7 DOF for a 6-DOF task, so infinitely
  many joint configurations reach the same hand pose),
- 2 prismatic finger joints (each 0 to 0.04 m),
- joint limits from the datasheet (rad, approximately):
  `q1 ±2.90, q2 ±1.76, q3 ±2.90, q4 [−3.07, −0.07], q5 ±2.90, q6 [−0.02, 3.75], q7 ±2.90`.
  Note that `q4` is **never** zero or positive, because the elbow is always bent.

**Two ways to describe the same robot:**
- *maximal coordinates*: each link has a full 6-DOF pose (`body_q`) and joints
  are constraints between them;
- *generalised (reduced) coordinates*: only the joint angles (`joint_q`). The
  link poses follow from FK.

Newton stores both. Know which one your solver integrates.

**URDF vs MJCF**: both are XML robot descriptions. They contain links (with
mass, centre of mass, inertia tensor, visual mesh, collision mesh) and joints
(type, axis, parent/child, origin transform, limits). Open the Franka file in a
text editor and find: the `panda_joint4` limits, the `panda_hand` link, and the
finger joints. Reading it once teaches you more than any summary.

### 5. Forward kinematics (FK)

FK = "given joint angles `q`, where is every link?". It's just a chain of
transforms from the base:

```
T_base_hand(q) = T_0(q1) · T_1(q2) · … · T_6(q7) · T_flange_hand
```

where each `T_i` = (fixed origin transform from the URDF) × (rotation about the
joint axis by `qi`). Newton computes this for you (`eval_fk` or similar). In
Exercise 1 you implement it yourself once, to understand it.

### 6. The Jacobian

The Jacobian `J(q)` (6×7 for the Panda) maps joint velocities to the hand's
**twist** (linear velocity `v` and angular velocity `ω`):

```
┌ v ┐
│   │ = J(q) · q̇
└ ω ┘
```

Column `i` = "how the hand moves if only joint `i` moves". You can get it by
**finite differences** of FK (perturb each `qi` by `ε ≈ 1e-6`, measure the pose
change). That's slow in theory, but at 7 joints it's perfectly fine and very
educational.

A pose is **singular** where `J` loses rank. Some hand direction then needs
infinite joint speed. Measure "how far from singular" with the *manipulability*
`w = √det(J Jᵀ)`.

### 7. Inverse kinematics (IK)

IK = "which `q` puts the hand at a target pose?". For a general robot there is
no closed form, so you **iterate** (Newton–Raphson on the pose error):

1. pose error `e` (6-vector):
   - position part: `p_target − p_current`
   - orientation part: the axis-angle vector of `R_target · R_currentᵀ`
2. step: `Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ · e`  ← **damped least squares**
3. `q ← clamp(q + Δq, limits)`. Repeat until `|e|` is small.

Why the damping `λ`? Plain pseudoinverse `Jᵀ(JJᵀ)⁻¹` blows up near
singularities. `λ` trades a little accuracy for stability. Typical
`λ ≈ 0.01–0.1`.

**Redundancy:** because the Panda has 7 DOF, you can add a secondary goal in the
**null space** of `J` without disturbing the hand. For example, "stay close to
the home pose": `Δq += (I − J⁺J) · k·(q_home − q)`. This keeps the elbow from
wandering.

### 8. Kinematic vs dynamic control

| | kinematic | dynamic (PD) |
|---|---|---|
| what you set | `joint_q` directly, then FK | target `q*`; the solver applies torques `τ = kp(q* − q) − kd·q̇` |
| tracking | perfect | lags, overshoots if gains are wrong |
| can objects push the robot? | no | yes |
| good for this project? | **yes**: a real Franka in position control is not pushed around by a light cable | only if you later need contact forces or arm dynamics |

---

## Tasks

Write your code in this folder (`01_franka_in_newton/`).

**Task 1.1 — Run the examples.** Install Newton (package `newton-physics`,
import `newton`). List the bundled examples and run one robot example and one
cable example. Write down the frame rate you get on your Mac. *You should see:*
a viewer window with a moving robot.

**Task 1.2 — Empty world.** Builder → ground plane → finalize → solver → viewer
→ loop with substeps and state swapping. Drop one box from 1 m and check it
lands. *You should see:* the box falls and rests on the ground.

**Task 1.3 — Load the Franka.** Import the URDF/MJCF (copy how the Newton robot
example obtains its Franka asset), fix the base, set the home pose
`[0, −0.785, 0, −2.356, 0, 1.571, 0.785]` + fingers open. Print every body name
and its index. Store the index of the hand body. *You should see:* the robot
standing still in the "ready" pose.

**Task 1.4 — Read the hand pose.** After FK, print `T_world_hand` (position +
quaternion). Move one joint by hand in the code and check the hand moves the way
you expect.

**Task 1.5 — Your own IK.** Implement damped-least-squares IK with a
finite-difference Jacobian. Give it a target 10 cm in front of the current hand
position. *Done when:* FK of the IK result is within 1 mm / 0.5° of the target.

**Task 1.6 — Gripper.** Open and close the fingers (prismatic joints 0 → 0.04 m).

---

## Exercises

1. **FK by hand.** Using only the joint origins and axes from the URDF (no Newton
   FK), compute `T_base_hand` for the home pose with numpy. Compare with Newton.
   They must agree to ~1e-6 m. If not, you have a transform order bug, and finding
   it will teach you more than the rest of this part.
2. **Damping study.** Run IK from the same start to the same target for
   `λ = 0.001, 0.01, 0.1, 1`. Plot error vs iteration. What does `λ` trade off?
3. **Singularity hunt.** Move the hand along a straight line that stretches the
   arm fully forward. Plot manipulability `w` along the path. What happens to
   `|Δq|` as `w → 0`, with and without damping?
4. **Null space.** Reach the same hand target twice: once with the null-space term,
   once without. Compare the final elbow positions.
5. **Quaternion drill.** Convert a quaternion to a rotation matrix by the formula
   (no library), compose two rotations both ways, and verify `q` and `−q` give the
   same rotation.

## Self-check

- Why do we keep two `State` objects?
- What's the difference between `body_q` and `joint_q`?
- Write `T_camera_hand` in terms of `T_world_camera` and `T_world_hand`.
- Why is the Panda "redundant", and what does that let IK do?
- What is the orientation part of the IK error, and why not just subtract Euler angles?
- Why is kinematic control acceptable for a robot holding a light cable?

## Common mistakes

- **Quaternion order.** Newton/Warp and SciPy use `(x, y, z, w)`. MuJoCo, Isaac
  Sim and many papers use `(w, x, y, z)`. Mixing them gives a rotation that
  *looks almost right*. Always test with a 90° rotation about one axis.
- **Up axis.** Check whether your world is Z-up (Newton's usual default) or Y-up,
  especially when importing assets.
- **Units.** Metres, kilograms, seconds, radians, everywhere, always.
- **Forgetting to fix the base.** The robot falls over or flies away.
- **Joint limits ignored in IK.** The solution "works" but is impossible on the
  real robot (remember `q4 < 0`).
- **Reading state before FK was evaluated.** You print the old pose.

## Further reading

- K. Lynch & F. Park, *Modern Robotics* (free PDF + videos). Chapters 3 (rigid
  motions), 4 (FK), 5 (Jacobian), 6 (IK). The best single reference for this part.
- S. Buss, *Introduction to Inverse Kinematics with Jacobian Transpose,
  Pseudoinverse and Damped Least Squares Methods* (2004).
- Newton repository: README, `newton/examples/` source.
- Franka Emika Panda datasheet: joint limits, velocity limits.
