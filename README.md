# Cable simulation with a Franka robot — implementation guide

This repo is being **rebuilt from scratch**. All previous work (the cable-hang
study: catenary math, photo measurement, Newton / PhysX / Warp solvers) now lives
in [`legacy/`](legacy/) and is kept only for reference.

This README is a **plan, not code**. It tells you what to build, in what order,
which tools to use, what to check before moving on, and where the traps are. You
write the code yourself.

---

## The goal

```
 Stage 1            Stage 2              Stage 3               Stage 4
 Franka in    ──▶   drive it with  ──▶   cable held in   ──▶   real point cloud
 Newton             a joystick (Mac)     the gripper            vs simulation:
                                                                how big is the error?
```

1. **Franka in Newton.** Load the Franka Emika Panda into the
   [Newton](https://github.com/newton-physics/newton) physics engine, see it in
   the viewer, and move the hand to a target position.
2. **Joystick teleoperation on macOS.** A gamepad (or SpaceMouse) moves the
   Franka hand in real time.
3. **A cable in the hand.** A deformable cable (a Newton rod) is held by the
   gripper and swings/drapes as you move the arm.
4. **Real vs sim.** You have a point cloud of a real cable held in a real
   Franka hand. Rebuild that exact scene in simulation, then **measure the
   difference** in millimetres.

Every stage ends with a **"done when"** checklist. Don't start the next stage
until the checklist passes. Most of the pain in the legacy project came from
debugging several layers at once.

---

## Suggested layout (create it as you go)

```
cable_simulation/
├── README.md               this file
├── legacy/                 old project, read-only reference
├── pyproject.toml          dependencies (uv or pip)
├── configs/
│   ├── cable.yaml          cable properties (length, radius, mass, EI, segments)
│   ├── robot.yaml          home pose, gains, gripper offset
│   └── capture_XXX.yaml    one per real capture (cloud path, joint angles, camera)
├── src/cablesim/
│   ├── world.py            builds the Newton model (ground + robot + cable)
│   ├── robot.py            Franka loading, FK, IK, gripper
│   ├── cable.py            cable creation, node read-back, stiffness formulas
│   ├── teleop.py           joystick → end-effector velocity command
│   ├── pointcloud.py       load, transform, segment, centerline
│   └── metrics.py          real-vs-sim distances
├── scripts/
│   ├── 01_franka_view.py
│   ├── 02_teleop.py
│   ├── 03_teleop_cable.py
│   └── 04_compare_cloud.py
├── data/                   real captures (big files: gitignore or use Git LFS)
└── results/                outputs (gitignored)
```

One rule from the legacy project worth keeping: **every number that describes
the physical setup lives in a config file, not in the code.** Then a new cable or
a new capture means a new YAML file, not an edited script.

---

## Stage 0 — Environment

### Newton on macOS: what to expect

- Newton is built on **NVIDIA Warp**. On macOS (Apple Silicon or Intel), Warp
  runs **on the CPU only**: there's no CUDA, and Metal isn't a Warp backend. Everything
  works, but it's slower.
- Practical consequence: **develop on the Mac with a small cable** (30–60
  segments) and fewer solver iterations. Run the heavy studies (Stage 4 parameter
  sweeps) on a Linux machine with an NVIDIA GPU if you have one. The same code
  runs on both. Only the Warp device changes (`cpu` vs `cuda:0`).
- The legacy code ran Newton **inside Isaac Sim** (`~/isaacsim/python.sh`). You
  don't need Isaac Sim this time. Plain Newton from pip has its own viewer.

### Install

1. Install Python 3.10–3.12 and [`uv`](https://docs.astral.sh/uv/) (or a plain venv).
2. Install Newton. Use the current command in the Newton README; the PyPI
   package is `newton-physics`, imported as `newton`. Install the extras for
   examples/viewer if the README lists them.
3. Run the bundled examples first, **before writing any code**:
   - list them (the Newton README shows the `python -m newton.examples ...`
     command),
   - run at least one **robot** example (look for `franka` / `panda` /
     `robot` in the names),
   - run at least one **cable** example (look for `cable` in the names).

   These examples are the best documentation you have. They show the exact API
   of *your installed version*. Newton's API is still changing, so method names
   in this README may drift. When in doubt, trust the examples.
4. Later stages also need: `pygame` (joystick), `open3d` (point clouds),
   `scipy`, `numpy`, `matplotlib`, `pyyaml`, and `opencv-python` for hand-eye
   calibration.

**Done when:** a Franka example and a cable example both open in the viewer on
your Mac, and you've written down which device (`cpu`) and frame rate you get.

---

## Stage 1 — Franka in Newton

### 1.1 Empty world

Build the smallest possible scene: a `newton.ModelBuilder`, a ground plane,
`finalize()`, a solver, a viewer, and the standard loop:

```
for each frame:
    for each substep:
        clear forces → solver.step(state_in, state_out, control, contacts, dt) → swap states
    viewer.log_state(state) / render
```

Learn the four objects you'll use everywhere: **Model** (static description),
**State** (positions/velocities, you keep two and swap them), **Control**
(joint targets, forces), **Contacts**.

### 1.2 Load the Franka

- Get a Franka description. Either:
  - the URDF/MJCF Newton's own Franka example uses (it downloads assets with a
    helper in `newton.utils`; copy that approach), or
  - MuJoCo Menagerie's `franka_emika_panda` (MJCF).
- Load it with the builder's URDF/MJCF importer. **Fix the base** to the world
  (no floating base).
- Set a sensible **home configuration** for the 7 arm joints (the classic
  "ready" pose is about `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]` rad) and the
  2 finger joints.
- Find and store the **index of the hand body** (`panda_hand` or the TCP frame).
  You'll need it in every later stage. Print all body names once and pick it.

### 1.3 Choose how the robot is driven

You have two options. Pick one consciously:

| option | how | pros | cons |
|---|---|---|---|
| **A. Kinematic** (recommended first) | You write `joint_q` directly every frame and run forward kinematics (`newton.eval_fk` or equivalent). The robot isn't simulated. | Perfect tracking, no gains to tune, any solver works for the cable. | The robot can't be pushed by anything. That's fine here: a real Franka under position control isn't moved by a light cable either. |
| **B. Dynamic** | Joint PD targets (`joint_target_ke / kd`) solved by e.g. `SolverMuJoCo` or `SolverFeatherstone`. | Realistic dynamics. | Gain tuning. Coupling a dynamic robot and a VBD cable in one step is harder (see Stage 3). |

For this project the robot is a **precise positioning device**, so **option A is
the right default**. Use option B only if you later need contact forces or arm
dynamics.

### 1.4 Inverse kinematics (IK)

You need "put the hand at pose X" rather than "set joint angles". Either:

- use Newton's IK module (`newton.ik`, check your version and its example), or
- **write your own damped-least-squares IK**. It's short, and you'll understand it fully:
  - error `e` = 6-vector (position error, orientation error as axis-angle),
  - Jacobian `J` (6×7): from Newton, or by finite differences of FK (fine at
    this size),
  - update `Δq = Jᵀ (J Jᵀ + λ² I)⁻¹ e`, with `λ ≈ 0.05`,
  - clamp to joint limits, iterate a few times per frame,
  - optional: use the 7th DOF (null space) to stay near the home pose.

**Done when:**
- [ ] The Franka stands in its home pose in the viewer, with a fixed base, not falling.
- [ ] You can print the hand pose (position + quaternion) in the world frame.
- [ ] Changing a target position in the code moves the hand there, and FK of the
      IK solution agrees with the target to < 1 mm.
- [ ] Opening/closing the gripper fingers works.

---

## Stage 2 — Joystick control on macOS

### 2.1 Hardware and reading the device

- **Gamepad** (Xbox / PS4 / PS5 over Bluetooth or USB): read it with
  **`pygame.joystick`** (SDL2 underneath, works well on macOS). Print axes and
  buttons first, because the axis numbering differs between controllers. Write the
  mapping into `configs/robot.yaml`, not the code.
- **3Dconnexion SpaceMouse** (very nice for 6-DOF): `pyspacemouse` +
  `brew install hidapi`. macOS may ask for **Input Monitoring** permission
  (System Settings → Privacy & Security) for your terminal/IDE.

### 2.2 macOS-specific traps

- **Window and event handling must run on the main thread** on macOS. Don't
  put the viewer or pygame in a background thread. Use one loop:
  `poll joystick → update target → IK → step sim → render`.
- pygame needs its event queue pumped every frame (`pygame.event.pump()` or
  `get()`), even if you only read axes. Otherwise values freeze.
- Initialise only `pygame.joystick` (and `pygame.display` if SDL complains).
  You don't need a pygame window.

### 2.3 The control law (Cartesian velocity teleop)

Each frame, with `dt = 1/fps`:

1. Read axes → apply a **deadzone** (≈ 0.1) → optionally square them for finer
   control near zero.
2. Scale to an end-effector **twist**: linear velocity `v` (e.g. max 0.2 m/s)
   and angular velocity `ω` (e.g. max 0.8 rad/s).
3. **Integrate a target pose**: `p_target += v·dt`,
   `R_target = exp(ω·dt) · R_target`.
4. **Clamp the target** to a safe workspace box in front of the robot.
5. IK → joint targets. If the IK error stays large, **don't** move the
   target further (stops runaway into unreachable space).
6. Buttons: gripper open/close, "go home", speed toggle, and a **record**
   button that saves the current joint angles (useful in Stage 4).

Suggested mapping: left stick = x/y, triggers = z, right stick = yaw/pitch,
bumpers = roll. Decide whether `v` is in the **world/base frame** or the
**hand frame**. World frame is easier for driving, hand frame is easier for
aligning the gripper with a cable. Make it a button toggle.

### 2.4 Optional: split machines

If the Mac is too slow for the cable later, run the **sim on a Linux GPU box** and
keep **only the joystick on the Mac**: send the twist over UDP (a tiny JSON or
packed-float message at 60 Hz). Design `teleop.py` so it outputs a plain
`(v, ω, buttons)` tuple. Then a network transport is just another source.

**Done when:**
- [ ] Moving each stick moves the hand smoothly in the expected direction.
- [ ] Releasing the sticks stops the hand dead (no drift, deadzone works).
- [ ] The hand can't leave the workspace box, and singular poses don't explode.
- [ ] It runs at a steady frame rate on the Mac (measure it and write it down).

---

## Stage 3 — A cable in the Franka hand

### 3.1 The cable model

Use Newton's rod: **`ModelBuilder.add_rod(positions, radius, ..., stretch_stiffness, bend_stiffness, ...)`**.
It builds a chain of capsule bodies joined by cable joints (a discrete
Cosserat rod), solved with **`SolverVBD`** (implicit, stable for stiff cables).
`legacy/methods/hang_newton_cable.py` uses exactly this. Read it for the
details below.

Things to get right:

- **Stiffness from material properties.** For segment length `l`:
  `stretch_k = E·A / l`, `bend_k = E·I / l`, with `A = πr²`, `I = πr⁴/4`
  (see `rod_stiffness` in `legacy/methods/cable_config.py`).
- **Mass.** Newton computes body mass from shape volume × density. Set
  `density = cable_mass / (π r² · L)` so the total mass equals the real
  cable's.
- **Measure the real cable**: mass (kitchen scale, then mass per metre), outer diameter
  (calipers), length. `E` (or directly `EI`) is the **one unknown**. It's
  identified in Stage 4.
- **Rest shape = initial shape.** The rod's rest curvature is the shape you
  create it in. Create it **straight** unless you deliberately model the real
  cable's natural curl.
- **Segments.** 30–60 on the Mac for teleop, 100–150 for the accuracy study.
  Check the result doesn't change when you double it.
- **Solver budget.** The legacy sweep found that accuracy depends on
  `substeps × iterations`, that VBD needs ≳100 iterations before the cable stops
  visibly stretching, and that `4 substeps × 200 iterations` was a good
  default at 150 segments. Damping the stretch constraint *hurt*.
- **Read back the centreline.** Bodies are capsules. Recover the N+1 nodes from
  the body transforms (`read_nodes` in the legacy file does this).

### 3.2 Attaching the cable to the gripper (the key design decision)

| option | how | notes |
|---|---|---|
| **A. Kinematic attachment** (start here) | Make the first 1–2 rod bodies **massless** (zero mass/inertia, as the legacy code does for supports) and **write their pose every substep** = hand pose × fixed grasp offset. Set their velocity consistently (finite difference of pose) so VBD sees a smooth motion. | Pairs perfectly with the kinematic robot (Stage 1, option A). One solver (VBD) handles only the cable. Simple and robust. |
| **B. Joint to the hand** | A fixed joint between the hand body and the first rod body, everything in one model/solver. | Only if your Newton version's solver handles the robot articulation and the cable joints together. Check the Newton cable examples for a gripper/robot case before attempting. |
| **C. Real grasp by friction** | Close the fingers on the cable and rely on contact. | Hardest: needs finger-cable contact tuning. Only if slip in the gripper is something you want to study. |

**Grasp offset**: define the TCP point between the fingertips and the cable's
direction in the hand frame (e.g. cable leaves along the hand's ±y between the
fingers). Put it in `robot.yaml`. Stage 4 depends on matching this to reality.

**Collisions**: turn on cable–ground first. Turn on cable–robot links later
(so the cable can drape over the hand). Keep self-collision **off** until
everything else works. It's expensive, and a hanging cable rarely touches
itself.

**Done when:**
- [ ] With the arm still, the cable hangs from the gripper and **settles**
      (kinetic energy → ~0).
- [ ] **Arc-length drift** < 0.5 % at rest (an inextensible cable must keep
      its length). This is the reference-free sanity check from the legacy project.
- [ ] Driving with the joystick, the cable swings and follows the hand without
      exploding, jittering, or detaching.
- [ ] The rest shape doesn't change noticeably when you double the segment count.
- [ ] Laying the cable on the ground works (contact is stable).

---

## Stage 4 — Real point cloud vs simulation

This is the scientific part. The question is: **"Given the same robot pose and
the same cable, how far (in mm) is the simulated cable from the real one?"**

### 4.1 What you need from each real capture

A point cloud alone isn't enough. For every capture, save:

| item | why |
|---|---|
| the point cloud (`.ply` / `.pcd`), with colours if available | the measurement |
| **robot joint angles `q`** at capture time | FK gives the exact hand pose. That's the boundary condition. |
| camera **intrinsics** and **extrinsics** `T_base_camera` | to express the cloud in the robot base frame |
| **free length** of cable from fingertips to tip (tape measure) | the sim must use the same length |
| where/how the cable sits between the fingers (photo) | the grasp offset and angle |
| cable mass, diameter | Stage 3 parameters |
| an RGB photo | for checking segmentation by eye |

Capture **after the cable has stopped swinging** (static equilibrium, which the sim
can reproduce exactly). Capture **several different poses**, with at least one
where the cable leaves the gripper **roughly horizontally** (see 4.6).

### 4.2 Putting the cloud in the robot frame

- If the camera is fixed in the room (eye-to-hand): do a **hand-eye
  calibration**. Put an ArUco/ChArUco board on the gripper, record ~15
  robot poses + images, and solve with OpenCV
  (`cv2.calibrateRobotWorldHandEye` or `cv2.calibrateHandEye`).
- If the camera is on the wrist (eye-in-hand): same idea, the other variant.
- **Check the calibration**: transform the cloud into the base frame and
  overlay the Franka's FK link meshes. The real gripper points must sit on the
  simulated gripper. The offset you see here is a **floor on every error you'll
  report**. Write it down.

### 4.3 Extracting the cable from the cloud

A pipeline with Open3D, each step saved so you can look at it:

1. Load → transform to the base frame.
2. **Crop** to a box around the region below/around the gripper.
3. Remove the **table** (RANSAC plane fit, `segment_plane`) if visible.
4. Remove the **robot**: drop points within a few mm of the robot link
   meshes posed at FK(q), or simply crop away the hand.
5. Optional colour filter if the cable colour is distinctive.
6. **Statistical outlier removal**, then **DBSCAN** clustering. Keep the
   cluster closest to the gripper.
7. Visualise it. This is the equivalent of the legacy `detection.png`, and you
   should **always look at it**.

### 4.4 From points to a centreline

The camera sees one side of a tube, so the points sit on the **surface**,
about one radius from the centreline, biased toward the camera. Options:

- **Simple and robust:** compare the sim's *tube surface* to the cloud
  (point-to-centreline distance minus radius, see 4.5). No centreline
  extraction needed.
- **Full centreline** (needed for arc-length comparison and length checks):
  voxel-downsample → k-NN graph → minimum spanning tree → **longest path**
  starting from the point nearest the gripper → smoothing spline
  (`scipy.interpolate.splprep`) → resample at equal arc-length steps → shift
  each point by `r` away from the camera.
- **Check:** the extracted length should match the tape-measured free length
  (within a few %, minus the part hidden by the fingers). If not, the
  segmentation is wrong, so fix it before comparing anything.

### 4.5 Simulating the same scene

- Set the robot to the recorded `q` (kinematic, no teleop).
- Create the cable with the **measured free length**, attached at the **same
  grasp offset/angle**, initially straight and pointing along the grasp
  direction.
- Let it **settle** (legacy `SettleMonitor` idea: stop when tip motion stays
  below ~0.1 mm per 0.1 s; average the last 0.5 s).
- Read the centreline nodes **in the robot base frame**.

**Don't ICP-align the sim to the cloud.** Both are already in the base frame.
Aligning them would hide exactly the error you want to measure (and absorb
calibration and grasp errors invisibly).

### 4.6 Metrics: how big is the difference?

Report all of these in **mm**:

| metric | definition | tells you |
|---|---|---|
| **cloud→sim surface distance** | for each real point: distance to the sim centreline − `r`. Report mean, RMS, 95th percentile, max. | overall shape error; robust to occlusion (only uses points you actually saw) |
| **arc-length error `e(s)`** | `‖c_real(s) − c_sim(s)‖` at equal arc length `s` from the gripper; plot vs `s` | *where* the error is: near the grasp (boundary condition) vs the tip (stiffness / length) |
| **tip error** | distance between real and sim free ends | single headline number |
| **Chamfer / Hausdorff** | symmetric average / worst-case set distance | standard for comparing with papers |

Figures to make: the 3D overlay (cloud + sim tube + robot), `e(s)`, and a
histogram of point distances.

### 4.7 Error budget: is the difference real?

Before blaming the simulator, estimate the measurement noise:

- calibration error (from 4.2, often 2–5 mm),
- depth-sensor noise (a RealSense-class camera: about 1–2 % of distance),
- grasp offset/angle uncertainty (rotating the grasp by 5° moves a 0.5 m
  cable's tip by ~4 cm, so **this is usually the biggest term**),
- cable length uncertainty.

A good check: perturb each input in simulation by its uncertainty and see how
much the metrics move. A sim error smaller than this budget means **"agrees within
measurement accuracy"**, which is a valid and useful result.

### 4.8 Identifying the stiffness (and why the grasp matters)

The legacy study found that **a cable hanging between supports barely depends
on its stiffness**. The shape is fixed by length and geometry, so a photo
can't tell a stiff cable from a soft one. **A cable held at one end by a gripper
is different**: it's a clamped cantilever, and its shape depends on the
**gravito-bending length** `ℓ = (EI / w)^(1/3)` (`w` = weight per metre)
compared with its free length:

- gripper pointing the cable **straight down** → it just hangs straight, so the shape
  says almost nothing about `EI`;
- cable leaving the gripper **horizontally** → the droop curve is highly
  sensitive to `EI`, which makes it the best pose for identification.

Procedure:
1. Sweep `EI` (log scale, e.g. 10⁻⁶ … 10⁻² N·m²) on **one** capture and pick
   the value that minimises the cloud→sim error.
2. **Validate** with that `EI` on the **other** captures (different poses),
   without refitting. Fitting and testing on the same capture proves
   nothing.
3. Report both: fit error and validation error.

Other lessons carried over from `legacy/`:
- **Boundary conditions dominate.** In the old study, how the middle support was
  modelled changed the error more than which solver was used. Here, the grasp
  (offset, angle, does the cable slip or twist in the fingers?) plays that
  role.
- **Natural curl.** Real cables keep a coiled shape from packaging. A straight
  rest shape in sim can't reproduce it. If `e(s)` grows steadily along the
  cable in a curved way, this is a likely cause.
- **Settle fully.** Under-settled sims look like a stiffness error.

**Done when:**
- [ ] The calibration overlay (real gripper on sim gripper) has been checked and its error noted.
- [ ] Segmented cable length matches the tape measurement.
- [ ] For each capture: overlay figure, `e(s)` plot, and a metrics table.
- [ ] `EI` fitted on one capture, validated on the others, with the error budget next to it.

---

## Build order cheat-sheet

1. Newton examples run on the Mac (Franka + cable).
2. Franka loads, home pose, hand pose printed.
3. IK reaches a coded target.
4. Joystick values printed → hand moves with joystick.
5. Free cable falls and settles (no robot), arc drift < 0.5 %.
6. Cable attached to a **fixed** point in space.
7. Cable attached to the hand, robot still.
8. Cable + joystick.
9. Point cloud loaded, transformed, overlaid on the FK robot.
10. Cable segmented + centreline + length check.
11. Same scene simulated, metrics computed.
12. `EI` fit + validation + error budget → write-up.

Commit after each step. Each step is small enough to debug alone.

---

## What's in `legacy/` and what's worth reading

| file | worth reading for |
|---|---|
| `legacy/methods/hang_newton_cable.py` | `add_rod` usage, clamping bodies by zeroing mass, `read_nodes`, VBD loop, parameter choices |
| `legacy/methods/cable_config.py` | cable property dataclass, `EI`, `EA`, `ℓ = (EI/w)^(1/3)`, stiffness formulas |
| `legacy/methods/hang_common.py` | settle detection, arc-length check, output format |
| `legacy/analysis/compare.py` | metrics table, overlay and error figures |
| `legacy/image_utils/` | 2-D photo segmentation (the idea carries over to 3-D) |
| `legacy/docs/` | write-ups: Newton rod vs capsule chain, FEM on thin objects |
| `legacy/README.md` | the full previous study and its results |

The legacy scripts expect to be run from inside `legacy/` (paths in its configs
are relative to that folder), with Isaac Sim's Python.

---

## References

- Newton: <https://github.com/newton-physics/newton> (README, `newton/examples/`, docs)
- NVIDIA Warp: <https://github.com/NVIDIA/warp>
- MuJoCo Menagerie (Franka model): <https://github.com/google-deepmind/mujoco_menagerie>
- pygame joystick: <https://www.pygame.org/docs/ref/joystick.html>
- Open3D: <https://www.open3d.org/docs/>
- OpenCV hand-eye calibration: `cv2.calibrateHandEye`, `cv2.calibrateRobotWorldHandEye`
- Damped least-squares IK: S. Buss, *Introduction to Inverse Kinematics with
  Jacobian Transpose, Pseudoinverse and Damped Least Squares methods* (2004)
- Discrete elastic rods / Cosserat rods: Bergou et al., *Discrete Elastic Rods* (SIGGRAPH 2008)
