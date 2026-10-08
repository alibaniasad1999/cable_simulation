# Cable simulation with a Franka robot — study guide

Personal study notes for rebuilding this project from scratch. They aren't a
software README. Each part tells you **which Newton calls to use, what every
argument means and how to choose its value**, with tasks and checks. Newton
handles the physics and math. These notes are about using it correctly.

**Main goal:** understand and rebuild the reference example
[`reference/example_franka_cable_ik_pick_place.py`](reference/example_franka_cable_ik_pick_place.py)
(a Franka picking up a cable in Newton), then use the same building blocks to
**simulate a real cable held in a Franka hand and compare it with its `.ply` point
cloud**.

The previous project (cable hanging between supports vs catenary math, photo,
several solvers) is in [`legacy/`](legacy/), kept for reference.

---

## Parts

| part | folder | what you build | Newton pieces you learn |
|---|---|---|---|
| 1 | [`01_franka_in_newton/`](01_franka_in_newton/README.md) | floor + Franka, moved with IK | example skeleton, `ModelBuilder`, `add_ground_plane`, `add_urdf`, `joint_q` / `joint_target_q`, PD gains, gravity compensation, `SolverMuJoCo`, `newton.ik`, TCP offset, keyframes, kinematic robot |
| 2 | [`02_cable/`](02_cable/README.md) | a cable: size, segments, mass, elasticity | `Rod.create_straight`, `add_rod`, `ShapeConfig`, choosing `radius` / `segment_count` / `density` / `stretch_stiffness` / `bend_stiffness` / damping, `SolverVBD`, fixing and moving an end, reading the shape back |
| 3 | [`03_franka_holds_cable/`](03_franka_holds_cable/README.md) | the gripper holds the cable | kinematic attachment (Option K), `SolverCoupledProxy` (Option C, the example), split collision pipelines, gripper tuning, multiple worlds |
| 4 | [`04_pointcloud_vs_sim/`](04_pointcloud_vs_sim/README.md) | real `.ply` vs simulation | PLY format, units & frames, aligning via the gripper, cable extraction, centreline, metrics, error budget, identifying `bend_stiffness` |
| side quest | [`05_side_quest_joystick_mac/`](05_side_quest_joystick_mac/README.md) | drive the Franka with a gamepad on macOS | replaces the example's keyframe function |

**Working now — the Ethernet cable vs its scan.** `configs/ethernet_cat6.json`
holds every number; two scripts do the rest:

```bash
# 1. scan -> centreline (once):          results/Ethernet_tube_fit/
python 04_pointcloud_vs_sim/tube_fit/fit_cable_tube.py Data/Ethernet.ply
# 2. Franka + cable from the JSON, settle, export (add --viewer gl to watch, --sweep for all EI values)
python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --sweep
# 3. compare with the scan:              results/ethernet_cat6/scan/compare/
python 04_pointcloud_vs_sim/compare_to_scan.py --config configs/ethernet_cat6.json
```

How it works and how to read the numbers:
[`03_franka_holds_cable/README.md` § Ethernet scene](03_franka_holds_cable/README.md#the-ethernet-scene-working-code)
and [`04_pointcloud_vs_sim/README.md` § Comparing](04_pointcloud_vs_sim/README.md#comparing-the-ethernet-scan-with-the-simulation).

Do Parts 1 → 4 **in order**. Each one ends with tasks and "check yourself"
questions. Your code for each part goes in its folder. Write it yourself and look
at the reference only when stuck.

---

## Setup

- Python 3.10–3.12 in a virtual environment (`uv` or `venv`).
- Newton: PyPI package **`newton`** (the old name `newton-physics` is now an empty
  stub that only says "renamed"): `pip install "newton[examples]"`. The code here
  was tested with `newton 1.6.1` + `warp-lang 1.18.0`. `SolverMuJoCo` also needs
  `mujoco` / `mujoco_warp`.
- First run the original:
  `python -m newton.examples franka_cable_ik_pick_place` (and `--help`).
- **On a Mac**, Warp runs on the **CPU** (no CUDA). CUDA graph capture is skipped
  automatically. Use fewer segments/substeps if it's slow. Large parameter sweeps
  (Part 4) are faster on a Linux + NVIDIA machine.
- Newton changes between versions, and the coupled solver lives in
  `newton.solvers.experimental`. If a call in these notes doesn't exist, the copy
  of this example **inside your installed Newton** and `help(...)` on the
  function are the truth.
- Part 4 adds: `open3d`, `plyfile`, `scipy`, `numpy`, `matplotlib`, `pyyaml`.
  Free viewer for `.ply` files: CloudCompare or MeshLab.

---

## Build order (cheat-sheet)

1. The reference example runs on your machine.
2. Floor + Franka standing in a pose (Part 1).
3. IK moves the TCP to a target; keyframes move it through a sequence.
4. Cable on the floor; cable hanging from a fixed point (Part 2).
5. Segment count, iterations and stiffness chosen with small tests; real cable's
   size, mass and table-edge stiffness measured.
6. Cable glued to the gripper (Option K); hand rotates, cable follows (Part 3).
7. The example's pick-and-place rebuilt with the coupled solver (Option C).
8. `.ply` inspected: units, frame, what's visible (Part 4).
9. Pipeline tested on a synthetic `.ply` made from your own simulation.
10. Real `.ply`: align via gripper → extract cable → centreline → length check.
11. Same scene simulated → metrics → error budget → `bend_stiffness` fit + validation.

---

## Habits

- **All physical numbers in a config file** (cable length, radius, mass, stiffness,
  grasp, capture info). A new cable = a new YAML, not edited code.
- **Keep index lists** (`franka_bodies`, `cable_bodies`, `gripper_bodies`, ...).
  Almost every Newton bug is a wrong index.
- **Test one thing at a time**, and look at the viewer and plots, not only numbers.
- **Commit after each task.**

---

## Legacy files worth reading

| file | useful for |
|---|---|
| `legacy/methods/hang_newton_cable.py` | `add_rod` + `SolverVBD`, fixing bodies by zero mass, `read_nodes` |
| `legacy/methods/cable_config.py` | cable properties, stiffness from material (`E·A/l`, `E·I/l`) |
| `legacy/methods/hang_common.py` | settle detection, arc-length check |
| `legacy/analysis/compare.py` | comparison table and overlay plots |
| `legacy/README.md` | the previous study and its results |
