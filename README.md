# Cable simulation with a Franka robot — study guide

These are **personal study notes** for rebuilding this project from scratch. They
aren't a software README. Each part has its own guide with theory
(**Concepts**), step-by-step **Tasks**, **Exercises**, **Self-check**
questions, **Common mistakes** and **Further reading**. There's no solution
code: you write everything yourself.

The previous project (cable hanging between supports, compared with the catenary
math, a photo, and several solvers) is in [`legacy/`](legacy/), kept as a
reference.

---

## The four parts

```
 Part 1             Part 2               Part 3                Part 4
 Franka in    ──▶   drive it with  ──▶   cable held in   ──▶   real .ply point cloud
 Newton             a joystick (Mac)     the gripper            vs simulation:
                                                                how big is the error?
```

| part | guide | you learn |
|---|---|---|
| 1 | [`01_franka_in_newton/`](01_franka_in_newton/README.md) | physics-engine basics, Newton's Model/State/Solver, poses & transforms, URDF, FK, Jacobian, damped-least-squares IK |
| 2 | [`02_joystick_teleop_mac/`](02_joystick_teleop_mac/README.md) | HID input on macOS, deadzones & filters, integrating twists into poses, real-time loop design, teleop safety |
| 3 | [`03_cable_in_gripper/`](03_cable_in_gripper/README.md) | cable mechanics (EA, EI, gravito-bending length), Cosserat rods, implicit solvers & VBD, attaching to the hand, verification against the cantilever formula, measuring your cable's EI at home |
| 4 | [`04_pointcloud_vs_sim/`](04_pointcloud_vs_sim/README.md) | the PLY format, units & frames, Kabsch/ICP, RANSAC/DBSCAN, centreline extraction, real-vs-sim metrics, error budget, identifying EI honestly |

Do them **in order**. Each part ends with a "done when" list. Don't start the
next part until it passes. Debugging one layer at a time is the single biggest
time-saver (the legacy project learned that the hard way).

Each folder is also where **your code** for that part goes.

---

## Before Part 1: environment

- Python 3.10–3.12, a virtual environment (`uv` or `venv`).
- **Newton**: PyPI package `newton-physics`, imported as `newton`. Use the
  install command from the Newton README for the current version.
- On a **Mac**, Newton/Warp runs on the **CPU** (no CUDA; Metal isn't a Warp
  backend). Fine for learning, with small scenes. Heavy parameter sweeps (Part 4)
  are faster on a Linux machine with an NVIDIA GPU. The same code runs on both.
- First thing to do: list and run Newton's **bundled examples** (one robot, one
  cable). Their source code is the most accurate documentation of your installed
  version. Newton's API still changes between releases.
- Later parts add: `pygame` (joystick), `open3d` + `plyfile` (point clouds),
  `scipy`, `numpy`, `matplotlib`, `pyyaml`, `opencv-python`.
- Free GUI tools worth installing: **CloudCompare** or **MeshLab** (view and
  measure `.ply` files).

---

## Habits for the whole project

- **All physical numbers in config files** (cable length, mass, radius, EI, grasp
  offset, capture info), never hard-coded. A new cable or a new capture = a new
  YAML file.
- **Verify against something exact before trusting a result.** In the legacy
  project that was the catenary. Here it's the cantilever formula (Part 3) and a
  synthetic point cloud with a known answer (Part 4).
- **Look at every intermediate result** (plots, overlays), not only final numbers.
- **Name transforms `T_a_b`** (maps frame-b coordinates into frame a) and never
  break that convention.
- **Commit after each task.**

---

## Overall build order (cheat-sheet)

1. Newton examples run on the Mac.
2. Franka loads in its home pose; hand pose printed.
3. Your IK reaches a target.
4. Joystick values printed → a marker moves → the robot moves.
5. Free cable falls and settles; arc-length drift < 0.5 %.
6. Cantilever matches `δ = wL⁴/(8EI)`.
7. Cable attached to a fixed "hand", then to the robot, then teleop.
8. Real cable's EI measured with the table-edge test.
9. PLY inspected; data card written.
10. Synthetic PLY pipeline recovers a known EI.
11. Real PLY: frame → segment → centreline → length check.
12. Same scene simulated → metrics → error budget → EI fit + validation.

---

## What's in `legacy/` and what's worth reading

| file | useful for |
|---|---|
| `legacy/methods/hang_newton_cable.py` | complete `add_rod` + `SolverVBD` setup, clamping bodies by zero mass, `read_nodes` |
| `legacy/methods/cable_config.py` | cable properties, `EI`, `EA`, `ℓ = (EI/w)^(1/3)`, stiffness formulas |
| `legacy/methods/hang_common.py` | settle detection, arc-length check, output format |
| `legacy/analysis/compare.py` | metrics table, overlay and error plots |
| `legacy/image_utils/` | 2-D photo segmentation (the same ideas carry over to 3-D) |
| `legacy/docs/` | write-ups: Newton rod vs capsule chain; FEM on thin objects |
| `legacy/README.md` | the full previous study and its results |

Legacy scripts run from inside `legacy/`, with Isaac Sim's Python.
