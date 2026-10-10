# Part 5 (optional) — Driving the Franka with a joystick on macOS

> **Optional.** Not needed for the main goal (Parts 1–4). Do it when Parts 1–3
> work, to pose the simulated robot by hand or drive it live.
> Prerequisite: Part 1 (IK moves the hand to a target).

## Where it plugs into the reference example

In [`reference/example_franka_cable_ik_pick_place.py`](../reference/example_franka_cable_ik_pick_place.py),
the hand target comes from **keyframes**: `update_ik_targets()` (`L438–L462`)
interpolates a table and writes the result into `ik_target_positions`,
`ik_target_rotations` and `finger_pos_buf`. The joystick simply **replaces that
one function**:

```
update_ik_targets():                     # called once per frame, before simulate()
    read joystick → (v, ω, gripper button)
    target_pos ← target_pos + v·frame_dt          (clamped to a safe box)
    target_rot ← small rotation(ω·frame_dt) · target_rot
    finger     ← GRIP_OPEN or GRIP_HOLD from the button
    write them into the same three arrays
```

Everything else (IK solve, PD targets, cable, coupling) stays unchanged. On the
Mac there's no CUDA graph, so `simulate()` runs as plain Python every frame and
new targets are picked up immediately. (On a GPU with graph capture, the targets
must be written into the *same* arrays the graph was captured with, which is why
the example uses a kernel to fill them in place.)

The sections below explain each piece.

## What you'll be able to do at the end

- Read a gamepad or SpaceMouse on macOS and understand what the numbers mean.
- Turn noisy stick values into smooth, safe end-effector motion.
- Integrate velocities into a pose correctly, including rotations.
- Structure a real-time loop (input → control → physics → render) that runs
  reliably on a Mac.

---

## Concepts

### 1. How a joystick talks to your program

Gamepads and SpaceMice are **HID** (Human Interface Device) devices. The OS
delivers their raw reports. A library turns them into:

- **axes**: floats, usually in `[−1, 1]` (sticks, triggers),
- **buttons**: 0/1,
- **hats**: the D-pad, as (x, y) in {−1, 0, 1}.

Libraries to use on macOS:
- **`pygame`** (built on SDL2): works with Xbox, PlayStation and most generic pads
  over USB or Bluetooth. Initialise only the joystick module, no game window
  needed.
- **`pyspacemouse`** (needs `hidapi`, from Homebrew) for a 3Dconnexion
  SpaceMouse. A SpaceMouse gives all 6 axes at once, which is perfect for 6-DOF
  teleoperation.

**Polling vs events.** You can either ask "what is axis 0 now?" every frame
(polling) or receive "axis 0 changed" messages (events). Polling is simpler for
continuous control. But SDL still needs its **event queue pumped every
frame**, or the values freeze. That is the #1 "my joystick doesn't work" bug.

**Every controller numbers its axes differently.** First task: print all axes
and buttons while you press things, and write the mapping into a config file.
Triggers often rest at `−1` instead of `0`.

### 2. macOS specifics

- macOS requires **windows and input event handling on the main thread**
  (Cocoa rule). Don't put the viewer or pygame in a background thread. Use one
  loop on the main thread.
- macOS may ask for **Input Monitoring** permission (System Settings → Privacy &
  Security) for your terminal/IDE, especially for HID devices like the SpaceMouse.
- Newton on the Mac runs on the CPU (Warp has no Metal backend). Keep the scene
  small so the loop keeps up.
- A real Franka is controlled through `libfranka`, which needs a **Linux PC with
  a real-time kernel**. A Mac can't control the real robot directly. If you
  later teleoperate the real robot, the Mac sends commands over the network to
  that Linux PC. Design your code now so the joystick part only produces a
  plain command message (see §6).

### 3. Shaping the raw signal

Raw stick values are noisy and never exactly zero at rest. Process them in this
order:

**a) Deadzone.** A *scaled radial* deadzone treats the stick as a 2-D vector `u`:

```
if |u| < d:  u = 0
else:        u = (u / |u|) · (|u| − d) / (1 − d)       (d ≈ 0.1)
```

Unlike clipping each axis separately, this has no "jump" at the edge and doesn't
make diagonal motion square-shaped.

**b) Response curve.** `u ← sign(u)·|u|^γ` with `γ ≈ 2`, so small deflections give
fine motion and full deflection gives full speed.

**c) Scaling to physical units.** `v = u · v_max` (e.g. `v_max = 0.15 m/s`),
`ω = u · ω_max` (e.g. `0.6 rad/s`).

**d) Smoothing (low-pass filter).** First-order exponential filter:

```
y ← y + α (x − y),     α = Δt / (τ + Δt)
```

`τ` is the time constant (e.g. 0.1 s). Bigger `τ` is smoother but laggier. You'll
measure this trade-off in Exercise 2.

**e) Rate limit (acceleration limit).** Cap how much `v` can change per frame:
`|v_new − v_old| ≤ a_max·Δt`. Real robots have acceleration limits, and the real
Franka rejects commands that violate them. Practising this now pays off later.

### 4. From velocity command to target pose

Velocity (rate) control: the stick commands a **twist**, and you **integrate**
it into a target pose every frame:

```
p_target ← p_target + v · Δt
R_target ← Exp(ω · Δt) · R_target          (world-frame ω)
R_target ← R_target · Exp(ω · Δt)          (hand-frame ω)
```

`Exp(θ)` is the rotation by angle `|θ|` about axis `θ/|θ|` (Rodrigues' formula).
With quaternions, multiply by the small-rotation quaternion and **renormalise**
every frame, or numerical drift slowly makes `|q| ≠ 1`.

**Which frame is `v` in?**
- *World frame*: "stick up = robot moves up", whatever the hand orientation.
  Intuitive for gross motion.
- *Hand (tool) frame*: "stick forward = move along the gripper's pointing
  direction". Best for fine alignment with a cable. Convert with
  `v_world = R_world_hand · v_hand`.

Give yourself a button to toggle between them.

Then IK (Part 1) turns the target pose into joint angles. Important: start each
IK solve from the **current** joint angles (*warm start*). Then it converges in
1–3 iterations, and the arm doesn't jump to a different elbow configuration.

### 5. Safety habits, even in simulation

The habits you build in simulation are the ones you'll have on the real robot:

- **Deadman button**: the robot moves only while a button is held. Release = stop.
- **Workspace box**: clamp `p_target` inside a safe box in front of the robot.
- **Don't run away**: if IK can't reach the target (error stays large), stop
  integrating the target further. Otherwise it drifts far outside reach and the
  robot snaps when it comes back.
- **Joint-velocity limit**: scale down `Δq` if any joint would move faster than its limit.
- **Home button**: smoothly return to the home pose.

### 6. The real-time loop

```
every frame (target 60 Hz):
    pump events, read axes/buttons
    shape signal (deadzone → curve → scale → filter → rate limit)
    integrate target pose (+ clamp to workspace)
    IK (warm start) → joint targets
    physics substeps
    render
    measure frame time
```

Separate the code into three layers with simple data between them:

```
InputDevice  ──(raw axes, buttons)──▶  TeleopMapper ──(v, ω, gripper, flags)──▶  RobotController
```

Then you can swap a gamepad for a SpaceMouse, a recorded file (for repeatable
tests) or a UDP message from another machine, without touching the robot code.
Recording the `(v, ω)` stream to a file and replaying it is also a great
debugging tool.

**Timing:** measure the real frame time each loop. If the sim can't keep up,
`Δt` in the integration must be the **real** elapsed time (or you must slow
down). Otherwise the robot moves slower than commanded and teleop feels laggy.

---

## Tasks

Code goes in `02_joystick_teleop_mac/`.

**Task 2.1 — Read the device.** Print all axes/buttons at 10 Hz. Write the
mapping (which axis is which, rest values, inversions) into a YAML file.

**Task 2.2 — Signal shaping.** Implement deadzone → curve → scale → filter →
rate limit, as functions you can test without a joystick.

**Task 2.3 — Teleop a marker.** Before the robot: move a small sphere in the
viewer with the joystick (position + orientation). This isolates input problems
from IK problems.

**Task 2.4 — Teleop the Franka.** Target pose → IK → robot. Add the deadman
button, workspace box, gripper button, home button, world/hand frame toggle.

**Task 2.5 — Record button.** Pressing a button saves the current joint angles
and hand pose to a file. Handy for reproducing poses later (Part 4).

*Done when:* the hand moves smoothly in the expected direction, stops dead when
you release, can't leave the box, never jumps between elbow configurations, and
the loop runs at a steady rate (write the measured fps down).

---

## Exercises

1. **Deadzone comparison.** Plot the output of an axial deadzone vs the radial
   deadzone for a stick moved in a circle. Explain the difference.
2. **Filter trade-off.** Log raw and filtered axis values while flicking the
   stick. Plot both for `τ = 0.02, 0.1, 0.3 s`. Measure the lag (time to reach
   63 % of a step) and compare with `τ`.
3. **Rotation integration.** Integrate a constant `ω` about z for 10 s two ways:
   (a) adding Euler angles, (b) the exponential map. Then do it with `ω` having
   two components. Which one stays correct? Check `|q|` with and without
   renormalisation.
4. **Latency budget.** Estimate the delay from stick to visible motion: input
   polling + filter lag + frame time + render. Which term dominates?
5. **Replay.** Record a 20 s teleop session as `(t, v, ω, buttons)`, replay it,
   and check the final hand pose is identical. If it isn't, find the
   non-determinism.

## Self-check

- Why must the SDL event queue be pumped even if you only poll axes?
- What happens without a "don't run away" rule when the target leaves the workspace?
- Write the formula that converts a hand-frame velocity into a world-frame velocity.
- Why warm-start IK from the current joints?
- Why can't the Mac run the real Franka directly, and how would your design adapt?

## Common mistakes

- Axis numbering assumed from another controller; triggers resting at −1.
- Integrating with a fixed `Δt` while the real loop runs slower.
- Forgetting to renormalise the quaternion.
- Using hand-frame `ω` with the world-frame update formula (rotations feel "wrong"
  after the hand has turned).
- GUI or pygame in a background thread on macOS: random crashes or a frozen window.

## Further reading

- pygame joystick docs: <https://www.pygame.org/docs/ref/joystick.html>
- *Modern Robotics*, Ch. 3: exponential coordinates of rotation, twists.
- franky / libfranka docs, for when you move to the real robot (motion
  generators, velocity and acceleration limits).
