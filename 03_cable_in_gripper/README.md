# Part 3 — A cable held in the Franka gripper

> Study notes. Prerequisites: Part 1 (robot + IK) and Part 2 (teleop).
> This is the most "physics" part. Take time with the Concepts.

## What you'll be able to do at the end

- Explain how a cable is modelled mathematically: stretch, bend, twist, Cosserat rods.
- Convert material properties (E, radius, mass) into simulator parameters.
- Explain why stiff cables are hard to simulate and what implicit solvers / VBD do about it.
- Attach a cable to a moving robot hand correctly.
- **Verify** a cable simulation against an analytic solution, and measure your
  real cable's bending stiffness at home.

---

## Concepts

### 1. What makes a cable a cable

A cable is a long, thin elastic body. Its shape is dominated by four effects,
from strongest to weakest:

| effect | stiffness | typical behaviour |
|---|---|---|
| **stretch** | axial stiffness `EA` | huge, so the cable is practically **inextensible** |
| **bend** | bending stiffness `EI` | small. This is what you "feel" when you bend a cable. |
| **twist** | torsional stiffness `GJ` | small, but causes loops/kinks when you twist the gripper |
| **gravity** | weight per length `w = ρ A g` | pulls it down |

with `A = π r²` (cross-section area), `I = π r⁴ / 4` (second moment of area),
`J = 2I`, `G = E / (2(1+ν))`.

Because `EA ≫ EI / L²`, the cable keeps its length almost exactly while bending
easily. This **stiffness gap** is the whole difficulty of cable simulation (§4).

> Real cables are not homogeneous rods: copper wires + insulation + jacket. `E`
> for "the cable" is an *effective* value. Measure `EI` directly (§7) rather
> than trusting a textbook `E`.

### 2. The key length scale: gravito-bending length

Compare bending with gravity:

```
ℓ = ( EI / w )^(1/3)          [m]
```

- Cable much **longer** than `ℓ`: gravity wins and it drapes like a rope (a
  catenary between supports).
- Cable much **shorter** than `ℓ`: bending wins and it sticks out like a rod.

For a typical charger cable, `ℓ` is a few centimetres to ~15 cm. The legacy
study found that in the hang-between-supports problem the shape barely depends
on `EI`. In Part 4 you'll see this is **not** true for a cable held at one
end. That's what makes the gripper experiment interesting.

### 3. Rod models: from continuum to discrete

**Cosserat rod theory**: a cable is a centreline curve `r(s)` (with `s` = arc
length) plus a **material frame** (an orientation) at each point. Strains
measure how it deforms:
- stretch/shear: how `∂r/∂s` deviates from the frame's tangent;
- bend/twist: how the frame rotates along `s` (curvature `κ₁, κ₂` and twist `τ`).

The elastic energy is quadratic in the strains:

```
E = ½ ∫ [ EA·ε² + EI·(κ₁² + κ₂²) + GJ·τ² ] ds
```

**Discretisation**: chop the cable into `N` segments. Newton's
`ModelBuilder.add_rod` makes each segment a small **capsule rigid body** (it has
position *and* orientation, i.e. the material frame) and connects neighbours with
**cable joints** that carry a stretch spring and a bend/twist spring. For segment
length `l`, the discrete stiffnesses are:

```
stretch_stiffness = E·A / l          [N/m]
bend_stiffness    = E·I / l          [N·m / rad]
```

(`legacy/methods/cable_config.py → rod_stiffness` does exactly this.)

**Rest shape**: the energy is zero in the shape the rod was *created* in. Create
it straight, and it wants to be straight. Real cables usually have a **natural
curl** from being coiled in the box. Keep this in mind as an error source in Part 4.

**Mass**: Newton computes each capsule's mass from volume × density. Set
`density = m_cable / (π r² L)` so the total mass equals the real cable's.

### 4. Why stiff cables explode, and what implicit solvers do

Recall from Part 1: an explicit integrator is stable only for `Δt < 2/ω`,
`ω = √(k/m)`. A cable segment of 1 cm and 0.3 g with a stretch stiffness of
~10⁵ N/m gives `ω ≈ 2×10⁴ rad/s`, so `Δt < 10⁻⁴ s`, i.e. hundreds of substeps per
frame. Explicit integration is hopeless.

**Implicit integration** asks instead: "which end-of-step positions `x` minimise
the total energy?"

```
x_{n+1} = argmin_x   1/(2Δt²) · ‖x − y‖²_M  +  E_elastic(x)
          with  y = x_n + Δt·v_n + Δt²·g      (where inertia alone would take it)
```

This is stable for any `Δt`. The cost is an optimisation problem per step.

**VBD (Vertex Block Descent)**, Newton's `SolverVBD`: solve that optimisation
by updating one small **block** (a vertex, or here a body) at a time. Each block
is moved to minimise the energy with its neighbours held fixed, then the next
block, and so on (Gauss–Seidel). Blocks that don't share a constraint can be
updated **in parallel**. Finding such groups is **graph colouring**, which is
why the legacy code calls `builder.color()` before `finalize()`.

**Iterations matter**: Gauss–Seidel converges gradually. Too few iterations and
the stiff stretch constraint isn't satisfied, so the cable visibly **stretches
under its own weight**. That's a solver artefact, not physics. The legacy sweep
measured (150 segments): ~100 iterations needed before stretch becomes small,
200 was the knee; accuracy depends on `substeps × iterations`; damping the
stretch constraint made things *worse*.

### 5. Damping and settling

Without damping, a simulated cable can swing almost forever. Bend damping
removes wiggles; a little global damping helps it **settle** for static
comparisons. But damping must **not change the final static shape**, only how
fast you reach it. Always check that.

"Settled" needs a definition. Use something like: tip moves less than 0.1 mm per
0.1 s for 1 s, then average the last 0.5 s. (See `SettleMonitor` in
`legacy/methods/hang_common.py`.)

### 6. Attaching the cable to the gripper

The cable is held by the fingers. Model the grip as a **clamp**: the first
segment(s) of the cable move rigidly with the hand.

**Kinematic attachment (recommended).** Make the first 1–2 capsule bodies
**massless** (zero mass and inertia, so the solver treats them as immovable).
Then *you* set their pose every substep:

```
T_world_seg0 = T_world_hand · T_hand_grasp · T_grasp_seg0
```

- `T_hand_grasp`: where the cable sits between the fingers and which way it
  leaves the hand (measure this on the real setup!).
- Also set the bodies' **velocity** consistently (finite difference of the pose
  between substeps). A body that "teleports" with zero velocity injects energy
  into the neighbouring segments, and you'll see jitter.
- Interpolate the hand pose **within** the frame across substeps instead of
  jumping once per frame.

Alternatives (for later): a fixed joint between the hand body and the cable in
one solver (only if the solver handles both robot and cable joints), or a real
friction grasp (hard, needs contact tuning, needed only to study slipping).

### 7. Contacts

Turn on in this order: cable–ground, cable–robot links, and last (optionally)
cable self-collision. Self-collision is expensive and rarely needed for a
hanging cable. The collision radius = the cable radius. Too thin and it tunnels
through things; too fat and it changes the shape.

### 8. Verification: test against math before trusting anything

The legacy project's core lesson was **compare against an exact solution
first**. Here the classic one is the **cantilever**: a rod clamped horizontally
at one end, bending under its own weight. For *small* deflection
(Euler–Bernoulli beam theory), the tip drops by

```
δ = w · L⁴ / (8 · EI)
```

Valid while `δ ≲ 0.1·L` (equivalently `w L³ / EI ≲ 1`). For large deflections,
there's no simple formula, but the shape (scaled by `L`) depends only on the
single number `w L³ / EI`. That's a good check too: any change that keeps
`w L³ / EI` fixed must give the same scaled shape (Exercise 3).

**Bonus: measure your real cable's EI at home.** Tape the cable to a table edge
so a length `L` sticks out horizontally. Measure the tip drop `δ` with a ruler.
Weigh a known length to get `w`. Then `EI = w L⁴ / (8 δ)`. Use a short overhang
so the deflection is small. Repeat for several `L` and check you get the same
`EI`. That's your validation that the formula applies.

---

## Tasks

Code goes in `03_cable_in_gripper/`.

**Task 3.1 — Free cable.** No robot. Create a straight rod (30–60 segments on the
Mac), clamp one end in mid-air (massless first segment), let it fall and settle.
Print the arc length every 0.1 s.

**Task 3.2 — Cantilever verification.** Clamp it horizontally. Compare the
simulated tip drop with `w L⁴ / (8 EI)` in the small-deflection regime.
*Done when:* they agree within a few %.

**Task 3.3 — Convergence.** Repeat 3.2 with 2× segments and 2× iterations.
Record how the answer changes. Choose settings where it changes < 1 mm.

**Task 3.4 — Attach to a fixed "hand".** Attach the cable to a fixed transform
`T_world_hand · T_hand_grasp` (no robot yet). Check that changing the grasp
angle changes the shape as expected.

**Task 3.5 — Attach to the robot.** Use the hand pose from FK. Robot still → cable settles.

**Task 3.6 — Teleop with cable.** Combine with Part 2. Swing the cable, lay it
on the ground, lift it again.

*Done when:* arc-length drift < 0.5 % at rest, cantilever matches theory,
results converged in segments/iterations, no jitter or explosions while
teleoperating.

---

## Exercises

1. **Energy plot.** Log kinetic + gravitational potential energy over time after
   dropping the cable. With no damping, should total energy be constant? What do
   you actually see, and why (numerical damping of implicit integration)?
2. **Stretch vs iterations.** Plot arc-length drift (%) vs VBD iterations
   (10, 20, 50, 100, 200, 400). Find the knee.
3. **Similarity.** Show by simulation that two cantilevers with different
   `w, L, EI` but the same `w L³ / EI` have the same shape (scaled by `L`).
   Explain this using dimensional analysis.
4. **Stability limit.** If you can, run the same cable with an explicit/semi-
   implicit solver and find the largest stable `Δt`. Compare with `2/ω`.
5. **Grasp angle sensitivity.** Rotate the grasp by 1°, 2°, 5°. How far does the
   tip of a 0.5 m cable move? (You'll need this number for Part 4's error budget.)
6. **ℓ intuition.** Plot the cantilever shape for `L/ℓ = 0.5, 1, 2, 5, 10`. Where
   does "rod" turn into "rope"?
7. **Real EI.** Measure your real cable's `EI` with the table-edge test (§8) at three
   overhang lengths. How consistent is it?

## Self-check

- Why is a cable "practically inextensible", and why does that make it hard to simulate?
- Write the formulas for `stretch_stiffness` and `bend_stiffness` of a segment.
- What does VBD minimise, and why does it need graph colouring?
- What visible symptom tells you there are too few solver iterations?
- Why must a kinematically moved body have a consistent velocity?
- What's `ℓ`, and what does `L/ℓ` tell you about a cable's shape?

## Common mistakes

- Trusting a textbook `E`: the cable's effective `EI` can be off by 10×.
- Too few iterations, so the cable stretches, and you misread it as "too soft".
- Comparing un-settled shapes (still swinging slowly).
- Changing damping and accidentally changing the static result.
- Grasp frame defined in the wrong frame (hand vs world). Check by rotating the hand 90°.
- Forgetting that the rest shape is the creation shape (creating it curved by accident).

## Further reading

- Bergou et al., *Discrete Elastic Rods*, SIGGRAPH 2008: the standard reference for discrete rods.
- Chen et al., *Vertex Block Descent*, SIGGRAPH 2024: the algorithm behind `SolverVBD`.
- Any strength-of-materials text: Euler–Bernoulli beam, cantilever under uniform load.
- `legacy/methods/hang_newton_cable.py`: a complete, commented `add_rod` + VBD
  setup from the previous project (clamping by zero mass, `read_nodes`, parameter choices).
- `legacy/docs/cable_model_comparison/`: Newton rod vs capsule chain write-up.
