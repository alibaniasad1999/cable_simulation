# Build it yourself: cable centreline by tube fitting

> Study notes for writing your own version of `fit_cable_tube.py`.
> Each stage gives the idea, the maths, the library calls to look up, and the
> numbers you should get on `data/Ethernet.ply` so you can check yourself.
> The finished script is in this folder. Look at it only when stuck.

**Input:** a `.ply` with points, normals and (optionally) colours.
**Output:** the cable's centreline as an ordered 3-D curve, its radius, its
length, and which points of the scan belong to the cable.

**The one idea behind everything:** a cable is a tube. A point is on the cable
if it sits at distance `r` from a smooth curve, with its surface normal pointing
straight away from that curve. Every stage below is that sentence applied at a
different scale.

You need `numpy`, `scipy` and `open3d`. Work in **millimetres** throughout.

---

## Stage 0. Load and downsample

- Read the file with `open3d.io.read_point_cloud`. Get points, normals, colours
  as numpy arrays.
- Find the units from the bounding box: extents of a few hundred mean
  millimetres, below one mean metres. Convert to millimetres.
- Make a **downsampled copy** with `voxel_down_sample(0.8)` for stages 1 to 3.
  Open3D averages the normals inside each voxel, so **re-normalise them to
  length 1**. Keep the full-resolution cloud for stage 4.

**Check:** 162,577 points; extent about 228 × 373 × 331 mm; about 64,000 points
after downsampling.

---

## Stage 1. Which points lie on a tube?

For each point, take its neighbours within a ball of radius `ρ` (start with
`ρ = 5` mm). Use `scipy.spatial.cKDTree` and `query_ball_point`. Skip points
with fewer than 12 neighbours.

### 1a. The local axis, from the normals

On a cylinder every normal is perpendicular to the axis. So the axis is the one
direction the neighbours' normals have nothing in common with.

```
M = (1/k) Σ nⱼ nⱼᵀ          3×3 matrix from the k neighbour normals
eigenvalues  μ₀ ≤ μ₁ ≤ μ₂    (numpy.linalg.eigh returns them in this order)
axis a = eigenvector of μ₀
```

What the eigenvalues tell you (they sum to 1):

| surface | μ₀ | μ₁ | μ₂ |
|---|---|---|---|
| plane | ≈ 0 | ≈ 0 | ≈ 1 |
| tube | ≈ 0 | clearly > 0 | the rest |
| corner, sphere, clutter | > 0 | > 0 | > 0 |

### 1b. The local radius, in closed form

Look along the axis: project the neighbour points and normals onto the plane
across `a`.

```
uⱼ = pⱼ − (pⱼ·a) a          projected point
mⱼ = nⱼ − (nⱼ·a) a          projected normal
```

On a circle of radius `r` with centre `c`, every point satisfies
`uⱼ − r mⱼ = c`. Subtract the means (`ũ = u − ū`, `m̃ = m − m̄`) to remove the
unknown `c`, and solve for `r` by least squares:

```
r = Σ ũⱼ·m̃ⱼ / Σ m̃ⱼ·m̃ⱼ
scatter = sqrt( mean ‖ũⱼ − r m̃ⱼ‖² )     how well the patch is a circle
```

### 1c. Keep the tube-like points

```
μ₀ < 0.05   and   μ₁ > 0.04   and   scatter < 0.35 |r|
```

If most of the `r` values come out **negative**, the file's normals point
inward: flip all normals and all `r`.

**Check:** colour the kept points and look at them. The cable should light up
almost completely, with a few stray patches on rounded parts of the gripper.

---

## Stage 2. The cable's radius

The cable is the longest thin tube in the scene, so its radius is the most
common value among the `r` from stage 1.

- Histogram `log(r)` (60 bins between 0.3 and 40 mm), smooth it lightly, take
  the peak.
- The radius is the **median** of the values within 0.75 to 1.33 times the peak.
- If `ρ` was far from `2r`, repeat stage 1 with `ρ = 2r` and redo this.

Then select the cable candidates and collapse them onto the axis:

```
keep points with  |r_local − r| < 0.4 r
qⱼ = pⱼ − r nⱼ        each surface point moved inward by one radius
```

The `q` points now form a thin thread along the cable's centre.

**Check:** `r ≈ 3.9` mm on both passes (the final fit corrects it to 3.60);
about 10,000 candidate points. Plot the `q` points: a thread, not a tube.

---

## Stage 3. March along the cable

### 3a. Where to start

- Connect `q` points closer than `0.6 r` (`cKDTree.query_pairs`, then
  `scipy.sparse.csgraph.connected_components`).
- Take the largest component. Its "middle" point (median position along the
  component's main direction) is the seed.
- Start position `x₀` = mean of the `q` points within `1.5 r` of the seed.
- Start direction `d₀` = mean of their local axes. An axis has no sign, so flip
  each one to agree with the seed's axis before averaging.

### 3b. One step

```
x ← x + r·d                       step one radius forward
repeat twice:
    S = raw points near x that fit a tube section there   (test below)
    if fewer than 6 points: no support at this step
    centre  = mean over S of (p − r n)
    x ← x + (centre − x), without its component along d   (move sideways only)
    a = eigenvector of the smallest eigenvalue of Σ n nᵀ over S
    d ← normalise( d + 0.5 · a·sign(a·d) )
```

**The tube-section test** for a point `p` with normal `n`, given `x` and `d`:

```
v     = p − x
along = v·d                      must satisfy |along| < r
perp  = v − along·d
dist  = ‖perp‖                   must satisfy |dist − r| < 0.35 r
perp·n / dist                    must be > 0.5   (normal points outward)
```

The last line is what lets the march pass through a **self-crossing**: the other
strand's points are nearby, but their normals point the wrong way.

### 3c. Gaps and stopping

- If a step has no support, keep going straight and add `r` to a gap counter.
  Loosen the distance tolerance a little as the gap grows
  (`0.35 r + 0.1·gap`).
- Stop when the gap exceeds 20 mm. Remove the unsupported nodes at the tail.
- March once with `+d₀` and once with `−d₀`, then join:
  `reversed(backward) + [x₀] + forward`.

**Check:** about 245 nodes and 951 mm, running from the gripper, through the
crossing, to the plug. If it stops at the crossing (about 754 mm), your normal
test is missing or you are marching on the pre-filtered `q` points instead of
the raw points.

---

## Stage 4. Fit the tube

The marched nodes are close. This stage makes them accurate, using every
full-resolution point.

### 4a. Starting curve

- Parameter of each node: cumulative chord length, scaled to `[0, 1]`.
- Cubic B-spline with **clamped uniform knots** and one control point about
  every 8 mm: `scipy.interpolate.make_lsq_spline`.
- Look up: "clamped knot vector", "Schoenberg-Whitney conditions" (why you
  can't have more control points than data allows).

### 4b. Which points belong to the tube (repeat every iteration)

- Sample the curve every 0.25 mm. Put the samples in a KD-tree.
- For each scan point find the nearest sample: distance `d`, radial direction
  `e = (p − sample)/d`, parameter `t`.
- Keep a point if `|d − r| < band` and `e·n > 0.3`.
- **Drop points whose nearest sample is the first or the last one.** They lie
  beyond the ends of the curve.

### 4c. One Gauss-Newton step

Unknowns: the control points `C` (`n_c × 3`) and the radius `r`.
Residual of point `i`: `ρᵢ = dᵢ − r`.

With the parameters `tᵢ` held fixed, the curve point is `c(tᵢ) = Σₖ Bₖ(tᵢ) Cₖ`,
so the derivatives are

```
∂ρᵢ/∂Cₖ = −Bₖ(tᵢ) eᵢ          ∂ρᵢ/∂r = −1
```

Get the basis matrix `B` from `BSpline.design_matrix(t, knots, 3)`. The
Jacobian is `J = [ −eₓ·B | −e_y·B | −e_z·B | −1 ]`.

Robust weights (so stray points don't pull the curve):

```
σ  = 1.4826 · median|ρ − median(ρ)|        robust noise level
wᵢ = (1 − (ρᵢ / 4.685σ)²)²   if |ρᵢ| < 4.685σ,  else 0     Tukey biweight
```

Smoothness, which only matters where there are no points:

```
D₂ = second-difference matrix on the control points,  K = D₂ᵀD₂
λ  = 1e-5 · Σw / n_c
```

Solve and update:

```
(JᵀWJ + λK) δ = −JᵀWρ − λK·C        (K applied to each of x, y, z)
C ← C + δ_C,   r ← r + δ_r
```

Iterate 12 times, redoing 4b each time with `band = max(4σ, 0.08 r)`. Start
with `σ = 0.15 r`.

**Check:** `r` settles at 3.60 mm; residual RMS about 0.08 mm; about 24,000 to
25,000 cable points.

---

## Stage 5. Trim the ends

The plug and the gripper fingers are not tubes. The curve wanders there and
bends tighter than a cable can.

- Cut the curve into 2 mm bins by arc length. Per bin: number of inlier points,
  and the largest curvature.
- Curvature from the dense samples: `κ = ‖dT/ds‖`, with `T` the unit tangent.
- A bin is **good** if its count is at least half the median count **and**
  `κ < 1/(4r)` (bend radius above 4 cable radii).
- From each end, move inward to the first run of good bins `4r` long. Cut there.
- Re-fit (stage 4) on the trimmed curve.
- Finally set the ends at the smallest and largest arc length that actually has
  inlier points. Without this the curve overshoots the cable by a millimetre or
  two.

**Check:** 36 mm and 24 mm trimmed; final length 892.7 mm; minimum bend radius
above 20 mm everywhere.

---

## Stage 6. Export

- Resample at **equal arc length** (151 nodes): interpolate on the dense
  samples' cumulative length.
- Tangents from the spline's derivative, normalised.
- The cable points are the inliers of the last 4b pass, between the trimmed
  ends. Save them in the scan's own units and frame so they overlay it.
- Save the centreline in metres for the simulation.

---

## Test on a known answer first

Do this **before** trusting the real result (Part 4 §12).

1. Choose a curve with a loop, for example a prolate cycloid lifted in z:
   `x = aφ − b sin φ, y = −b cos φ, z = kφ` with `b > a`. Pick `k` so the two
   strands are `2r + 0.3` mm apart where they cross.
2. Sample points on the tube surface around it. The outward direction is the
   normal.
3. Keep only points whose normal faces one of two view directions.
4. Add 0.05 mm position noise, about 4° normal noise, and clutter: a plate, a
   fat cylinder, a wall.
5. Run your pipeline. Compare your centreline with the true one using a KD-tree.

**Targets** (what the reference reaches): centreline mean error 0.007 mm, max
0.09 mm, radius error 0.001 mm, length error 0.02 %.

---

## Mistakes I made, so you don't have to

| symptom | cause | fix |
|---|---|---|
| march stops at the self-crossing | marched on the pre-filtered `q` points; they vanish where two strands mix | march on raw points with the tube-section test (3b) |
| curve 1 to 8 mm too long | spline extends past the last real points | set ends from the inliers' arc-length range (stage 5, last step) |
| sharp hook at each end | fit runs into the plug / fingers | curvature and support trim, then re-fit |
| radius comes out negative | normals point inward | flip normals |
| centreline sits on the cable surface | averaged surface points instead of `p − r n` | always collapse by the radius |
| nothing is tube-like | `ρ` much smaller than `r`: the patch looks flat | use `ρ ≈ 2r` |

## Self-check questions

- Why is the axis the eigenvector of the **smallest** eigenvalue of `Σ n nᵀ`?
- Why must you subtract the means before solving for `r`?
- Why does the normal test separate two touching strands when distance alone can't?
- Why is the radius better determined by the global fit than by one local patch?
- What does the 0.08 mm residual tell you, and what does it not tell you?
- Why does the smoothness term barely change the curve where there are points?

## Search terms

- Gauss map, normal covariance, cylinder axis from normals
- cylinder fitting point cloud normals
- `scipy.spatial.cKDTree`: `query`, `query_ball_point`, `query_pairs`
- B-spline least squares, clamped knot vector, `make_lsq_spline`, `BSpline.design_matrix`
- Gauss-Newton, iteratively reweighted least squares (IRLS), Tukey biweight, MAD
- curve reconstruction from point cloud, deformable linear object perception
- curvature of a space curve, arc-length parametrisation
