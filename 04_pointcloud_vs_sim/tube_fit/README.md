# Cable centreline from a point cloud (tube fit)

Takes a scan (`.ply`) that contains a cable and returns the cable's 3-D
centreline, radius and length, plus the cable points separated from everything
else. This is the "fit a tube" method from Part 4 §7.

```bash
# from the repository root
.env/bin/python3 04_pointcloud_vs_sim/tube_fit/fit_cable_tube.py data/Ethernet.ply
```

No clicking is needed: the longest thin tube in the scan is taken as the cable.
If the scan has several, point at the right one with `--pick` (click it in a
window) or `--seed X Y Z`.

## Method

To write your own version, follow [GUIDE.md](GUIDE.md): the maths, the library
calls and the numbers to expect at each stage.

1. **Find tube-like points.** On a cylinder the surface normals fan out across
   the axis and never along it. Around each point, the direction the normals do
   not span is the local axis, and the local radius follows in closed form.
   Flat surfaces and corners fail this test.
2. **Radius.** The most common local radius is the cable's.
3. **March.** From a seed on the cable, step along the axis. At each step keep
   the raw points that sit one radius from the axis with their normal pointing
   outward, and re-centre on them. The normal test rejects a second strand lying
   against this one, so the march follows the cable through a self-crossing.
4. **Tube fit.** A cubic B-spline centreline and one radius are adjusted until
   every cable surface point is at distance `r` from the curve (robust least
   squares on the full-resolution points).
5. **Trim.** Ends that are poorly supported or bend tighter than 4 cable radii
   are cut off (plug, gripper fingers), then the fit is repeated.

## Outputs

Written to `results/<name>_tube_fit/`:

| file | content |
|---|---|
| `centerline.csv` | `s_m, x_m, y_m, z_m, tx, ty, tz, supported`: 151 nodes at equal arc length, in **metres**, with the unit tangent. `supported = 0` marks nodes inside a hidden stretch. |
| `summary.json` | radius, length, residuals, end positions, hidden stretches |
| `cable_points.ply` | the segmented cable points, in the scan's own units and frame |
| `centerline.ply` | the centreline as red points every 0.5 mm, same units and frame |
| `preview.png` | three views: scan grey, cable points green, centreline red |

Load the scan, `cable_points.ply` and `centerline.ply` together in CloudCompare
to inspect the result.

## Result on `data/Ethernet.ply`

| | |
|---|---|
| radius | 3.60 mm (diameter 7.20 mm) |
| length of visible cable | 892.7 mm |
| cable points | 24,096 of 162,577 |
| fit residual (distance of points from the tube surface) | 0.078 mm RMS, 0.043 mm median |
| trimmed at the ends | 36 mm and 24 mm (plug and gripper fingers) |

## Checked against a known answer

`synthetic_test.py` builds a fake scan with a known centreline: a looping cable
whose strands are 0.3 mm apart where they cross, seen from one side only, with
0.05 mm noise, 4° normal noise and clutter (plate, fat cylinder, wall).

```bash
cd 04_pointcloud_vs_sim/tube_fit && ../../.env/bin/python3 synthetic_test.py
```

| | error |
|---|---|
| centreline, mean | 0.007 mm |
| centreline, max | 0.09 mm |
| radius | 0.001 mm |
| length | 0.02 % |

## What this does not do

- **Frame.** The centreline is in the scanner's frame. It is not aligned to
  gravity or to the robot hand; that is Part 4 §5.
- **Hidden length.** The 892.7 mm is the cable the scanner saw and that looks
  like a tube. The part inside the fingers and the plug are not included; add
  them from a tape measurement before comparing with a simulation.
- **Accuracy on real data.** 0.078 mm is how well the points fit a tube, not
  the error of the centreline. The synthetic test shows the method is accurate
  when the scan is; scanner calibration error is on top and not measured here.
- **Normals.** The method relies on the normals in the file. If a PLY has none
  they are estimated, which is less accurate and has not been tested.

## Options

| option | default | meaning |
|---|---|---|
| `--out DIR` | `results/<name>_tube_fit` | output folder |
| `--units mm\|m\|auto` | `auto` | units of the PLY (auto: by its size) |
| `--radius-mm R` | estimated | fix the cable radius |
| `--seed X Y Z` / `--pick` | longest tube | choose which cable |
| `--nodes N` | 151 | nodes in `centerline.csv` |
| `--max-gap-mm G` | 20 | longest hidden stretch to bridge |
| `--min-bend-radii K` | 4 | ends bending tighter than `K·r` are trimmed |
| `--show` | off | open a 3-D view of the result |

Needs `numpy`, `scipy`, `open3d` (all already in `.env`).
