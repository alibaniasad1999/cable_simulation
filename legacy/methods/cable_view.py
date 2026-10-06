"""
Live Isaac Sim viewport for solvers that run OUTSIDE the Isaac Sim engine.

WHY THIS EXISTS
---------------
Two of the methods here -- ``hang_newton_cable`` (Newton's ``add_rod`` + VBD)
and ``hang_warp`` (XPBD rod in Warp) -- drive their own solver and never hand
the cable to Isaac Sim's physics. That is deliberate: Isaac Sim 6's USD->Newton
parser cannot represent a cable fixed at BOTH ends (an articulation must be a
tree, a doubly-anchored cable is a closed loop -- see hang_newton_engine.py for
the full bisection). Newton's own ``ModelBuilder.add_rod`` has no such limit, so
the physically correct route is to run Newton directly.

The cost of that route is that there is no USD scene, and therefore nothing to
look at. ``CableView`` supplies one: purely VISUAL capsule prims in the Isaac
Sim stage, snapped onto whatever centreline the caller's solver produced. It
does no physics -- the caller's solver remains the sole source of truth -- so
turning the viewport on cannot change a benchmark number.

That is what makes "Newton in the Isaac Sim GUI" possible: Newton is the
physics, Isaac Sim is the renderer.

THREE THINGS A NAIVE VIEWPORT GETS WRONG
----------------------------------------
Learned the hard way -- a window that opens on a correct simulation and shows
nothing looks exactly like a crash:

  1. SCALE. The real cable is ~1.5 mm across. At true radius it renders as a
     near-invisible thread. ``vis_radius`` floors it at 6 mm so it reads as a
     rope. Visual only; the solver still uses the true radius.
  2. FRAMING. The default camera looks at the origin, but the cable hangs
     around z = 1 m and is under a metre wide. ``_frame_camera`` aims at the
     cable's bounding box so it fills the view.
  3. LIGHTING. With no lights the cable is a black silhouette on a black
     background. A dome light plus a key light fixes it.

The SimulationApp must be created before any ``pxr``/``omni`` import, so the
CALLER owns it (created at the very top of the script) and passes it in here.
Pass ``simulation_app=None`` to disable the viewport entirely -- every method
then behaves exactly as it did headless.
"""

from __future__ import annotations

import math

import numpy as np


def _quat_z_to(direction: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion (w, x, y, z) rotating +Z onto `direction` (unit vector)."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, direction))
    if c > 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if c < -1.0 + 1e-12:  # antiparallel: any perpendicular axis works
        return (0.0, 1.0, 0.0, 0.0)
    axis = np.cross(z, direction)
    axis /= np.linalg.norm(axis)
    half = 0.5 * math.acos(max(-1.0, min(1.0, c)))
    s = math.sin(half)
    return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


class CableView:
    """Visual-only capsule cable in the Isaac Sim stage, synced to node arrays.

    Args:
        simulation_app: a live ``SimulationApp``, or None to disable everything.
        num_nodes: number of centreline nodes (segments = num_nodes - 1).
        radius: the cable's true radius [m]; the drawn radius is floored at 6 mm.
        label: prim name under /World.
        supports: optional (K, 3) hold points, drawn as red spheres.
        vis_radius: explicit drawn radius [m], overriding the 6 mm floor.
        style: "tube" (default) draws ONE continuous BasisCurves through the
            nodes; "capsules" draws one capsule per rod segment.

    WHY "tube" IS THE DEFAULT
    -------------------------
    The rod really is a chain of capsule bodies, so drawing one capsule per
    segment is the literal picture. But the segments are only ~17 mm long
    (60 of them over a 1 m cable) while the drawn radius has to be inflated to
    ~6 mm to be visible at all -- and a 12 mm-wide, 17 mm-long capsule is
    almost a sphere. The honest-looking result is a string of beads, which
    misrepresents a cable that is physically smooth and makes the rendering
    look coarser than the simulation actually is.

    A BasisCurves tube through the same node positions stays visible at any
    width without changing its aspect ratio, so it reads as one continuous
    cable -- which is how Newton's own viewer draws it. Same nodes, same
    physics, honest picture. Use style="capsules" when you specifically want
    to SHOW the discretisation (e.g. a figure about segment count).
    """

    def __init__(
        self,
        simulation_app,
        num_nodes: int,
        radius: float,
        *,
        label: str = "cable",
        supports: np.ndarray | None = None,
        vis_radius: float | None = None,
        style: str = "tube",
    ):
        self.app = simulation_app
        self.enabled = simulation_app is not None
        self.num_nodes = num_nodes
        if not self.enabled:
            return

        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, Vt

        self._Gf = Gf
        self._Vt = Vt
        self.style = style
        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(self.stage, "/World")

        # (1) SCALE: floor the drawn radius so a 1.5 mm cable is actually visible.
        self.vis_radius = float(vis_radius) if vis_radius else max(radius, 6.0e-3)

        # (3) LIGHTING: ambient fill + a key light, else it is a black silhouette.
        UsdLux.DomeLight.Define(self.stage, "/World/DomeLight").CreateIntensityAttr().Set(1000.0)
        key = UsdLux.DistantLight.Define(self.stage, "/World/KeyLight")
        key.CreateIntensityAttr().Set(3000.0)
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(45.0, 0.0, 25.0))

        UsdGeom.Xform.Define(self.stage, f"/World/{label}")
        self._curve = None
        self._caps = []
        self._ops = []

        if style == "tube":
            # ONE prim for the whole cable. Only its `points` change per frame,
            # so this is also cheaper than moving 60 capsule transforms.
            curve = UsdGeom.BasisCurves.Define(self.stage, f"/World/{label}/tube")
            curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
            curve.CreateCurveVertexCountsAttr().Set([num_nodes])
            curve.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.0, 0.0, 0.0)] * num_nodes))
            # A single constant width keeps the tube uniform along its length.
            curve.CreateWidthsAttr().Set(Vt.FloatArray([2.0 * self.vis_radius]))
            curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            curve.CreateDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.5, 0.15)])
            self._curve = curve
        elif style == "capsules":
            for i in range(num_nodes - 1):
                cap = UsdGeom.Capsule.Define(self.stage, f"/World/{label}/seg_{i:03d}")
                cap.CreateRadiusAttr().Set(self.vis_radius)
                cap.CreateHeightAttr().Set(1.0e-3)
                cap.CreateAxisAttr().Set("Z")
                cap.CreateDisplayColorAttr().Set([Gf.Vec3f(0.9, 0.5, 0.15)])
                xf = UsdGeom.Xformable(cap.GetPrim())
                xf.ClearXformOpOrder()
                self._caps.append(cap)
                self._ops.append((xf.AddTranslateOp(), xf.AddOrientOp()))
        else:
            raise ValueError(f"style must be 'tube' or 'capsules', got {style!r}")

        # Static markers at the hold points, so the boundary conditions are visible.
        if supports is not None:
            for k, p in enumerate(np.asarray(supports, dtype=float)):
                s = UsdGeom.Sphere.Define(self.stage, f"/World/{label}_hold_{k:02d}")
                s.CreateRadiusAttr().Set(self.vis_radius * 2.0)
                s.CreateDisplayColorAttr().Set([Gf.Vec3f(0.85, 0.1, 0.1)])
                xf = UsdGeom.Xformable(s.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))

        self._camera_done = False

    # -- internals ---------------------------------------------------------
    def _frame_camera(self, nodes: np.ndarray) -> None:
        """(2) FRAMING: aim the viewport camera at the cable's bounding box.

        The 2.8x standoff leaves margin around the cable so the hold markers
        stay in shot. Framing off the bounding-box DIAGONAL (not just the span)
        keeps a deep 3-support sag in frame as well as a shallow 2-support one.
        """
        lo, hi = nodes.min(axis=0), nodes.max(axis=0)
        center = 0.5 * (lo + hi)
        span = float(np.linalg.norm(hi - lo)) + 1e-3
        eye = np.array([float(center[0]), float(center[1] - 2.8 * span), float(center[2] + 0.55 * span)])
        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=eye, target=np.asarray(center, dtype=float))
        except Exception as exc:  # framing is cosmetic; never kill a run over it
            print(f"[cable_view] camera framing failed: {exc}")
        self._camera_done = True

    # -- public API --------------------------------------------------------
    def sync(self, nodes: np.ndarray, render: bool = True) -> None:
        """Snap the capsules onto `nodes`, shape (num_nodes, 3).

        ``render=True`` pumps the app to draw a frame, which is right when the
        caller drives its own solver loop. Pass ``render=False`` when an
        isaacsim ``World`` owns stepping -- calling ``simulation_app.update()``
        in the middle of ``world.step`` invalidates its physics view.
        """
        if not self.enabled:
            return
        Gf = self._Gf
        nodes = np.asarray(nodes, dtype=float)
        if not self._camera_done:
            self._frame_camera(nodes)

        if self._curve is not None:  # tube: rewrite the points, one attribute
            self._curve.GetPointsAttr().Set(
                self._Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in nodes])
            )
        else:  # capsules: place each segment between consecutive nodes
            for i in range(min(len(self._ops), len(nodes) - 1)):
                t_op, o_op = self._ops[i]
                a, b = nodes[i], nodes[i + 1]
                d = b - a
                seg = float(np.linalg.norm(d)) or 1.0e-6
                mid = 0.5 * (a + b)
                w, x, y, z = _quat_z_to(d / seg)
                self._caps[i].GetHeightAttr().Set(max(seg - 2.0 * self.vis_radius, 1.0e-4))
                t_op.Set(Gf.Vec3d(float(mid[0]), float(mid[1]), float(mid[2])))
                o_op.Set(Gf.Quatf(w, x, y, z))

        if render:
            self.app.update()

    def hold_open(self, nodes: np.ndarray, message: str = "") -> None:
        """Keep the window alive after the run so the result can be inspected.

        Returns when the user closes the Isaac Sim window.
        """
        if not self.enabled:
            return
        if message:
            print(message, flush=True)
        while self.app.is_running():
            self.sync(nodes)

    def save_usd(self, path: str) -> None:
        """Export the stage (useful for checking a run on a machine with no display)."""
        if self.enabled:
            self.stage.Export(path)

    def screenshot(self, path: str) -> None:
        """Capture the viewport to a PNG, after letting RTX converge."""
        if not self.enabled:
            return
        try:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            for _ in range(60):
                self.app.update()
            capture_viewport_to_file(get_active_viewport(), path)
            for _ in range(20):
                self.app.update()
        except Exception as exc:
            print(f"[cable_view] screenshot failed: {exc}")
