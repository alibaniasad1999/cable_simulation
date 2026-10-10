"""The fitted cable in the Isaac Sim viewport: Newton solves it, Isaac Sim only draws it.

    ~/isaacsim/python.sh 03_franka_holds_cable/isaac_cable_scene.py --config results/ethernet_cat6_fit/best_config.json

Newton runs in one of two places (--newton, default auto):

  * here    inside this Isaac Sim process, as legacy/methods/hang_newton_cable.py --gui
            did. Needs a Newton that can run ethernet_scene.py (newton.Rod, Newton 1.6)
            importable from Isaac Sim's Python.
  * worker  in a second process under your own Newton Python, which sends the segment
            poses here after every frame. Used when Isaac Sim's Newton is too old
            (Isaac Sim 6.0 bundles 1.2). Its Python is --newton-python, else
            $NEWTON_PYTHON, else the repository's .env / .venv, else `python3`.

Either way it is the same scene as ethernet_scene.py (its setup(), SolverVBD); this is
not Isaac Sim's own Newton engine, which does not support cables.

What is in the Isaac stage:
  * cable   one capsule per segment at its real radius, with NO physics on it: each
            frame it is moved to the pose Newton computed (twist included);
  * scan    the scanned centreline as a red line, as in the Newton GL viewer;
  * Franka  the one in the Newton scene (kinematic, placed by IK). It is drawn with
            Isaac's franka.usd (a Panda: same link frames as Newton's FR3), each link
            put at the pose Newton holds it in. --no-robot leaves it out.

No physics engine runs in Isaac Sim at all (no PhysX, and not Isaac's own Newton
engine): the stage is only a picture of the Newton state.

Nothing is written: for the run folder and the comparison with the scan use
ethernet_scene.py and 04_pointcloud_vs_sim/compare_to_scan.py.

The Isaac side follows the Isaac Sim 6.0 API used by legacy/methods/cable_view.py.
Other versions may need small changes (asset path, import names): the error says where.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


# =============================================================================
# The simulation: ethernet_scene.py's scene, one frame at a time
# =============================================================================
def simulate(config, bend_scale):
    """Returns (header, frames): frames yields (t, poses) with poses [n x 7: px py pz qx qy qz qw]."""
    sys.path.insert(0, str(HERE))
    import warp as wp

    import ethernet_scene as es

    wp.config.quiet = True
    cfg = es.load_config(config)
    scan = es.load_scan(cfg)
    S = es.setup(cfg, scan, bend_scale)
    sim, plan, cable = cfg["sim"], S.plan, S.cable_bodies
    body_q = S.state_0.body_q.numpy()
    links = {S.model.body_label[b].split("/")[-1]: np.round(body_q[b], 6).tolist() for b in S.franka_bodies}
    header = {"segments": len(cable), "segment_length_m": plan["h"], "radius_m": scan["radius"],
              "ground_z_m": scan["ground_z"], "robot_links": links,  # the Franka does not move: sent once
              "scan_nodes": np.round(scan["nodes"], 6).tolist(), "EI_Nm2": S.info["EI_Nm2"]}

    def frames():
        state_0, state_1 = S.state_0, S.state_1
        fps, substeps = sim["fps"], int(sim["substeps"])
        dt = 1.0 / fps / substeps
        half = 0.5 * plan["h"]
        window = max(1, round(sim["settle_window_s"] * fps))
        snapshot = es.nodes_from_bodies(state_0.body_q.numpy()[cable], half)
        t, frame = 0.0, 0
        print(f"[newton] {len(cable)} segments, EI {S.info['EI_Nm2']:.3g} N m^2, SolverVBD on {S.device}")
        yield t, state_0.body_q.numpy()[cable]
        while t < sim["max_time_s"]:
            for _ in range(substeps):
                state_0.clear_forces()
                S.pipeline.collide(state_0, S.contacts)
                S.solver.step(state_0, state_1, S.control, S.contacts, dt)
                state_0, state_1 = state_1, state_0
            t += 1.0 / fps
            frame += 1
            yield t, state_0.body_q.numpy()[cable]
            if frame % window:
                continue
            nodes = es.nodes_from_bodies(state_0.body_q.numpy()[cable], half)
            if not np.all(np.isfinite(nodes)):
                raise SystemExit("[newton] simulation blew up (NaN)")
            drift = float(np.linalg.norm(nodes - snapshot, axis=1).max())
            snapshot = nodes
            print(f"[newton] t = {t:4.1f} s   shape change over {sim['settle_window_s']} s: {1000 * drift:7.3f} mm")
            if drift < sim["settle_tol_m"] and t >= sim["min_time_s"]:
                print(f"[newton] settled at t = {t:.2f} s")
                return
        print(f"[newton] NOT settled after {t:.1f} s")

    return header, frames()


def newton_here():
    """(ok, note): can this Python run ethernet_scene.py itself?"""
    try:
        import newton
    except ImportError:
        try:  # Isaac Sim ships Newton as an extension: put it on the path as the legacy code did
            sys.path.insert(0, str(REPO / "legacy" / "methods"))
            import isaac_env

            isaac_env.bootstrap_newton()
            import newton
        except Exception as exc:
            return False, f"no Newton in this Python ({exc})"
    if not hasattr(newton, "Rod"):
        return False, f"Newton {newton.__version__} here has no newton.Rod (ethernet_scene.py needs 1.6)"
    return True, f"Newton {newton.__version__}"


# =============================================================================
# Worker mode: the simulation under another Python. One JSON line (the header),
# then per frame: time [float64] + poses [n x 7 float32].
# =============================================================================
def worker(args):
    out = os.fdopen(os.dup(1), "wb")  # the pipe to the viewer
    os.dup2(2, 1)                     # every print goes to the terminal instead
    sys.stdout = sys.stderr
    header, frames = simulate(args.config, args.bend_scale)
    try:
        out.write((json.dumps(header) + "\n").encode())
        for t, poses in frames:
            out.write(struct.pack("<d", t) + poses.astype("<f4").tobytes())
            out.flush()
    except BrokenPipeError:  # the viewer was closed
        pass


# =============================================================================
# Worker mode, viewer side: start the worker and read what it sends
# =============================================================================
def newton_python(choice):
    if choice:
        return choice
    if os.environ.get("NEWTON_PYTHON"):
        return os.environ["NEWTON_PYTHON"]
    for cand in (REPO / ".env" / "bin" / "python", REPO / ".venv" / "bin" / "python"):
        if cand.exists():
            return str(cand)
    return "python3"


def start_worker(args):
    """Launch the Newton worker; returns (process, header)."""
    cmd = [newton_python(args.newton_python), str(Path(__file__).resolve()), "--worker",
           "--config", args.config, "--bend-scale", str(args.bend_scale)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, cwd=str(REPO))
    line = proc.stdout.readline()
    if not line:
        raise SystemExit(f"the Newton worker failed to start (see the error above). It was run as:\n  {' '.join(cmd)}\n"
                         "Point --newton-python (or $NEWTON_PYTHON) at the Python where ethernet_scene.py runs.")
    return proc, json.loads(line)


def read_frames(proc, n_seg, frames):
    """Thread body: put (t, poses) on the queue; None when the worker is done."""
    size = 8 + n_seg * 7 * 4
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        frames.put((struct.unpack("<d", buf[:8])[0], np.frombuffer(buf, dtype="<f4", offset=8).reshape(n_seg, 7)))
    frames.put(None)


WAIT = object()  # no new frame yet


def viewer(args):
    # SimulationApp must exist before any other isaacsim / pxr import, and before Warp.
    from isaacsim.simulation_app import SimulationApp

    app = SimulationApp({"headless": False})
    proc = None
    try:
        here, note = newton_here() if args.newton != "worker" else (False, "--newton worker")
        if args.newton == "here" and not here:
            raise SystemExit(f"--newton here: {note}")
        if here:
            print(f"[isaac] Newton runs in this process ({note})")
            head, sim_frames = simulate(args.config, args.bend_scale)

            def next_frame():
                return next(sim_frames, None)
        else:
            print(f"[isaac] Newton runs in a worker process: {note}")
            proc, head = start_worker(args)
            inbox = queue.Queue(maxsize=4)  # small: Newton waits for the window, no frame is skipped
            threading.Thread(target=read_frames, args=(proc, int(head["segments"]), inbox), daemon=True).start()

            def next_frame():
                try:
                    return inbox.get_nowait()
                except queue.Empty:
                    return WAIT

        n, h, r = int(head["segments"]), float(head["segment_length_m"]), float(head["radius_m"])
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux, Vt

        stage = omni.usd.get_context().get_stage()
        UsdGeom.Xform.Define(stage, "/World")
        UsdLux.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(1000.0)
        key = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key.CreateIntensityAttr().Set(3000.0)
        UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(45.0, 0.0, 25.0))

        # --- table: a thin slab with its top at the table height ----------------
        scan_nodes = np.asarray(head["scan_nodes"], float)
        lo, hi = scan_nodes.min(axis=0), scan_nodes.max(axis=0)
        centre, span = 0.5 * (lo + hi), float(np.linalg.norm(hi - lo)) + 1e-3
        table = UsdGeom.Cube.Define(stage, "/World/table")
        table.CreateSizeAttr().Set(1.0)
        table.CreateDisplayColorAttr().Set([Gf.Vec3f(0.55, 0.55, 0.55)])
        xf = UsdGeom.Xformable(table.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(float(centre[0]), float(centre[1]), float(head["ground_z_m"]) - 0.005))
        xf.AddScaleOp().Set(Gf.Vec3f(3.0, 3.0, 0.01))

        # --- cable: visual capsules, no physics ---------------------------------
        UsdGeom.Xform.Define(stage, "/World/cable")
        ops = []
        for i in range(n):
            cap = UsdGeom.Capsule.Define(stage, f"/World/cable/seg_{i:03d}")
            cap.CreateRadiusAttr().Set(r)
            cap.CreateHeightAttr().Set(h)  # Newton's capsule: a cylinder of the segment length plus two caps
            cap.CreateAxisAttr().Set("Z")
            cap.CreateDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.25, 0.85)])
            xf = UsdGeom.Xformable(cap.GetPrim())
            xf.ClearXformOpOrder()
            ops.append((xf.AddTranslateOp(), xf.AddOrientOp()))

        def show(poses):
            for (move, turn), p in zip(ops, poses.tolist()):
                move.Set(Gf.Vec3d(p[0], p[1], p[2]))
                turn.Set(Gf.Quatf(p[6], p[3], p[4], p[5]))  # Newton (x, y, z, w) -> Gf (w, x, y, z)

        # --- scan: red line -----------------------------------------------------
        line = UsdGeom.BasisCurves.Define(stage, "/World/scan")
        line.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        line.CreateCurveVertexCountsAttr().Set([len(scan_nodes)])
        line.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in scan_nodes.tolist()]))
        line.CreateWidthsAttr().Set(Vt.FloatArray([0.003]))
        line.SetWidthsInterpolation(UsdGeom.Tokens.constant)
        line.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.2, 0.2)])

        # --- Franka: Isaac's meshes at the link poses of the Newton scene ---------
        if not args.no_robot and head.get("robot_links"):
            draw_franka(stage, head["robot_links"])

        try:
            from isaacsim.core.utils.viewports import set_camera_view

            set_camera_view(eye=centre + np.array([0.0, -2.0 * span, 0.8 * span]), target=centre)
        except Exception as exc:  # framing is cosmetic
            print(f"[isaac] camera framing failed: {exc}")

        print(f"[isaac] drawing {n} segments of {1000 * h:.1f} mm, r {1000 * r:.2f} mm")
        running = True
        while app.is_running():
            if running:
                item = next_frame()
                if item is None:
                    running = False
                    print("[isaac] Newton finished: showing the final shape. Close the window to exit.")
                elif item is not WAIT:
                    show(item[1])
            app.update()
    finally:
        if proc is not None:
            proc.kill()
        app.close()


def draw_franka(stage, links):
    """Reference Isaac's franka.usd and put every link at its Newton pose (x, y, z, qx, qy, qz, qw)."""
    import importlib

    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Usd, UsdGeom

    root = None
    for mod in ("isaacsim.storage.native", "isaacsim.core.utils.nucleus"):
        try:
            root = importlib.import_module(mod).get_assets_root_path()
            break
        except Exception:
            continue
    if root is None:
        raise SystemExit("could not find the Isaac assets root (Nucleus / local assets); use --no-robot")
    usd = None
    for rel in ("/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "/Isaac/Robots/Franka/franka.usd"):
        try:
            add_reference_to_stage(usd_path=root + rel, prim_path="/World/franka")
            if stage.GetPrimAtPath("/World/franka").GetChildren():
                usd = root + rel
                break
        except Exception:
            continue
    if usd is None:
        raise SystemExit(f"Franka asset not found under {root}; use --no-robot")

    # Newton's FR3 link "fr3_link3" is the asset's "panda_link3". The FR3 URDF turns the
    # right finger's frame by 180 deg about z where the Panda turns the mesh instead.
    placed = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/franka")):
        name = prim.GetName().replace("panda_", "fr3_")
        if name not in links or not prim.IsA(UsdGeom.Xformable):
            continue
        x, y, z, qx, qy, qz, qw = links[name]
        if name == "fr3_rightfinger":  # q * (half turn about the link's own z)
            qx, qy, qz, qw = qy, -qx, qw, -qz
        world = Gf.Matrix4d().SetRotate(Gf.Quatd(qw, qx, qy, qz))
        world.SetTranslateOnly(Gf.Vec3d(x, y, z))
        parent = UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(world * parent.GetInverse())
        placed += 1
    print(f"[isaac] Franka from {usd}: {placed} links placed at their Newton poses")
    if placed < 8:
        print("[isaac] WARNING: few link names matched (expected panda_link0..7, panda_hand, fingers); "
              "the robot may be drawn wrong. It does not affect the cable.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True, help="JSON of the scene, e.g. results/ethernet_cat6_fit/best_config.json")
    ap.add_argument("--bend-scale", type=float, default=1.0, help="multiplier on EI and GJ")
    ap.add_argument("--newton", choices=["auto", "here", "worker"], default="auto",
                    help="where Newton runs: in this process, or in a worker under --newton-python (auto: here if it can)")
    ap.add_argument("--newton-python", default=None, help="worker mode: the Python that runs ethernet_scene.py")
    ap.add_argument("--no-robot", action="store_true", help="leave the Franka out")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    worker(args) if args.worker else viewer(args)


if __name__ == "__main__":
    main()
