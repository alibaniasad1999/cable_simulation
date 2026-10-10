"""Move the Franka's hand with the keyboard while it holds the (fitted) cable. Newton.

    # from the repository root
    python 03_franka_holds_cable/keyboard_teleop.py --config results/ethernet_cat6_fit/best_config.json
    python 03_franka_holds_cable/keyboard_teleop.py --config configs/ethernet_cat6.json --device cuda:0
    python 03_franka_holds_cable/keyboard_teleop.py --config configs/ethernet_cat6.json --fast    # quicker, softer cable

Keys (click into the window first):
    left / right arrow   move the hand left / right   (world y)
    up / down arrow      move the hand up / down      (world z)
    W / S                move the hand forward / back (world x)
    R                    back to the start pose
    mouse                turn / zoom the camera (the keyboard camera keys are switched off)

The cable starts on the scanned shape, held in the gripper, and follows the hand.
The plug end is set free here (fix_plug_end off) so the cable can come along;
--keep-plug-fixed keeps it held. The Franka is kinematic: it is put where IK says
and does not collide with the cable (the cable can pass through the arm).

On a CPU the full solver settings (20 substeps x 20 iterations) are slower than
real time; use a GPU (--device cuda:0) or --fast (fewer substeps/iterations: the
cable bends more easily than the fitted one, fine for playing, not for measuring).
"""

from __future__ import annotations

import os
import sys

if sys.platform == "darwin" and os.environ.pop("DYLD_LIBRARY_PATH", None):
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ethernet_scene", HERE / "ethernet_scene.py")
es = importlib.util.module_from_spec(spec)
spec.loader.exec_module(es)


def qrot(q, v):
    """Rotate vectors v by quaternions q (x, y, z, w); both broadcast, shapes (..., 4) / (..., 3)."""
    u, w = q[..., :3], q[..., 3:4]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def tf_mul(pa, qa, pb, qb):
    """(pa, qa) o (pb, qb) for rigid transforms."""
    return pa + qrot(qa, pb), es.quat_mul(np.broadcast_to(qa, np.shape(qb)), qb)


def tf_inv(p, q):
    qi = q * np.array([-1.0, -1.0, -1.0, 1.0])
    return -qrot(qi, p), qi


class Teleop:
    """The scene plus a hand target that the keyboard moves (no window needed: testable)."""

    def __init__(self, cfg, scan, speed):
        self.cfg, self.scan, self.speed = cfg, scan, speed
        self.S = S = es.setup(cfg, scan)
        m = S.model
        sim = cfg["sim"]
        self.substeps = int(sim["substeps"])
        self.frame_dt = 1.0 / sim["fps"]
        self.dt = self.frame_dt / self.substeps
        self.n_grip = S.plan["n_grip"]
        self.grip_bodies = np.asarray(S.cable_bodies[: self.n_grip])
        labels = [m.body_label[b] for b in S.franka_bodies]
        self.tcp = S.franka_bodies[next(i for i, l in enumerate(labels) if l.endswith("fr3_hand_tcp"))]
        self.jq = m.joint_q.numpy().copy()  # full joint vector; the Franka is coordinates 0..8
        self.finger = float(scan["radius"])

        # The grip segments keep their pose relative to the TCP.
        bq = S.state_0.body_q.numpy()
        self.tcp_p0, self.tcp_q0 = bq[self.tcp, :3].copy(), bq[self.tcp, 3:7].copy()
        ip, iq = tf_inv(self.tcp_p0, self.tcp_q0)
        self.off_p, self.off_q = tf_mul(ip, iq, bq[self.grip_bodies, :3], bq[self.grip_bodies, 3:7])
        self.tcp_p, self.tcp_q = self.tcp_p0.copy(), self.tcp_q0.copy()
        self.target = self.tcp_p0.copy()

        # Persistent IK on a Franka-only model at the same base pose.
        robot = S.info["robot"]
        self.base = np.asarray(robot["base_xyz_m"], float)
        b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        es.add_franka(b, self.base, np.radians(robot["base_yaw_deg"]))
        b.joint_q[:9] = robot["joint_q"]
        self.ik_model = im = b.finalize(device=m.device)
        ik_tcp = im.body_label.index(next(l for l in im.body_label if l.endswith("fr3_hand_tcp")))
        n = im.joint_coord_count
        self.ik_q = wp.array(np.asarray(robot["joint_q"], np.float32).reshape(1, n), dtype=float, device=m.device)
        self.ik_target = wp.array([wp.vec3(*self.target)], dtype=wp.vec3, device=m.device)
        objectives = [
            ik.IKObjectivePosition(link_index=ik_tcp, link_offset=wp.vec3(0.0, 0.0, 0.0),
                                   target_positions=self.ik_target),
            ik.IKObjectiveRotation(link_index=ik_tcp, link_offset_rotation=wp.quat_identity(),
                                   target_rotations=wp.array([wp.vec4(*self.tcp_q0)], dtype=wp.vec4, device=m.device)),
            ik.IKObjectiveJointLimit(joint_limit_lower=im.joint_limit_lower, joint_limit_upper=im.joint_limit_upper,
                                     weight=10.0),
        ]
        self.ik = ik.IKSolver(model=im, n_problems=1, objectives=objectives, lambda_initial=0.05,
                              jacobian_mode=ik.IKJacobianType.ANALYTIC)

    def move(self, v):
        """Move the hand target by velocity v [m/s] for one frame, kept inside a safe region."""
        t = self.target + self.speed * np.asarray(v, float) * self.frame_dt
        t[2] = max(t[2], self.scan["ground_z"] + 0.03)  # stay above the table
        reach = t - self.base
        if np.linalg.norm(reach) > 0.80:  # Franka reach ~0.85 m
            t = self.base + reach / np.linalg.norm(reach) * 0.80
        self.target = t

    def reset(self):
        self.target = self.tcp_p0.copy()

    def step(self):
        """IK to the target, place the arm, then the cable substeps with the grip following the hand."""
        S = self.S
        self.ik_target.assign(np.asarray([self.target], np.float32))
        self.ik.step(self.ik_q, self.ik_q, iterations=20)
        q = self.ik_q.numpy().reshape(-1)
        self.jq[:7] = q[:7]
        self.jq[7:9] = self.finger
        jq = wp.array(self.jq, dtype=float, device=S.device)
        for st in (S.state_0, S.state_1):  # arm links (rod bodies are never touched by eval_fk)
            newton.eval_fk(S.model, jq, S.model.joint_qd, st)
        bq = S.state_0.body_q.numpy()
        new_p, new_q = bq[self.tcp, :3].copy(), bq[self.tcp, 3:7].copy()
        if np.dot(new_q, self.tcp_q) < 0.0:
            new_q = -new_q
        old_p, old_q = self.tcp_p, self.tcp_q
        for k in range(1, self.substeps + 1):
            a = k / self.substeps  # move the grip smoothly within the frame, not in one jump
            p = (1 - a) * old_p + a * new_p
            qn = (1 - a) * old_q + a * new_q
            qn /= np.linalg.norm(qn)
            gp, gq = tf_mul(p, qn, self.off_p, self.off_q)
            for st in (S.state_0, S.state_1):
                b = st.body_q.numpy()
                b[self.grip_bodies, :3] = gp
                b[self.grip_bodies, 3:7] = gq
                st.body_q.assign(b)
            S.state_0.clear_forces()
            S.pipeline.collide(S.state_0, S.contacts)
            S.solver.step(S.state_0, S.state_1, S.control, S.contacts, self.dt)
            S.state_0, S.state_1 = S.state_1, S.state_0
        self.tcp_p, self.tcp_q = new_p, new_q
        return float(np.linalg.norm(new_p - self.target))  # IK error [m]


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True, help="e.g. results/ethernet_cat6_fit/best_config.json")
    ap.add_argument("--device", default=None, help="cpu or cuda:0 (default: JSON)")
    ap.add_argument("--speed", type=float, default=0.10, help="hand speed [m/s]")
    ap.add_argument("--fast", action="store_true", help="8 substeps x 10 iterations: quicker, cable bends more easily")
    ap.add_argument("--keep-plug-fixed", action="store_true", help="keep the plug end held at its scanned place")
    ap.add_argument("--hide-scan", action="store_true", help="do not draw the scanned centreline (red)")
    args = ap.parse_args()

    wp.config.quiet = True
    cfg = es.load_config(args.config)
    cfg["robot"]["enabled"] = True
    if not args.keep_plug_fixed:
        cfg["cable"]["fix_plug_end"] = False
    if args.device:
        cfg["sim"]["device"] = args.device
    if args.fast:
        cfg["sim"]["substeps"], cfg["sim"]["iterations"] = 8, 10
    scan = es.load_scan(cfg)
    T = Teleop(cfg, scan, args.speed)

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(T.S.model)
    viewer.camera_speed = 0.0  # arrow keys and W/S drive the hand, not the camera
    scan_X = wp.array(scan["nodes"].astype(np.float32), dtype=wp.vec3, device=T.S.device)
    print(__doc__.split("Keys")[1].split("The cable starts")[0].rstrip())

    t = 0.0
    while viewer.is_running():
        if not viewer.is_paused():
            v = np.zeros(3)
            v[1] += float(viewer.is_key_down("left")) - float(viewer.is_key_down("right"))
            v[2] += float(viewer.is_key_down("up")) - float(viewer.is_key_down("down"))
            v[0] += float(viewer.is_key_down("w")) - float(viewer.is_key_down("s"))
            if viewer.is_key_down("r"):
                T.reset()
            T.move(v)
            T.step()
            t += T.frame_dt
        viewer.begin_frame(t)
        viewer.log_state(T.S.state_0)
        if not args.hide_scan:
            viewer.log_lines("scan", scan_X[:-1], scan_X[1:], (1.0, 0.2, 0.2), width=0.003)
        viewer.end_frame()
    viewer.close()


if __name__ == "__main__":
    main()
