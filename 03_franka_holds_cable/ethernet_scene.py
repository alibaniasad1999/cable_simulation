"""Franka holds the scanned Ethernet cable: build the scene from a JSON config,
simulate it in Newton (VBD), and export the simulated centreline for comparison.

    # from the repository root
    python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json
    python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --viewer gl
    python 03_franka_holds_cable/ethernet_scene.py --config configs/ethernet_cat6.json --sweep

Then compare with the scan:

    python 04_pointcloud_vs_sim/compare_to_scan.py --config configs/ethernet_cat6.json

What is simulated (see README.md in this folder for the reasoning):

  * scan    the tube-fit centreline (04_pointcloud_vs_sim/tube_fit), Z-up, metres.
            Node 0 is put at the gripper end.
  * cable   a Newton rod built from section rigidities (EI, EA, GJ, kGA) in the
            JSON, so Newton divides by the segment length itself. Its REST shape
            is straight; only its starting pose follows the scan.
  * grip    the first `grip_length_m` of cable (inside the fingers) is clamped
            (zero mass = kinematic) at the scanned position and tangent of the
            gripper end. That is the boundary condition.
  * plug    `plug_length_m` of extra cable past the last scanned node, with the
            plug mass spread over it.
  * table   a ground plane at the height where the cable lies on it.
  * Franka  kinematic, placed with IK so its TCP sits on the grip. It does not
            touch the cable in the simulation and does not change the result.
  * settle  until the shape stops changing (sim.settle_tol_m over settle_window_s).
"""

from __future__ import annotations

import os
import sys

# macOS workaround (same as 01_franka_in_newton/franka_example.py): with
# DYLD_LIBRARY_PATH pointing at Homebrew, the viewer can die decoding a PNG.
if sys.platform == "darwin" and os.environ.pop("DYLD_LIBRARY_PATH", None):
    os.execv(sys.executable, [sys.executable, *sys.argv])

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton.solvers import SolverVBD

REPO = Path(__file__).resolve().parents[1]
UP = np.array([0.0, 0.0, 1.0])


# =============================================================================
# 1. Config and scan
# =============================================================================
def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    # Strip the "_comment" keys so they never reach the code below.
    def clean(d):
        return {k: clean(v) if isinstance(v, dict) else v for k, v in d.items() if not k.startswith("_")}
    return clean(cfg)


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def load_scan(cfg):
    """Tube-fit centreline, oriented so that node 0 is the gripper end."""
    scan_cfg = cfg["scan"]
    csv = repo_path(scan_cfg["centerline_csv"])
    if not csv.exists():
        raise SystemExit(
            f"scan centreline not found: {csv}\n"
            "Run the tube fit first, e.g.\n"
            "  python 04_pointcloud_vs_sim/tube_fit/fit_cable_tube.py Data/Ethernet.ply"
        )
    data = np.genfromtxt(csv, delimiter=",", names=True)
    X = np.column_stack([data["x_m"], data["y_m"], data["z_m"]])
    supported = data["supported"].astype(bool) if "supported" in data.dtype.names else np.ones(len(X), bool)

    end = scan_cfg.get("gripper_end", "auto")
    if end == "auto":
        end = "start" if X[0, 2] >= X[-1, 2] else "end"
    if end == "end":
        X, supported = X[::-1].copy(), supported[::-1].copy()
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])

    summary = {}
    summary_path = repo_path(scan_cfg["summary_json"]) if scan_cfg.get("summary_json") else None
    if summary_path is not None and summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    radius = cfg["cable"].get("radius_m")
    if radius is None:
        if "radius_mm" not in summary:
            raise SystemExit("cable.radius_m is null and summary.json has no radius_mm: set the radius in the JSON")
        radius = summary["radius_mm"] / 1000.0

    ground_z = scan_cfg.get("ground_z_m")
    if ground_z is None:
        ground_z = float(X[supported, 2].min()) - radius

    return {
        "nodes": X,
        "s": s,
        "supported": supported,
        "radius": float(radius),
        "ground_z": float(ground_z),
        "gripper_end": end,
        "ply_units": summary.get("ply_units", "m"),
        "path": str(csv),
    }


# =============================================================================
# 2. Geometry helpers (numpy only)
# =============================================================================
def unit(v):
    return v / np.linalg.norm(v)


def end_tangent(X, at_start, span=0.02):
    """Average direction over the first/last `span` metres (less noisy than one segment)."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    if at_start:
        k = max(1, int(np.searchsorted(s, span)))
        return unit(X[k] - X[0])
    k = min(len(X) - 2, int(np.searchsorted(s, s[-1] - span)))
    return unit(X[-1] - X[k])


def resample(X, n_seg):
    """n_seg + 1 points at equal arc length along polyline X."""
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    t = np.linspace(0.0, s[-1], n_seg + 1)
    return np.column_stack([np.interp(t, s, X[:, i]) for i in range(3)])


def quat_from_matrix(R):
    """3x3 rotation matrix -> quaternion (x, y, z, w), Newton/Warp order."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * S, (m[2, 1] - m[1, 2]) / S, (m[0, 2] - m[2, 0]) / S, (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / S, 0.25 * S, (m[0, 1] + m[1, 0]) / S, (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / S, (m[0, 1] + m[1, 0]) / S, 0.25 * S, (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / S, (m[0, 2] + m[2, 0]) / S, (m[1, 2] + m[2, 1]) / S, 0.25 * S
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quat_rotate_z(q):
    """Local +Z axis of each quaternion (x, y, z, w), shape (N, 3)."""
    qx, qy, qz, qw = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.column_stack([2 * (qx * qz + qy * qw), 2 * (qy * qz - qx * qw), 1 - 2 * (qx * qx + qy * qy)])


# =============================================================================
# 3. The cable path: grip + scan + plug
# =============================================================================
def plan_cable(scan, cable_cfg, sim_cfg, init_mode):
    """Segment layout and the starting centreline of the simulated cable.

    Arc length s is measured from the first scanned node (the edge of the
    fingers): the grip is s in [-grip, 0], the scan s in [0, L_scan], the plug
    after it. The segment length is adjusted so the grip is a whole number of
    segments; the plug absorbs the rounding of the total.
    """
    X = scan["nodes"]
    L_scan = scan["s"][-1]
    grip = float(cable_cfg["grip_length_m"])
    h0 = float(sim_cfg["segment_length_m"])
    n_grip = max(1, round(grip / h0))
    h = grip / n_grip
    n_seg = n_grip + max(2, round((L_scan + float(cable_cfg["plug_length_m"])) / h))
    plug = n_seg * h - grip - L_scan
    n_plug = max(1, round(plug / h)) if cable_cfg["plug_length_m"] > 0 else 0

    t0 = end_tangent(X, at_start=True)
    t1 = end_tangent(X, at_start=False)
    grip_start = X[0] - grip * t0

    if init_mode == "scan":
        # Grip straight into the fingers, the scanned curve, the plug straight on.
        dense = np.vstack([
            grip_start[None],
            X,
            (X[-1] + max(plug, 1e-4) * t1)[None],
        ])
        nodes = resample(dense, n_seg)
    elif init_mode == "drop":
        # Grip as scanned, then the rest straight and horizontal at grip height.
        horiz = t0 - np.dot(t0, UP) * UP
        if np.linalg.norm(horiz) < 1e-6:
            horiz = unit(X.mean(0) - X[0]) - np.dot(unit(X.mean(0) - X[0]), UP) * UP
        horiz = unit(horiz)
        s_rest = np.linspace(0.0, n_seg * h - grip, n_seg - n_grip + 1)
        rest = X[0] + s_rest[:, None] * horiz
        grip_nodes = grip_start + np.linspace(0.0, grip, n_grip + 1)[:, None] * t0
        nodes = np.vstack([grip_nodes[:-1], rest])
    else:
        raise SystemExit(f"sim.init must be 'scan' or 'drop', got {init_mode!r}")

    s_nodes = np.arange(n_seg + 1) * h - grip
    return {
        "nodes": nodes,
        "s": s_nodes,
        "h": h,
        "n_seg": n_seg,
        "n_grip": n_grip,
        "n_plug": n_plug,
        "plug_length": plug,
        "grip_tangent": t0,
        "grip_center": X[0] - 0.5 * grip * t0,
        "length_total": n_seg * h,
    }


def segment_poses(nodes):
    """Body transforms (p, q) for capsules with body_frame_origin='com' along `nodes`.

    Frames come from Newton's own parallel transport (newton.Rod), so the
    starting pose carries no twist.
    """
    q = np.asarray(newton.Rod(nodes).quaternions, dtype=float)
    p = 0.5 * (nodes[:-1] + nodes[1:])
    return p, q


def nodes_from_bodies(body_q, half_len):
    """Centreline nodes back from capsule transforms (body origin at the segment centre)."""
    p, q = body_q[:, :3], body_q[:, 3:7]
    z = quat_rotate_z(q)
    starts = p - half_len * z
    ends = p + half_len * z
    nodes = np.empty((len(p) + 1, 3))
    nodes[0] = starts[0]
    nodes[1:-1] = 0.5 * (ends[:-1] + starts[1:])
    nodes[-1] = ends[-1]
    return nodes


# =============================================================================
# 4. Franka: kinematic, placed with IK
# =============================================================================
FRANKA_URDF = "urdf/fr3_franka_hand.urdf"
FRANKA_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]


def franka_base(cfg_robot, scan, plan):
    """Base transform: given in the JSON, or on the table behind the gripper, facing it."""
    grip = plan["grip_center"]
    if cfg_robot.get("base_xyz_m") is not None:
        base = np.asarray(cfg_robot["base_xyz_m"], float)
        if cfg_robot.get("base_yaw_deg") is not None:
            yaw = np.radians(cfg_robot["base_yaw_deg"])
        else:
            d = grip - base
            yaw = np.arctan2(d[1], d[0])
    else:
        away = scan["nodes"].mean(0) - grip  # where the cable is
        away[2] = 0.0
        if np.linalg.norm(away) < 1e-6:
            away = -plan["grip_tangent"].copy()
            away[2] = 0.0
        away = unit(away) if np.linalg.norm(away) > 1e-6 else np.array([1.0, 0.0, 0.0])
        base = grip - float(cfg_robot.get("base_distance_m", 0.5)) * away
        base[2] = scan["ground_z"]
        yaw = np.arctan2(away[1], away[0])
    return base, float(yaw)


def tcp_target(plan):
    """TCP pose: between the fingertips on the grip, hand z along the cable."""
    z = plan["grip_tangent"]
    y = np.cross(UP, z)
    y = unit(y) if np.linalg.norm(y) > 1e-6 else np.array([0.0, 1.0, 0.0])  # finger-closing axis, horizontal
    x = np.cross(y, z)
    return plan["grip_center"], quat_from_matrix(np.column_stack([x, y, z]))


def add_franka(builder, base, yaw):
    xform = wp.transform(wp.vec3(*base), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
    start = builder.body_count
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / FRANKA_URDF,
        xform=xform,
        floating=False,
        enable_self_collisions=False,
        parse_visuals_as_colliders=False,
    )
    return list(range(start, builder.body_count))


def solve_franka_ik(base, yaw, target_p, target_q, finger, device):
    """Joint angles putting fr3_hand_tcp at the target, on a Franka-only model."""
    b = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    add_franka(b, base, yaw)
    b.joint_q[:9] = FRANKA_HOME
    m = b.finalize(device=device)
    tcp = m.body_label.index(next(l for l in m.body_label if l.endswith("fr3_hand_tcp")))
    n = m.joint_coord_count

    q = wp.array(np.asarray(m.joint_q.numpy(), np.float32).reshape(1, n), dtype=float, device=device)
    pos_obj = ik.IKObjectivePosition(
        link_index=tcp, link_offset=wp.vec3(0.0, 0.0, 0.0),
        target_positions=wp.array([wp.vec3(*target_p)], dtype=wp.vec3, device=device))
    rot_obj = ik.IKObjectiveRotation(
        link_index=tcp, link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(*target_q)], dtype=wp.vec4, device=device))
    lim_obj = ik.IKObjectiveJointLimit(
        joint_limit_lower=m.joint_limit_lower, joint_limit_upper=m.joint_limit_upper, weight=10.0)
    solver = ik.IKSolver(model=m, n_problems=1, objectives=[pos_obj, rot_obj, lim_obj],
                         lambda_initial=0.05, jacobian_mode=ik.IKJacobianType.ANALYTIC)
    solver.step(q, q, iterations=300)

    q_np = q.numpy().reshape(-1).copy()
    q_np[7:9] = finger
    st = m.state()
    newton.eval_fk(m, wp.array(q_np, dtype=float, device=device), m.joint_qd, st)
    got = st.body_q.numpy()[tcp]
    pos_err = float(np.linalg.norm(got[:3] - target_p))
    ang_err = float(np.degrees(2 * np.arccos(min(1.0, abs(np.dot(got[3:7], target_q))))))
    return q_np, pos_err, ang_err


# =============================================================================
# 5. Build the Newton model
# =============================================================================
def build_model(cfg, scan, plan, bend_scale, device):
    cab, con, rob = cfg["cable"], cfg["contact"], cfg["robot"]
    r = scan["radius"]
    h = plan["h"]

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverVBD.register_custom_attributes(builder)

    # --- Franka (kinematic) ---------------------------------------------------
    franka_bodies, robot_info = [], {}
    if rob.get("enabled", True):
        base, yaw = franka_base(rob, scan, plan)
        tp, tq = tcp_target(plan)
        q_ik, pos_err, ang_err = solve_franka_ik(base, yaw, tp, tq, finger=r, device=device)
        franka_bodies = add_franka(builder, base, yaw)
        builder.joint_q[:9] = q_ik.tolist()
        builder.joint_target_q[:9] = q_ik.tolist()
        for b in franka_bodies:  # zero mass = kinematic: VBD never moves it
            make_kinematic(builder, b)
        robot_info = {"base_xyz_m": base.round(4).tolist(), "base_yaw_deg": round(np.degrees(yaw), 2),
                      "joint_q": np.round(q_ik, 5).tolist(), "ik_pos_err_mm": round(1000 * pos_err, 2),
                      "ik_ang_err_deg": round(ang_err, 2)}
        if pos_err > 0.01 or ang_err > 5.0:
            print(f"WARNING: Franka cannot reach the grip exactly (IK error {1000 * pos_err:.1f} mm, "
                  f"{ang_err:.1f} deg). The robot is only visual; the cable clamp is unaffected. "
                  "Set robot.base_xyz_m in the JSON to move the robot.")

    # --- Cable ----------------------------------------------------------------
    EI = cab["bend_rigidity_EI_Nm2"] * bend_scale
    GJ = cab["twist_rigidity_GJ_Nm2"] * bend_scale
    # Straight rest shape. Newton turns the section rigidities into per-joint
    # stiffness (rigidity / segment length) by itself.
    rod = newton.Rod.create_straight(
        start=wp.vec3(*plan["nodes"][0]), direction=wp.vec3(*plan["grip_tangent"]),
        length=plan["length_total"], segment_count=plan["n_seg"], radius=r,
        stretch_rigidity=cab["stretch_rigidity_EA_N"], shear_rigidity=cab["shear_rigidity_kGA_N"],
        bend_rigidity=EI, twist_rigidity=GJ,
    )
    # Newton's mass = capsule volume x density, and each capsule has two
    # hemispherical caps on top of its length h. Correct the density so the mass
    # per metre is the real one.
    mu_lin = cab["mass_per_length_kg_m"]
    capsule_volume = np.pi * r * r * h + 4.0 / 3.0 * np.pi * r ** 3
    cable_cfg = newton.ModelBuilder.ShapeConfig(
        density=mu_lin * h / capsule_volume, ke=con["ke"], kd=con["kd"], mu=con["mu"],
        margin=0.0, gap=con["gap_m"])
    tau = cab["bend_damping_time_s"]
    cable_bodies, cable_joints = builder.add_rod(
        rod=rod, cfg=cable_cfg, body_frame_origin="com", label="ethernet",
        bend_damping=tau * EI / h, twist_damping=tau * GJ / h, stretch_damping=0.0,
    )
    for b in cable_bodies[: plan["n_grip"]]:  # clamped in the fingers
        make_kinematic(builder, b)
    if plan["n_plug"] > 0 and cab["plug_mass_kg"] > 0:
        dm = cab["plug_mass_kg"] / plan["n_plug"]
        for b in cable_bodies[-plan["n_plug"]:]:
            add_mass(builder, b, dm)

    # --- Table ----------------------------------------------------------------
    ground_cfg = newton.ModelBuilder.ShapeConfig(ke=con["ke"], kd=con["kd"], mu=con["mu"], margin=0.0, gap=con["gap_m"])
    ground_shape = builder.add_ground_plane(height=scan["ground_z"], cfg=ground_cfg, label="table")

    builder.color()  # VBD needs the colouring
    model = builder.finalize(device=device)

    masses = model.body_mass.numpy()
    info = {
        "segments": plan["n_seg"], "segment_length_m": h, "grip_segments": plan["n_grip"],
        "plug_segments": plan["n_plug"], "plug_length_m": round(plan["plug_length"], 4),
        "length_total_m": round(plan["length_total"], 4), "radius_m": r,
        "EI_Nm2": EI, "GJ_Nm2": GJ, "bend_scale": bend_scale,
        "per_joint_bend_stiffness_Nm_per_rad": EI / h,
        "mass_free_cable_kg": round(float(masses[cable_bodies[plan["n_grip"]:]].sum()), 5),
        "ground_z_m": scan["ground_z"], "robot": robot_info,
    }
    return model, cable_bodies, franka_bodies, ground_shape, info


def make_kinematic(builder, b):
    builder.body_mass[b] = 0.0
    builder.body_inv_mass[b] = 0.0
    builder.body_inertia[b] = wp.mat33(0.0)
    builder.body_inv_inertia[b] = wp.mat33(0.0)


def add_mass(builder, b, dm):
    """Add mass dm to body b, scaling its inertia by the same factor."""
    m = builder.body_mass[b]
    f = (m + dm) / m
    I = np.array(builder.body_inertia[b], dtype=float).reshape(3, 3) * f
    builder.body_mass[b] = m + dm
    builder.body_inv_mass[b] = 1.0 / (m + dm)
    builder.body_inertia[b] = wp.mat33(*I.reshape(-1))
    builder.body_inv_inertia[b] = wp.mat33(*np.linalg.inv(I).reshape(-1))


def contact_pairs(model, cable_bodies, franka_bodies, self_collision, device):
    """Cable-table pairs, plus cable-cable pairs more than 3 segments apart.

    The Franka is left out: it is kinematic, and its fingers overlap the
    clamped cable on purpose.
    """
    shape_body = model.shape_body.numpy()
    robot = set(franka_bodies)
    seg = {b: i for i, b in enumerate(cable_bodies)}
    keep = []
    for a, c in model.shape_contact_pairs.numpy():
        ba, bc = int(shape_body[a]), int(shape_body[c])
        if ba in robot or bc in robot:
            continue
        if ba in seg and bc in seg:
            if not self_collision or abs(seg[ba] - seg[bc]) <= 3:
                continue
        keep.append((int(a), int(c)))
    return wp.array(np.asarray(keep, np.int32), dtype=wp.vec2i, device=device), len(keep)


# =============================================================================
# 6. Simulate one run
# =============================================================================
def run(cfg, scan, bend_scale, out_dir, viewer_kind="null"):
    sim = cfg["sim"]
    device = sim.get("device")
    plan = plan_cable(scan, cfg["cable"], sim, sim["init"])
    model, cable_bodies, franka_bodies, ground_shape, info = build_model(cfg, scan, plan, bend_scale, device)
    device = model.device

    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)  # places the Franka; rod bodies untouched
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)

    # Start pose of the cable (the model itself keeps the straight rest shape).
    p, q = segment_poses(plan["nodes"])
    for st in (state_0, state_1):
        bq = st.body_q.numpy()
        bq[cable_bodies, :3] = p
        bq[cable_bodies, 3:7] = q
        st.body_q.assign(bq)
        bqd = st.body_qd.numpy()
        bqd[cable_bodies] = 0.0
        st.body_qd.assign(bqd)

    pairs, n_pairs = contact_pairs(model, cable_bodies, franka_bodies, cfg["contact"]["self_collision"], device)
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit", shape_pairs_filtered=pairs)
    contacts = pipeline.contacts()
    solver = SolverVBD(model, iterations=int(sim["iterations"]), friction_epsilon=float(sim["friction_epsilon"]),
                       rigid_compliant_alm=True, rigid_contact_history=False)

    viewer = None
    if viewer_kind == "gl":
        viewer = newton.viewer.ViewerGL()
        viewer.set_model(model)
        scan_X = wp.array(scan["nodes"].astype(np.float32), dtype=wp.vec3, device=device)

    fps, substeps = sim["fps"], int(sim["substeps"])
    frame_dt = 1.0 / fps
    dt = frame_dt / substeps
    half = 0.5 * plan["h"]
    window = max(1, round(sim["settle_window_s"] * fps))  # frames between shape checks
    snapshot = nodes_from_bodies(state_0.body_q.numpy()[cable_bodies], half)
    t, frame, drift, settled = 0.0, 0, float("nan"), False
    wall0 = time.time()
    print(f"[{info['segments']} segments of {1000 * plan['h']:.1f} mm, EI {info['EI_Nm2']:.3g} N m^2, "
          f"{n_pairs} contact pairs, device {device}]")

    def render():
        if viewer is not None:
            viewer.begin_frame(t)
            viewer.log_state(state_0)
            viewer.log_lines("scan", scan_X[:-1], scan_X[1:], (1.0, 0.2, 0.2), width=0.003)
            viewer.end_frame()

    while t < sim["max_time_s"]:
        if viewer is not None and not viewer.is_running():
            break
        for _ in range(substeps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
        t += frame_dt
        frame += 1
        render()
        if frame % window:
            continue
        # Settled = the shape stopped changing (largest node displacement over one window).
        nodes = nodes_from_bodies(state_0.body_q.numpy()[cable_bodies], half)
        if not np.all(np.isfinite(nodes)):
            raise SystemExit("simulation blew up (NaN): try more substeps/iterations or a softer contact ke")
        drift = float(np.linalg.norm(nodes - snapshot, axis=1).max())
        snapshot = nodes
        print(f"  t = {t:4.1f} s   shape change over {sim['settle_window_s']} s: {1000 * drift:7.3f} mm")
        if drift < sim["settle_tol_m"] and t >= sim["min_time_s"]:
            settled = True
            break

    nodes = nodes_from_bodies(state_0.body_q.numpy()[cable_bodies], half)
    info.update({"init": sim["init"], "settled": settled, "sim_time_s": round(t, 3),
                 "wall_time_s": round(time.time() - wall0, 1), "final_drift_m_per_window": drift,
                 "arc_length_m": float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum()),
                 "scan": scan["path"], "gripper_end": scan["gripper_end"]})
    info["stretch_percent"] = 100.0 * (info["arc_length_m"] / plan["length_total"] - 1.0)
    write_run(out_dir, plan, nodes, info, scan)
    print(f"  {'settled' if settled else 'NOT settled'} at t = {t:.2f} s ({info['wall_time_s']} s wall), "
          f"stretch {info['stretch_percent']:+.3f} %  ->  {out_dir}")

    if viewer is not None:  # keep showing the final shape until the window is closed
        while viewer.is_running():
            render()
        viewer.close()
    return info


# =============================================================================
# 7. Output
# =============================================================================
def write_run(out_dir, plan, nodes, info, scan):
    out_dir.mkdir(parents=True, exist_ok=True)
    part = np.where(plan["s"] < -1e-9, 0, np.where(plan["s"] <= scan["s"][-1] + 1e-9, 1, 2))
    header = "s_m,x_m,y_m,z_m,part"
    note = "part: 0 = in the gripper, 1 = scanned stretch, 2 = plug. s = 0 at the first scanned node."
    np.savetxt(out_dir / "sim_centerline.csv", np.column_stack([plan["s"], nodes, part]), delimiter=",",
               header=header, comments="", fmt=["%.6f"] * 4 + ["%d"])
    np.savetxt(out_dir / "init_centerline.csv", np.column_stack([plan["s"], plan["nodes"], part]), delimiter=",",
               header=header, comments="", fmt=["%.6f"] * 4 + ["%d"])
    info["columns_note"] = note
    with open(out_dir / "meta.json", "w") as f:
        json.dump(info, f, indent=2, default=float)
    # Same units as the scan's PLY, so it overlays the scan in CloudCompare.
    scale = 1000.0 if scan["ply_units"] == "mm" else 1.0
    write_ply(out_dir / "sim_centerline.ply", resample(nodes, 4 * len(nodes)) * scale, (0, 120, 255))
    write_as_scan(out_dir / "as_scan", plan, nodes, scan)


def write_as_scan(folder, plan, nodes, scan):
    """The simulated cable in the tube-fit output format (centerline.csv + summary.json).

    Point scan.centerline_csv at it to get a synthetic scan whose stiffness you
    know: the comparison must then find that stiffness (see README, self-test).
    """
    folder.mkdir(exist_ok=True)
    on_scan = (plan["s"] >= -1e-9) & (plan["s"] <= scan["s"][-1] + 1e-9)
    X = nodes[on_scan]  # leave out the part in the fingers and the plug, as the tube fit does
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(X, axis=0), axis=1))])
    T = np.gradient(X, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    np.savetxt(folder / "centerline.csv", np.column_stack([s, X, T, np.ones(len(X))]), delimiter=",",
               header="s_m,x_m,y_m,z_m,tx,ty,tz,supported", comments="", fmt=["%.6f"] * 7 + ["%d"])
    with open(folder / "summary.json", "w") as f:
        json.dump({"radius_mm": 1000.0 * scan["radius"], "ply_units": "m", "length_mm": 1000.0 * s[-1],
                   "frame": "synthetic scan written by ethernet_scene.py"}, f, indent=2)


def write_ply(path, points, rgb):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {rgb[0]} {rgb[1]} {rgb[2]}\n")


def run_label(bend_scale):
    return f"bend_x{bend_scale:g}"


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--config", required=True, help="JSON file, e.g. configs/ethernet_cat6.json")
    ap.add_argument("--bend-scale", type=float, default=1.0, help="multiplier on EI and GJ for a single run")
    ap.add_argument("--sweep", action="store_true", help="run every sweep.bend_scale value from the JSON")
    ap.add_argument("--viewer", choices=["null", "gl"], default="null", help="'gl' opens a window (single run only)")
    args = ap.parse_args()

    wp.config.quiet = True
    cfg = load_config(args.config)
    scan = load_scan(cfg)
    print(f"scan: {len(scan['nodes'])} nodes, {1000 * scan['s'][-1]:.1f} mm, radius {1000 * scan['radius']:.2f} mm, "
          f"gripper at the {scan['gripper_end']} of the CSV, table z = {scan['ground_z']:.4f} m")

    out_root = repo_path(cfg["output_dir"]) / cfg["sim"]["init"]
    scales = cfg["sweep"]["bend_scale"] if args.sweep else [args.bend_scale]
    for k in scales:
        run(cfg, scan, float(k), out_root / run_label(float(k)), args.viewer if not args.sweep else "null")
    with open(out_root / "config_used.json", "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
