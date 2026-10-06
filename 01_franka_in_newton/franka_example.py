import os
import sys

# macOS workaround: with DYLD_LIBRARY_PATH pointing at Homebrew, the system
# image library picks up Homebrew's libpng and the viewer dies with a bus error
# while decoding a PNG. Restart once without it. Does nothing on Linux.
if sys.platform == "darwin" and os.environ.pop("DYLD_LIBRARY_PATH", None):
    os.execv(sys.executable, [sys.executable, *sys.argv])

import warp as wp
import newton

def build_franka_scene(include_table=True, include_cube=True, use_targets=True):
    """Build a Franka + table scene and optionally add a cube."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = 0.0
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

    # Load the Franka from Newton's downloadable asset (same call Newton's own example uses)
    builder.add_urdf(
        newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,              # fixed base
        enable_self_collisions=False,
    )

    # Home pose: 7 arm joints + 2 finger joints
    builder.joint_q[:9] = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
    builder.joint_target_q[:9] = builder.joint_q[:9]

    table_height = 0.1
    table_pos = wp.vec3(0.0, -0.5, 0.5 * table_height)

    return builder, None, table_pos, None


# --- Your code on line 17 ---
builder, _, _, _ = build_franka_scene(include_table=False, include_cube=False, use_targets=True)



# 4. Finalize the model and initialize forward kinematics
model = builder.finalize()
state_0 = model.state()
state_1 = model.state()
control = model.control()
collision_pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
contacts = collision_pipeline.contacts()
newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

# =================================================================
# 5. SIMULATION & VISUALIZATION LOOP
# =================================================================

solver = newton.solvers.SolverMuJoCo(model)

sim_time = 0.0
dt = 1.0 / 100.0

viewer = newton.viewer.ViewerGL()
viewer.set_model(model)

while viewer.is_running():
    viewer.begin_frame(sim_time)

    collision_pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, dt)

    state_0, state_1 = state_1, state_0
    sim_time += dt

    viewer.log_state(state_0)
    viewer.end_frame()

viewer.close()
