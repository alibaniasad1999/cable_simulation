"""
Locate Isaac Sim and put its bundled Warp / Newton on ``sys.path``.

Isaac Sim ships Warp and Newton as Kit extensions rather than as packages
installed into its interpreter, so a bare ``./python.sh -c "import warp"``
fails. Kit adds them to the path when the corresponding extension is enabled,
which means a script that only wants the Newton PHYSICS (no renderer, no USD
stage, no Kit app) would otherwise have to boot the entire simulator just to
get an import to work -- tens of seconds, a GPU context, and a window.

``bootstrap_newton()`` adds those two extension directories directly, so the
pure-Newton method starts in about a second. The methods that genuinely need
the simulator (the Newton ENGINE via UsdPhysics, and both PhysX methods) still
boot Kit through ``isaacsim.simulation_app.SimulationApp`` as usual.

Override the install location with ``ISAAC_SIM_PATH=/path/to/isaacsim``.
"""

from __future__ import annotations

import glob
import os
import sys

_CANDIDATES = (
    os.environ.get("ISAAC_SIM_PATH", ""),
    os.path.expanduser("~/isaacsim"),
    "/isaac-sim",
    os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-*"),
)


def find_isaac_sim() -> str:
    """Return the Isaac Sim install root, or raise with a helpful message."""
    for cand in _CANDIDATES:
        if not cand:
            continue
        for path in sorted(glob.glob(cand), reverse=True):
            if os.path.isfile(os.path.join(path, "python.sh")):
                return path
    raise RuntimeError(
        "Isaac Sim not found. Set ISAAC_SIM_PATH=/path/to/isaacsim "
        f"(looked in: {', '.join(c for c in _CANDIDATES if c)})")


def isaac_version(root: str | None = None) -> str:
    root = root or find_isaac_sim()
    try:
        with open(os.path.join(root, "VERSION")) as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


def bootstrap_newton() -> tuple[str, str]:
    """Make ``import warp`` and ``import newton`` work without booting Kit.

    Returns ``(warp_version, newton_version)``.
    """
    root = find_isaac_sim()

    warp_dirs = sorted(glob.glob(os.path.join(root, "extscache", "omni.warp.core-*")))
    if not warp_dirs:
        raise RuntimeError(f"omni.warp.core extension not found under {root}/extscache")
    newton_dir = os.path.join(root, "exts", "isaacsim.pip.newton", "pip_prebundle")
    if not os.path.isdir(newton_dir):
        raise RuntimeError(
            f"{newton_dir} not found -- this Isaac Sim build does not bundle Newton. "
            "Newton cable support needs Isaac Sim 6.0 or newer.")

    # Prepend so Isaac's bundled versions win over anything pip-installed:
    # mixing a newer standalone Newton with Isaac's Warp is a known breakage.
    for path in (newton_dir, warp_dirs[-1]):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    import warp as wp
    import newton

    return wp.config.version, newton.__version__


if __name__ == "__main__":
    root = find_isaac_sim()
    warp_v, newton_v = bootstrap_newton()
    print(f"Isaac Sim : {root}  ({isaac_version(root)})")
    print(f"Warp      : {warp_v}")
    print(f"Newton    : {newton_v}")
