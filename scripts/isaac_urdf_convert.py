#!/usr/bin/env python3
"""
URDF → USD conversion, headless-safe.

Thin wrapper around Isaac Lab's own scripts/tools/convert_urdf.py. It exists
because on Isaac Sim 4.5, the `isaaclab.python.headless.kit` experience file
does not list `isaacsim.asset.importer.urdf` as a dependency (only the GUI
`isaaclab.python.kit` does — see IsaacLab/apps/isaacsim_4_5/*.kit), so calling
the stock tool with --headless fails with:
    ModuleNotFoundError: No module named 'isaacsim.asset'
Isaac Sim 5.1+ works around this itself (UrdfConverter.__init__ enables the
extension unconditionally), but that fix is version-gated and doesn't apply
on 4.5. Running the stock tool WITHOUT --headless to pick up the GUI kit file
instead segfaults in _wait_for_viewport on this machine's 4 GB VRAM GPU (same
reason train_rl.py/watch_rl.py always pass --headless).

The extension itself is already present locally (Isaac Sim ships it), it
just isn't auto-enabled by the headless experience file — so we enable it
explicitly via the Kit extension manager before constructing UrdfConverter,
and stay fully headless.

Usage (same argument shape as IsaacLab's convert_urdf.py):
    ~/IsaacLab/isaaclab.sh -p scripts/isaac_urdf_convert.py \\
        <input.urdf> <output_usd_base> --merge-joints --headless
"""
from __future__ import annotations

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert a URDF into USD format (headless-safe).")
parser.add_argument("input", type=str, help="Path to the input URDF file.")
parser.add_argument("output", type=str, help="Path to store the USD file (base name, no extension).")
parser.add_argument("--merge-joints", action="store_true", default=False,
                     help="Consolidate links connected by fixed joints.")
parser.add_argument("--fix-base", action="store_true", default=False,
                     help="Fix the base to where it is imported.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app  # noqa: E402
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from isaaclab.utils.assets import check_file_path  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

# ── The actual workaround: enable the importer extension ourselves ──────────
# isaaclab's own UrdfConverter pins to isaacsim.asset.importer.urdf==2.4.31 for
# Isaac Sim >= 5.1 (see urdf_converter.py), but that version isn't resolvable
# against this 4.5 install's extension registry (only 2.3.14 is available
# locally) — so we enable whatever version is actually available instead.
_URDF_EXT = "isaacsim.asset.importer.urdf"
_ext_manager = omni.kit.app.get_app().get_extension_manager()
if not _ext_manager.is_extension_enabled(_URDF_EXT):
    print(f"[isaac_urdf_convert] Enabling {_URDF_EXT}")
    _ext_manager.set_extension_enabled_immediate(_URDF_EXT, True)

# 2.3.14 predates ImportConfig.set_merge_fixed_ignore_inertia(), which
# isaaclab's UrdfConverter._get_urdf_import_config() calls unconditionally
# (added upstream alongside the 2.4.31 pin). Patch in a no-op so the merge
# behaviour falls back to plain set_merge_fixed_joints() — the only thing the
# newer call adds is *also* zeroing the inertia contribution of the merged-in
# fixed-joint link, which is a refinement, not a requirement.
import isaacsim.asset.importer.urdf._urdf as _urdf_ext  # noqa: E402

if not hasattr(_urdf_ext.ImportConfig, "set_merge_fixed_ignore_inertia"):
    _urdf_ext.ImportConfig.set_merge_fixed_ignore_inertia = lambda self, _v: None


def main() -> None:
    urdf_path = os.path.abspath(args_cli.input)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    dest_path = os.path.abspath(args_cli.output)

    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args_cli.fix_base,
        merge_fixed_joints=args_cli.merge_joints,
        force_usd_conversion=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
            target_type="position",
        ),
    )

    print("-" * 80)
    print(f"Input URDF file: {urdf_path}")
    print("URDF importer config:")
    print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)

    urdf_converter = UrdfConverter(urdf_converter_cfg)
    print(f"Generated USD file: {urdf_converter.usd_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
