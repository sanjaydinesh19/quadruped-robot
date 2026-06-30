#!/usr/bin/env python3
"""
Parametric URDF generator for the quadruped robot.

Reads config/robot_params.yaml and writes
src/ros2/quadruped_description/urdf/quadruped.urdf.

Usage
-----
    python scripts/generate_urdf.py            # write to default output path
    python scripts/generate_urdf.py --preview  # dump XML to stdout
"""
from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = REPO_ROOT / "config" / "robot_params.yaml"
OUTPUT_FILE = (
    REPO_ROOT
    / "src" / "ros2" / "quadruped_description" / "urdf" / "quadruped.urdf"
)

# ── Inertia helpers ───────────────────────────────────────────────────────────
# All return (Ixx, Iyy, Izz) assuming product-of-inertia terms are zero
# (uniform density, symmetric geometry).


def _box_inertia(m: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Solid box with full side-lengths x, y, z."""
    return (
        m / 12.0 * (y**2 + z**2),
        m / 12.0 * (x**2 + z**2),
        m / 12.0 * (x**2 + y**2),
    )


def _cylinder_inertia(
    m: float, r: float, h: float, axis: str = "z"
) -> tuple[float, float, float]:
    """
    Solid cylinder.  axis indicates which body axis the cylinder is aligned with.
    URDF default is z; rotate the visual/collision origin to match.
    """
    i_axial = 0.5 * m * r**2
    i_trans = m / 12.0 * (3.0 * r**2 + h**2)
    if axis == "z":
        return i_trans, i_trans, i_axial
    if axis == "y":
        return i_trans, i_axial, i_trans
    if axis == "x":
        return i_axial, i_trans, i_trans
    raise ValueError(f"axis must be x/y/z, got '{axis}'")


def _sphere_inertia(m: float, r: float) -> tuple[float, float, float]:
    i = 2.0 / 5.0 * m * r**2
    return i, i, i


# ── XML element builders ──────────────────────────────────────────────────────

def _fmt3(a: float, b: float, c: float) -> str:
    return f"{a:.6f} {b:.6f} {c:.6f}"


def _origin(parent: ET.Element, xyz: tuple, rpy: tuple = (0.0, 0.0, 0.0)) -> None:
    ET.SubElement(parent, "origin", xyz=_fmt3(*xyz), rpy=_fmt3(*rpy))


def _inertial(
    parent: ET.Element,
    mass: float,
    ixx: float, iyy: float, izz: float,
    com: tuple = (0.0, 0.0, 0.0),
) -> None:
    el = ET.SubElement(parent, "inertial")
    _origin(el, com)
    ET.SubElement(el, "mass", value=f"{mass:.6f}")
    ET.SubElement(
        el, "inertia",
        ixx=f"{ixx:.8f}", ixy="0.0", ixz="0.0",
        iyy=f"{iyy:.8f}", iyz="0.0",
        izz=f"{izz:.8f}",
    )


def _material(parent: ET.Element, rgba: tuple) -> None:
    mat = ET.SubElement(parent, "material", name="")
    ET.SubElement(mat, "color", rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")


def _visual_box(
    parent: ET.Element,
    size: tuple,
    xyz: tuple = (0.0, 0.0, 0.0),
    rgba: tuple = (0.3, 0.3, 0.8, 1.0),
) -> None:
    vis = ET.SubElement(parent, "visual")
    _origin(vis, xyz)
    ET.SubElement(ET.SubElement(vis, "geometry"), "box",
                  size=_fmt3(*size))
    _material(vis, rgba)


def _visual_cylinder(
    parent: ET.Element,
    radius: float,
    length: float,
    xyz: tuple = (0.0, 0.0, 0.0),
    rpy: tuple = (0.0, 0.0, 0.0),
    rgba: tuple = (0.8, 0.5, 0.1, 1.0),
) -> None:
    vis = ET.SubElement(parent, "visual")
    _origin(vis, xyz, rpy)
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")
    _material(vis, rgba)


def _visual_sphere(
    parent: ET.Element,
    radius: float,
    xyz: tuple = (0.0, 0.0, 0.0),
    rgba: tuple = (0.9, 0.2, 0.1, 1.0),
) -> None:
    vis = ET.SubElement(parent, "visual")
    _origin(vis, xyz)
    ET.SubElement(ET.SubElement(vis, "geometry"), "sphere",
                  radius=f"{radius:.6f}")
    _material(vis, rgba)


def _collision_box(
    parent: ET.Element, size: tuple, xyz: tuple = (0.0, 0.0, 0.0)
) -> None:
    col = ET.SubElement(parent, "collision")
    _origin(col, xyz)
    ET.SubElement(ET.SubElement(col, "geometry"), "box",
                  size=_fmt3(*size))


def _collision_cylinder(
    parent: ET.Element,
    radius: float,
    length: float,
    xyz: tuple = (0.0, 0.0, 0.0),
    rpy: tuple = (0.0, 0.0, 0.0),
) -> None:
    col = ET.SubElement(parent, "collision")
    _origin(col, xyz, rpy)
    geom = ET.SubElement(col, "geometry")
    ET.SubElement(geom, "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")


def _collision_sphere(
    parent: ET.Element, radius: float, xyz: tuple = (0.0, 0.0, 0.0)
) -> None:
    col = ET.SubElement(parent, "collision")
    _origin(col, xyz)
    ET.SubElement(ET.SubElement(col, "geometry"), "sphere",
                  radius=f"{radius:.6f}")


# ── URDF builder ──────────────────────────────────────────────────────────────

class URDFBuilder:
    """Builds a URDF ElementTree from the robot parameter dict."""

    HALF_PI = math.pi / 2.0

    # (x_sign, y_sign) for each leg — places hips at body corners.
    # x_sign: +1 = front, -1 = rear
    # y_sign: +1 = left,  -1 = right
    LEGS: dict[str, tuple[int, int]] = {
        "FL": (+1, +1),
        "FR": (+1, -1),
        "RL": (-1, +1),
        "RR": (-1, -1),
    }

    def __init__(self, params: dict) -> None:
        self.p = params
        self.root = ET.Element("robot", name=params["robot"]["name"])

    def build(self) -> "URDFBuilder":
        self._add_body()
        for leg_name, (xs, ys) in self.LEGS.items():
            self._add_leg(leg_name, xs, ys)
        return self

    # ── Body ──────────────────────────────────────────────────────────────────

    def _add_body(self) -> None:
        b = self.p["body"]
        lx, ly, lz = b["length_m"], b["width_m"], b["height_m"]
        mass = b["mass_kg"]
        ixx, iyy, izz = _box_inertia(mass, lx, ly, lz)

        link = ET.SubElement(self.root, "link", name="base_link")
        _inertial(link, mass, ixx, iyy, izz)
        _visual_box(link, (lx, ly, lz), rgba=(0.18, 0.40, 0.78, 1.0))
        _collision_box(link, (lx, ly, lz))

    # ── Leg (hip → thigh → shin → foot) ──────────────────────────────────────

    def _add_leg(self, name: str, x_sign: int, y_sign: int) -> None:
        lg  = self.p["legs"]
        lim = self.p["joint_limits"]
        act = self.p["actuator"]

        # Unpack geometry
        l_hip   = lg["l_hip_m"]
        l_thigh = lg["l_thigh_m"]
        l_shin  = lg["l_shin_m"]

        r_hip   = lg["hip_radius_m"]
        r_thigh = lg["thigh_radius_m"]
        r_shin  = lg["shin_radius_m"]
        r_foot  = lg["foot_radius_m"]

        m_hip   = lg["hip_mass_kg"]
        m_thigh = lg["thigh_mass_kg"]
        m_shin  = lg["shin_mass_kg"]
        m_foot  = lg["foot_mass_kg"]

        hip_x = x_sign * self.p["hip_x_offset_m"]
        hip_y = y_sign * self.p["hip_y_offset_m"]

        # ── hip joint ────────────────────────────────────────────────────────
        # Connects base_link to {leg}_hip_link.
        # Rotates about the x-axis (abduction/adduction).
        # Position: at the body corner where the motor housing sits.
        self._revolute_joint(
            name=f"{name}_hip_joint",
            parent="base_link",
            child=f"{name}_hip_link",
            xyz=(hip_x, hip_y, 0.0),
            axis=(1, 0, 0),
            limits=lim["hip_abduction"],
            damping=act["damping"],
            friction=act["friction"],
        )

        # ── hip link ─────────────────────────────────────────────────────────
        # Short cylinder along y-axis (the abduction offset).
        # COM sits at the mid-point of the abduction link.
        hip_link = ET.SubElement(self.root, "link", name=f"{name}_hip_link")
        ixx, iyy, izz = _cylinder_inertia(m_hip, r_hip, l_hip, axis="y")
        com_hip = (0.0, y_sign * l_hip / 2.0, 0.0)
        _inertial(hip_link, m_hip, ixx, iyy, izz, com=com_hip)
        # Cylinder default is along z → rotate 90° about x to align with y.
        _visual_cylinder(
            hip_link, r_hip, l_hip,
            xyz=com_hip,
            rpy=(self.HALF_PI, 0.0, 0.0),
            rgba=(0.55, 0.55, 0.55, 1.0),
        )
        _collision_cylinder(
            hip_link, r_hip, l_hip,
            xyz=com_hip,
            rpy=(self.HALF_PI, 0.0, 0.0),
        )

        # ── thigh joint ───────────────────────────────────────────────────────
        # At the distal end of the hip link; rotates about y (flexion).
        self._revolute_joint(
            name=f"{name}_thigh_joint",
            parent=f"{name}_hip_link",
            child=f"{name}_thigh_link",
            xyz=(0.0, y_sign * l_hip, 0.0),
            axis=(0, 1, 0),
            limits=lim["thigh"],
            damping=act["damping"],
            friction=act["friction"],
        )

        # ── thigh link ────────────────────────────────────────────────────────
        # Cylinder hanging downward (−z in thigh frame).
        thigh_link = ET.SubElement(self.root, "link", name=f"{name}_thigh_link")
        ixx, iyy, izz = _cylinder_inertia(m_thigh, r_thigh, l_thigh, axis="z")
        com_thigh = (0.0, 0.0, -l_thigh / 2.0)
        _inertial(thigh_link, m_thigh, ixx, iyy, izz, com=com_thigh)
        _visual_cylinder(
            thigh_link, r_thigh, l_thigh,
            xyz=com_thigh,
            rgba=(0.28, 0.65, 0.28, 1.0),
        )
        _collision_cylinder(thigh_link, r_thigh, l_thigh, xyz=com_thigh)

        # ── knee joint ────────────────────────────────────────────────────────
        # At the bottom of the thigh link; rotates about y (knee flexion).
        self._revolute_joint(
            name=f"{name}_knee_joint",
            parent=f"{name}_thigh_link",
            child=f"{name}_shin_link",
            xyz=(0.0, 0.0, -l_thigh),
            axis=(0, 1, 0),
            limits=lim["knee"],
            damping=act["damping"],
            friction=act["friction"],
        )

        # ── shin link ─────────────────────────────────────────────────────────
        shin_link = ET.SubElement(self.root, "link", name=f"{name}_shin_link")
        ixx, iyy, izz = _cylinder_inertia(m_shin, r_shin, l_shin, axis="z")
        com_shin = (0.0, 0.0, -l_shin / 2.0)
        _inertial(shin_link, m_shin, ixx, iyy, izz, com=com_shin)
        _visual_cylinder(
            shin_link, r_shin, l_shin,
            xyz=com_shin,
            rgba=(0.20, 0.55, 0.85, 1.0),
        )
        _collision_cylinder(shin_link, r_shin, l_shin, xyz=com_shin)

        # ── foot joint (fixed) + link ──────────────────────────────────────
        # Contact point at the tip of the shin — no mass, just a collision sphere.
        foot_joint = ET.SubElement(
            self.root, "joint", name=f"{name}_foot_joint", type="fixed"
        )
        ET.SubElement(foot_joint, "parent", link=f"{name}_shin_link")
        ET.SubElement(foot_joint, "child",  link=f"{name}_foot_link")
        _origin(foot_joint, (0.0, 0.0, -l_shin))

        foot_link = ET.SubElement(self.root, "link", name=f"{name}_foot_link")
        ixx, iyy, izz = _sphere_inertia(m_foot, r_foot)
        _inertial(foot_link, m_foot, ixx, iyy, izz)
        _visual_sphere(foot_link, r_foot, rgba=(0.90, 0.20, 0.10, 1.0))
        _collision_sphere(foot_link, r_foot)

    # ── Joint helper ──────────────────────────────────────────────────────────

    def _revolute_joint(
        self,
        name: str,
        parent: str,
        child: str,
        xyz: tuple,
        axis: tuple,
        limits: dict,
        damping: float,
        friction: float,
    ) -> None:
        j = ET.SubElement(self.root, "joint", name=name, type="revolute")
        ET.SubElement(j, "parent", link=parent)
        ET.SubElement(j, "child",  link=child)
        _origin(j, xyz)
        ET.SubElement(j, "axis", xyz=_fmt3(*axis))
        ET.SubElement(
            j, "limit",
            lower=f"{limits['lower_rad']:.6f}",
            upper=f"{limits['upper_rad']:.6f}",
            velocity=f"{limits['velocity_rads']:.6f}",
            effort=f"{limits['effort_nm']:.6f}",
        )
        ET.SubElement(j, "dynamics",
                      damping=f"{damping:.4f}",
                      friction=f"{friction:.4f}")

    # ── Output ────────────────────────────────────────────────────────────────

    def to_xml_string(self) -> str:
        raw = ET.tostring(self.root, encoding="unicode")
        pretty = minidom.parseString(raw).toprettyxml(indent="  ")
        # minidom inserts a blank line after the XML declaration; remove empty lines.
        lines = [ln for ln in pretty.splitlines() if ln.strip()]
        return "\n".join(lines) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_xml_string())
        print(f"[generate_urdf] Written → {path}")
        _print_summary(self.p)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(p: dict) -> None:
    b = self_mass = p["body"]["mass_kg"]
    lg = p["legs"]
    per_leg = lg["hip_mass_kg"] + lg["thigh_mass_kg"] + lg["shin_mass_kg"] + lg["foot_mass_kg"]
    total_mass = b + 4 * per_leg
    reach_m = lg["l_thigh_m"] + lg["l_shin_m"]
    print(f"  Body mass   : {p['body']['mass_kg']:.3f} kg")
    print(f"  Per-leg mass: {per_leg:.3f} kg  ×4 = {4*per_leg:.3f} kg")
    print(f"  Total mass  : {total_mass:.3f} kg")
    print(f"  Max leg reach: {reach_m:.3f} m")
    print(f"  Links: 1 body + 4×(hip + thigh + shin + foot) = 17 links")
    print(f"  Joints: 12 revolute + 4 fixed = 16 joints")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preview", action="store_true",
        help="Print URDF to stdout instead of writing to file",
    )
    parser.add_argument(
        "--params", type=Path, default=PARAMS_FILE,
        help=f"Path to robot_params.yaml (default: {PARAMS_FILE})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_FILE,
        help=f"Output .urdf path (default: {OUTPUT_FILE})",
    )
    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    builder = URDFBuilder(params).build()

    if args.preview:
        print(builder.to_xml_string())
    else:
        builder.write(args.output)


if __name__ == "__main__":
    main()
