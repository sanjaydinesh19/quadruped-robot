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


def _material_ref(parent: ET.Element, name: str) -> None:
    """Reference a named <material> declared once at the robot root (see
    _add_material_library). Avoids repeating the same rgba on every one of
    the many cosmetic visuals attached to a link."""
    ET.SubElement(parent, "material", name=name)


def _add_material_library(
    root: ET.Element, materials: dict[str, tuple[float, float, float, float]]
) -> None:
    for name, rgba in materials.items():
        mat = ET.SubElement(root, "material", name=name)
        ET.SubElement(mat, "color", rgba=f"{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}")


def _visual_box(
    parent: ET.Element,
    size: tuple,
    xyz: tuple = (0.0, 0.0, 0.0),
    rpy: tuple = (0.0, 0.0, 0.0),
    material: str = "structural_grey",
) -> None:
    vis = ET.SubElement(parent, "visual")
    _origin(vis, xyz, rpy)
    ET.SubElement(ET.SubElement(vis, "geometry"), "box",
                  size=_fmt3(*size))
    _material_ref(vis, material)


def _visual_cylinder(
    parent: ET.Element,
    radius: float,
    length: float,
    xyz: tuple = (0.0, 0.0, 0.0),
    rpy: tuple = (0.0, 0.0, 0.0),
    material: str = "structural_grey",
) -> None:
    vis = ET.SubElement(parent, "visual")
    _origin(vis, xyz, rpy)
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "cylinder", radius=f"{radius:.6f}", length=f"{length:.6f}")
    _material_ref(vis, material)


def _visual_sphere(
    parent: ET.Element,
    radius: float,
    xyz: tuple = (0.0, 0.0, 0.0),
    material: str = "rubber_black",
) -> None:
    vis = ET.SubElement(parent, "visual")
    _origin(vis, xyz)
    ET.SubElement(ET.SubElement(vis, "geometry"), "sphere",
                  radius=f"{radius:.6f}")
    _material_ref(vis, material)


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


# ── Cosmetic-only geometry ─────────────────────────────────────────────────────
# Everything in this section emits <visual>-only elements: no <collision>, no
# <inertial>, no new <link> or <joint>. It cannot change contact detection,
# total mass, centre of mass, inertia, or the articulation graph — Isaac Lab's
# physics only ever looks at <collision>/<inertial>/<joint>. Dimensions are
# expressed as ratios of the real structural geometry so the cosmetic shell
# keeps tracking the robot if robot_params.yaml changes.

_MATERIALS: dict[str, tuple[float, float, float, float]] = {
    "chassis_dark":    (0.10, 0.10, 0.12, 1.0),   # matte chassis shell
    "chassis_accent":  (0.90, 0.40, 0.04, 1.0),   # orange trim (Unitree/ANYmal-style)
    "panel_dark":      (0.06, 0.06, 0.07, 1.0),   # side panels, vents, ribs, channels
    "structural_grey": (0.42, 0.43, 0.45, 1.0),   # leg tubes (was grey/green/blue — unified)
    "actuator_silver": (0.72, 0.73, 0.75, 1.0),   # QDD pancake motor housings
    "sensor_black":    (0.04, 0.04, 0.05, 1.0),   # camera/lidar housings, motor end caps
    "lens_glass":      (0.02, 0.05, 0.09, 1.0),   # camera lenses
    "led_green":       (0.15, 0.95, 0.25, 1.0),   # status LED
    "led_red":         (0.90, 0.08, 0.05, 1.0),   # rear tail light
    "rubber_black":    (0.05, 0.05, 0.06, 1.0),   # foot pads
}

_MOTOR_DISC_LEN = 0.018
_MOTOR_END_CAP_LEN = 0.004
_RIB_THICKNESS = 0.006
_RIB_RADIUS_SCALE = 1.15
_CHANNEL_THICKNESS = 0.006
_CHANNEL_WIDTH = 0.010
_SCREW_RADIUS = 0.0028
_SCREW_LENGTH = 0.004


def _motor_disc_radius(effort_nm: float) -> float:
    """Pancake QDD motor housing radius, scaled from the joint's rated torque
    (purely cosmetic: real MIT-Cheetah/Unitree-style QDD actuators visibly
    grow with torque rating). Does not touch the ImplicitActuatorCfg
    effort_limit actually enforced in sim — that still comes from
    robot_params.yaml via quadruped_env_cfg.py."""
    return 0.030 * math.sqrt(effort_nm / 10.0)


def _add_leg_ribs(link: ET.Element, radius: float, length: float) -> None:
    """Three raised rings along the tube — reads as ribbed structural reinforcement."""
    for frac in (0.30, 0.55, 0.80):
        _visual_cylinder(
            link, radius * _RIB_RADIUS_SCALE, _RIB_THICKNESS,
            xyz=(0.0, 0.0, -length * frac),
            material="panel_dark",
        )


def _add_cable_channel(link: ET.Element, radius: float, length: float) -> None:
    """Raised strip along the front face — reads as a cable guide / spine."""
    _visual_box(
        link, (_CHANNEL_THICKNESS, _CHANNEL_WIDTH, length * 0.82),
        xyz=(radius + _CHANNEL_THICKNESS / 2.0, 0.0, -length / 2.0),
        material="panel_dark",
    )


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
        _add_material_library(self.root, _MATERIALS)
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

        # Load-bearing chassis — the only box on this link with a matching
        # <collision>. Everything below is a bolt-on visual extra: no
        # collision, no mass, so it cannot change contact detection, total
        # mass, COM, or inertia.
        _visual_box(link, (lx, ly, lz), material="chassis_dark")
        _collision_box(link, (lx, ly, lz))

        # ── Top cover: electronics bay lid with an accent stripe ────────────
        cover_h = lz * 0.35
        cover_top_z = lz / 2.0 + cover_h
        _visual_box(
            link, (lx * 0.88, ly * 0.90, cover_h),
            xyz=(0.0, 0.0, lz / 2.0 + cover_h / 2.0),
            material="chassis_accent",
        )

        # ── Side panels + cooling vent slats ─────────────────────────────────
        panel_t = 0.006
        n_vents = 4
        vent_span = lx * 0.5
        for y_sign in (+1, -1):
            panel_y = y_sign * (ly / 2.0 + panel_t / 2.0)
            _visual_box(
                link, (lx * 0.82, panel_t, lz * 0.75),
                xyz=(0.0, panel_y, 0.0),
                material="panel_dark",
            )
            vent_y = y_sign * (ly / 2.0 + panel_t + 0.001)
            for i in range(n_vents):
                vx = -vent_span / 2.0 + i * (vent_span / (n_vents - 1))
                _visual_box(
                    link, (0.012, 0.002, lz * 0.35),
                    xyz=(vx, vent_y, 0.0),
                    material="sensor_black",
                )

        # ── Front sensor fascia: stereo camera housing + lens pair ──────────
        cam_face_x = lx / 2.0 + 0.022
        _visual_box(
            link, (0.022, ly * 0.55, lz * 0.55),
            xyz=(lx / 2.0, 0.0, lz * 0.05),
            material="sensor_black",
        )
        for y_sign in (+1, -1):
            _visual_sphere(
                link, 0.007,
                xyz=(cam_face_x, y_sign * ly * 0.16, lz * 0.05),
                material="lens_glass",
            )

        # ── LiDAR puck on the roof ────────────────────────────────────────────
        _visual_cylinder(
            link, 0.032, 0.022,
            xyz=(lx * 0.03, 0.0, cover_top_z + 0.011),
            material="sensor_black",
        )

        # ── IMU cover plate ──────────────────────────────────────────────────
        _visual_box(
            link, (0.022, 0.022, 0.008),
            xyz=(lx * 0.13, 0.0, cover_top_z + 0.004),
            material="chassis_accent",
        )

        # ── Status LED ────────────────────────────────────────────────────────
        _visual_sphere(
            link, 0.005,
            xyz=(lx * 0.36, 0.0, cover_top_z + 0.003),
            material="led_green",
        )

        # ── Rear fascia + tail light ──────────────────────────────────────────
        _visual_box(
            link, (0.018, ly * 0.5, lz * 0.5),
            xyz=(-(lx / 2.0 + 0.009), 0.0, 0.0),
            material="panel_dark",
        )
        _visual_sphere(
            link, 0.006,
            xyz=(-(lx / 2.0 + 0.019), 0.0, 0.0),
            material="led_red",
        )

        # ── Carry handle (rear-mounted, clear of the front sensor stack) ────
        handle_x = -lx * 0.15
        handle_span = ly * 0.5
        handle_h = 0.028
        handle_r = 0.006
        bar_z = cover_top_z + handle_h
        _visual_cylinder(
            link, handle_r, handle_span,
            xyz=(handle_x, 0.0, bar_z),
            rpy=(self.HALF_PI, 0.0, 0.0),
            material="panel_dark",
        )
        for y_sign in (+1, -1):
            _visual_cylinder(
                link, handle_r, handle_h,
                xyz=(handle_x, y_sign * handle_span / 2.0, cover_top_z + handle_h / 2.0),
                material="panel_dark",
            )

        # ── Fastener heads at the top-cover corners ──────────────────────────
        for x_sign in (+1, -1):
            for y_sign in (+1, -1):
                _visual_cylinder(
                    link, _SCREW_RADIUS, _SCREW_LENGTH,
                    xyz=(
                        x_sign * lx * 0.40,
                        y_sign * ly * 0.42,
                        cover_top_z + _SCREW_LENGTH / 2.0,
                    ),
                    material="actuator_silver",
                )

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
            material="structural_grey",
        )
        _collision_cylinder(
            hip_link, r_hip, l_hip,
            xyz=com_hip,
            rpy=(self.HALF_PI, 0.0, 0.0),
        )

        # Pancake QDD motor housing at the hip joint itself (joint axis = x,
        # so the disc's flat face — its z-axis before rotation — must point
        # along x: rotate 90° about y).
        hip_disc_r = _motor_disc_radius(lim["hip_abduction"]["effort_nm"])
        _visual_cylinder(
            hip_link, hip_disc_r, _MOTOR_DISC_LEN, xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, self.HALF_PI, 0.0), material="actuator_silver",
        )
        _visual_cylinder(
            hip_link, hip_disc_r + 0.004, _MOTOR_END_CAP_LEN, xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, self.HALF_PI, 0.0), material="sensor_black",
        )
        _visual_cylinder(
            hip_link, r_hip * 1.12, 0.006, xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, self.HALF_PI, 0.0), material="chassis_accent",
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
            material="structural_grey",
        )
        _collision_cylinder(thigh_link, r_thigh, l_thigh, xyz=com_thigh)

        leg_disc_r = _motor_disc_radius(lim["thigh"]["effort_nm"])
        _visual_cylinder(
            thigh_link, leg_disc_r, _MOTOR_DISC_LEN, xyz=(0.0, 0.0, 0.0),
            rpy=(self.HALF_PI, 0.0, 0.0), material="actuator_silver",
        )
        _visual_cylinder(
            thigh_link, leg_disc_r + 0.004, _MOTOR_END_CAP_LEN, xyz=(0.0, 0.0, 0.0),
            rpy=(self.HALF_PI, 0.0, 0.0), material="sensor_black",
        )
        _visual_cylinder(
            thigh_link, r_thigh * 1.12, 0.006, xyz=(0.0, 0.0, 0.0),
            rpy=(self.HALF_PI, 0.0, 0.0), material="chassis_accent",
        )
        _add_leg_ribs(thigh_link, r_thigh, l_thigh)
        _add_cable_channel(thigh_link, r_thigh, l_thigh)

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
            material="structural_grey",
        )
        _collision_cylinder(shin_link, r_shin, l_shin, xyz=com_shin)

        _visual_cylinder(
            shin_link, leg_disc_r, _MOTOR_DISC_LEN, xyz=(0.0, 0.0, 0.0),
            rpy=(self.HALF_PI, 0.0, 0.0), material="actuator_silver",
        )
        _visual_cylinder(
            shin_link, leg_disc_r + 0.004, _MOTOR_END_CAP_LEN, xyz=(0.0, 0.0, 0.0),
            rpy=(self.HALF_PI, 0.0, 0.0), material="sensor_black",
        )
        _visual_cylinder(
            shin_link, r_shin * 1.12, 0.006, xyz=(0.0, 0.0, 0.0),
            rpy=(self.HALF_PI, 0.0, 0.0), material="chassis_accent",
        )
        _add_leg_ribs(shin_link, r_shin, l_shin)
        _add_cable_channel(shin_link, r_shin, l_shin)

        # Ankle accent ring, just above the foot.
        _visual_cylinder(
            shin_link, r_shin * 1.12, 0.006,
            xyz=(0.0, 0.0, -l_shin + 0.01),
            material="chassis_accent",
        )

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
        _visual_sphere(foot_link, r_foot, material="rubber_black")
        _collision_sphere(foot_link, r_foot)

        # Rubber foot pad: a flattened disc visually wrapping the contact
        # sphere — same contact point, purely cosmetic bulge.
        _visual_cylinder(
            foot_link, r_foot * 1.3, r_foot * 0.5,
            xyz=(0.0, 0.0, -r_foot * 0.15),
            material="rubber_black",
        )

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
