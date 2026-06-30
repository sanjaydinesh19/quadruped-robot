"""
Single-leg kinematics for a 3-DOF leg: hip (abduction), thigh (flexion), knee (flexion).

Coordinate convention (body frame):
  x = forward, y = left, z = up
  Joint 0 (hip/abduction): rotation about x-axis
  Joint 1 (thigh/flexion): rotation about y-axis
  Joint 2 (knee/flexion):  rotation about y-axis

Link lengths (placeholder values — update from robot config):
  l_hip_offset : lateral distance from body centre to hip joint
  l_thigh      : thigh link length (hip → knee)
  l_shin       : shin link length  (knee → foot)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

# fmt: off
_LEG_SIGNS = {
    #         lateral_sign  (determines which side the hip offset goes)
    "FL": +1,
    "FR": -1,
    "RL": +1,
    "RR": -1,
}
# fmt: on


class LegKinematics:
    def __init__(
        self,
        leg_id: str,
        l_hip_m: float = 0.062,    # hip abduction offset
        l_thigh_m: float = 0.209,  # thigh length
        l_shin_m: float = 0.195,   # shin length
    ) -> None:
        if leg_id not in _LEG_SIGNS:
            raise ValueError(f"leg_id must be one of {list(_LEG_SIGNS)}, got '{leg_id}'")
        self.leg_id = leg_id
        self.l_hip = l_hip_m
        self.l_thigh = l_thigh_m
        self.l_shin = l_shin_m
        self._sign = _LEG_SIGNS[leg_id]

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def forward(self, joint_angles_rad: list[float] | NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Return foot position [x, y, z] in body frame (metres)."""
        q0, q1, q2 = joint_angles_rad
        sign = self._sign

        # Hip abduction: moves foot laterally
        y_hip = sign * self.l_hip

        # Thigh + knee planar chain in the sagittal plane
        x_foot = self.l_thigh * np.sin(q1) + self.l_shin * np.sin(q1 + q2)
        z_chain = -(self.l_thigh * np.cos(q1) + self.l_shin * np.cos(q1 + q2))

        # Hip abduction rotates the entire lower chain around x-axis
        y_foot = y_hip + z_chain * np.sin(q0)
        z_foot = z_chain * np.cos(q0)

        return np.array([x_foot, y_foot, z_foot])

    # ------------------------------------------------------------------
    # Inverse kinematics  (analytical solution)
    # ------------------------------------------------------------------
    def inverse(self, foot_pos_m: list[float] | NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """
        Return joint angles [hip, thigh, knee] in radians for a desired foot
        position [x, y, z] in body frame.
        Raises ValueError if the target is unreachable.
        """
        x, y, z = foot_pos_m
        sign = self._sign

        # --- Hip (abduction) angle ---
        # Lateral distance after removing hip offset
        y_rel = y - sign * self.l_hip
        r_yz = np.sqrt(y_rel**2 + z**2)  # reach in the yz plane
        q0 = np.arctan2(-y_rel, -z)      # hip abduction

        # Distance from hip joint to foot projected into the sagittal plane
        r_sag = np.sqrt(x**2 + r_yz**2)

        reach_max = self.l_thigh + self.l_shin
        if r_sag > reach_max:
            raise ValueError(
                f"Target unreachable: distance {r_sag:.4f} m > max reach {reach_max:.4f} m"
            )

        # --- Knee angle (cosine rule) ---
        cos_q2 = (r_sag**2 - self.l_thigh**2 - self.l_shin**2) / (
            2 * self.l_thigh * self.l_shin
        )
        cos_q2 = np.clip(cos_q2, -1.0, 1.0)
        q2 = -np.arccos(cos_q2)  # knee bends backward → negative

        # --- Thigh angle ---
        # Standard 2-link planar IK in the sagittal plane.
        # alpha: angle from downward vertical to the foot target.
        # r_yz == |z_chain| is the downward distance after abduction (always > 0).
        # beta: angular contribution from the knee bend.
        alpha = np.arctan2(x, r_yz)
        beta = np.arctan2(
            self.l_shin * np.sin(q2),
            self.l_thigh + self.l_shin * np.cos(q2),
        )
        q1 = alpha - beta

        return np.array([q0, q1, q2])
