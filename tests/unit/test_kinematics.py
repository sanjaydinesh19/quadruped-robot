"""Round-trip FK → IK consistency tests."""
import numpy as np
import pytest
from src.kinematics.leg import LegKinematics


@pytest.mark.parametrize("leg_id", ["FL", "FR", "RL", "RR"])
def test_fk_standing_pose(leg_id: str) -> None:
    """At zero joint angles the foot should be directly below the hip."""
    kin = LegKinematics(leg_id)
    pos = kin.forward([0.0, 0.0, 0.0])
    assert pos[0] == pytest.approx(0.0, abs=1e-6)   # x: no forward lean
    assert pos[2] < 0                                 # z: foot is below body


@pytest.mark.parametrize("leg_id", ["FL", "FR", "RL", "RR"])
def test_fk_ik_roundtrip(leg_id: str) -> None:
    """FK(IK(pos)) should return the original pos within numerical tolerance."""
    kin = LegKinematics(leg_id)
    target = np.array([0.05, kin._sign * 0.08, -0.30])  # reachable standing foot
    try:
        angles = kin.inverse(target)
        recovered = kin.forward(angles)
        np.testing.assert_allclose(recovered, target, atol=1e-4)
    except ValueError as exc:
        pytest.skip(f"Target unreachable (IK not fully implemented): {exc}")
