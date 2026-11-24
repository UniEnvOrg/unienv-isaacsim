"""Space metadata of the nodes' observations and actions, in UniEnv's vocabulary
(``unienv_interface.space.space_utils.metadata_utils``).

Pure functions of plain inputs (names, DOF types as ``"rotation"`` / ``"translation"``): this module imports nothing
from Isaac Sim / Isaac Lab, so it can be used and tested without them. The nodes read the inputs from their entities
(``ArticulationNode.dof_types``: PhysX's ``shared_metatype.dof_types``).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from unienv_interface.space.space_utils.metadata_utils import action_metadata, metadata, proprio_metadata

__all__ = ["ROTATION", "TRANSLATION", "DOF_UNITS", "POSE_NAMES", "POSE_UNITS", "dof_units", "joint_state_metadata",
           "joint_position_target_metadata", "root_pose_metadata", "contact_force_metadata"]  # fmt: skip

ROTATION, TRANSLATION = "rotation", "translation"
DOF_UNITS: dict[str, dict[str, str]] = {
    ROTATION: {"position": "rad", "velocity": "rad/s", "force": "N*m"},
    TRANSLATION: {"position": "m", "velocity": "m/s", "force": "N"},
}
"""``DOF type -> quantity -> unit`` (Isaac Lab's joint data is in rad for revolute and m for prismatic DOFs)."""

POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]
POSE_UNITS = ["m"] * 3 + ["normalized"] * 4


def dof_units(dof_types: Sequence[str | None] | None, quantity: str) -> list[str] | None:
    """The unit of ``quantity`` (``"position"``, ``"velocity"`` or ``"force"``) of each DOF; None if the DOF types
    are not known (None, or any of them not ``"rotation"`` / ``"translation"``)."""
    if dof_types is None or any(t not in DOF_UNITS for t in dof_types):
        return None
    return [DOF_UNITS[t][quantity] for t in dof_types]


def joint_state_metadata(names: Sequence[str], dof_types: Sequence[str | None] | None = None,
                         **extra: Any) -> dict[str, dict[str, Any]]:  # fmt: skip
    """``kind="proprio"`` metadata of ``joint_position`` and ``joint_velocity``: joint names, and units if the DOF
    types are known."""
    return {f"joint_{q}": proprio_metadata(units=dof_units(dof_types, q), names=list(names), **extra)
            for q in ("position", "velocity")}  # fmt: skip


def joint_position_target_metadata(names: Sequence[str], dof_types: Sequence[str | None] | None = None,
                                   **extra: Any) -> dict[str, Any]:  # fmt: skip
    """``kind="action"`` metadata of absolute joint position targets: joint names, and units if the DOF types are
    known."""
    return action_metadata(control="joint_position", mode="absolute", units=dof_units(dof_types, "position"),
                           names=list(names), **extra)  # fmt: skip


def root_pose_metadata(**extra: Any) -> dict[str, Any]:
    """``kind="pose"`` metadata of a root pose (position, wxyz quaternion) in the simulation world frame, as Isaac
    Lab's ``root_pose_w``: the env origins (``scene.env_origins``) are not subtracted."""
    return metadata("pose", frame="world", rotation="quat_wxyz", units=POSE_UNITS, names=POSE_NAMES,
                    label="root pose in the simulation world frame (env origins not subtracted)", **extra)  # fmt: skip


def contact_force_metadata(bodies: Sequence[str] | None = None, **extra: Any) -> dict[str, Any]:
    """``kind="force"`` metadata of world frame forces (..., bodies, 3); the body of each row is in ``bodies``."""
    return metadata("force", units="N", frame="world", names=["fx", "fy", "fz"],
                    bodies=None if bodies is None else list(bodies), **extra)  # fmt: skip
