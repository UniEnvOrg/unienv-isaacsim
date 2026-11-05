"""Articulations and rigid objects of the scene."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject

from .base import SceneEntityNode


class ArticulationNode(SceneEntityNode):
    """An articulation (``world.scene[entity]``): joint and body lookup by name in a given order."""

    entity: Articulation

    def find_joint_ids(self, names: Sequence[str]) -> list[int]:
        return self.entity.find_joints(list(names), preserve_order=True)[0]

    def find_body_ids(self, names: Sequence[str]) -> list[int]:
        return self.entity.find_bodies(list(names), preserve_order=True)[0]

    def write_joint_state(self, q: torch.Tensor, dq: torch.Tensor, env_ids: torch.Tensor,
                          joint_ids: Sequence[int] | None = None) -> None:
        self.entity.write_joint_state_to_sim(q, dq, joint_ids=joint_ids, env_ids=env_ids)


class RigidObjectNode(SceneEntityNode):
    """A rigid object (``world.scene[entity]``): root pose and velocity in the world frame."""

    entity: RigidObject

    def write_root_state(self, pose_w: torch.Tensor, vel_w: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Pose (n, 7: position, wxyz) and velocity (n, 6: linear, angular), world frame."""
        self.entity.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        self.entity.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
