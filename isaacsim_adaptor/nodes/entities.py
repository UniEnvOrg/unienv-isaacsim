"""Articulations and rigid objects of the scene: handles, and ready-made state and target nodes on top of them."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation, RigidObject
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace

from ..world import IsaacSimWorld
from .base import SceneEntityNode, unbounded_obs_space


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

    def _select_joints(self, joint_names: Sequence[str] | None) -> None:
        """``joint_ids`` and ``num_joints`` of the named joints, in that order (None: all)."""
        self.joint_ids = self.find_joint_ids(joint_names) if joint_names else list(range(self.entity.num_joints))
        self.num_joints = len(self.joint_ids)


class JointStateNode(ArticulationNode):
    """Observation: ``joint_position`` and ``joint_velocity`` (N, n) of the chosen joints."""

    def __init__(self, world: IsaacSimWorld, name: str, entity: str | None = None,
                 joint_names: Sequence[str] | None = None):
        super().__init__(world, name, entity)
        self._select_joints(joint_names)
        self.observation_space = unbounded_obs_space(world, joint_position=(self.num_joints,),
                                                     joint_velocity=(self.num_joints,))

    def get_observation(self) -> dict[str, torch.Tensor]:
        d = self.entity.data
        return {"joint_position": d.joint_pos[:, self.joint_ids].clone(),
                "joint_velocity": d.joint_vel[:, self.joint_ids].clone()}


class JointPositionTargetNode(ArticulationNode):
    """Action: joint position targets (N, n) of the chosen joints, within their soft limits, for the articulation's
    own actuators (PD gains come from its config). Without an action the last target is kept; a reset env's target
    becomes its joint positions after the reset, so it holds still."""

    reset_priorities = {0}  # after the robots (50-80) have written their reset joint state
    pre_environment_step_priorities = {50}

    def __init__(self, world: IsaacSimWorld, name: str, entity: str | None = None,
                 joint_names: Sequence[str] | None = None, control_timestep: float | None = None):
        super().__init__(world, name, entity)
        self._select_joints(joint_names)
        self.control_timestep = control_timestep
        limits = self.entity.data.soft_joint_pos_limits[:, self.joint_ids]
        self.action_space = BoxSpace(PyTorchComputeBackend, limits[..., 0], limits[..., 1], torch.float32,
                                     device=world.device, shape=(world.num_envs, self.num_joints))
        self.target = self.entity.data.joint_pos[:, self.joint_ids].clone()

    def set_next_action(self, action: torch.Tensor | None) -> None:
        if action is not None:
            self.target = action.to(self.target, copy=True)  # reset() writes into it, never into the caller's

    def reset(self, *, priority: int = 0, seed: int | None = None, mask: torch.Tensor | None = None, **kwargs) -> None:
        ids = self.world.reset_ids
        self.target[ids] = self.entity.data.joint_pos[ids][:, self.joint_ids]

    def pre_environment_step(self, dt: float, *, priority: int = 0) -> None:
        self.entity.set_joint_position_target(self.target, joint_ids=self.joint_ids)


class RigidObjectNode(SceneEntityNode):
    """A rigid object (``world.scene[entity]``): root pose and velocity in the world frame."""

    entity: RigidObject

    def write_root_state(self, pose_w: torch.Tensor, vel_w: torch.Tensor, env_ids: torch.Tensor) -> None:
        """Pose (n, 7: position, wxyz) and velocity (n, 6: linear, angular), world frame."""
        self.entity.write_root_pose_to_sim(pose_w, env_ids=env_ids)
        self.entity.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


class RootPoseNode(RigidObjectNode):
    """Observation: ``root_pose`` (N, 7: position, wxyz quaternion) in the world frame."""

    def __init__(self, world: IsaacSimWorld, name: str, entity: str | None = None):
        super().__init__(world, name, entity)
        self.observation_space = unbounded_obs_space(world, root_pose=(7,))

    def get_observation(self) -> dict[str, torch.Tensor]:
        return {"root_pose": self.entity.data.root_pose_w.clone()}
