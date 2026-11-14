"""Sensors of the scene."""

from __future__ import annotations

import torch
from isaaclab.sensors import ContactSensor

from ..world import IsaacLabWorld
from .base import SceneEntityNode, unbounded_obs_space


class ContactSensorNode(SceneEntityNode):
    """A contact sensor (``world.scene[entity]``); Isaac Lab updates its buffers in ``scene.update``."""

    entity: ContactSensor


class ContactForceNode(ContactSensorNode):
    """Observation: ``net_force`` (N, bodies, 3), the net contact force on each of the sensor's bodies, world frame."""

    def __init__(self, world: IsaacLabWorld, name: str, entity: str | None = None):
        super().__init__(world, name, entity)
        self.observation_space = unbounded_obs_space(world, net_force=(self.entity.num_bodies, 3))

    def get_observation(self) -> dict[str, torch.Tensor]:
        return {"net_force": self.entity.data.net_forces_w.clone()}
