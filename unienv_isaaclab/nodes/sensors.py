"""Sensors of the scene."""

from __future__ import annotations

from isaaclab.sensors import ContactSensor

from .base import SceneEntityNode


class ContactSensorNode(SceneEntityNode):
    """A contact sensor (``world.scene[entity]``); Isaac Lab updates its buffers in ``scene.update``."""

    entity: ContactSensor
