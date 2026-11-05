"""Nodes over one entity of an IsaacLabWorld's scene (an articulation, a rigid object, a sensor).

They hold the entity handle and the world; task logic belongs in subclasses or other nodes. Lifecycle priorities
follow UniEnv's bands (higher runs first): robots and objects 50-80, tasks 0-50, sensors below 0. A node only takes
part in the phases whose priority sets are non-empty, so these bases declare none; subclasses add what they use.
"""

from __future__ import annotations

from typing import Any

import torch
from unienv_interface.world import WorldNode

from ..world import IsaacLabWorld


class SceneEntityNode(WorldNode):
    """A node bound to ``world.scene[entity]``."""

    def __init__(self, world: IsaacLabWorld, name: str, entity: str | None = None):
        self.world, self.name = world, name
        self.entity_name = entity or name
        self.entity: Any = world.scene[self.entity_name]

    @property
    def num_envs(self) -> int:
        return self.world.num_envs

    def mask_ids(self, mask: torch.Tensor | None) -> torch.Tensor:
        """The env indices (long, (n,)) of a bool reset mask (None: all)."""
        if mask is None:
            return torch.arange(self.num_envs, device=self.world.device)
        return mask.nonzero().squeeze(-1)
