"""Nodes over one entity of an IsaacSimWorld's scene (an articulation, a rigid object, a sensor).

They hold the entity handle and the world; task logic belongs in subclasses or other nodes. Lifecycle priorities
follow UniEnv's bands (higher runs first): robots and objects 50-80, tasks 0-50, sensors below 0. A node only takes
part in the phases whose priority sets are non-empty, so these bases declare none; subclasses add what they use.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import WorldNode

from ..world import IsaacSimWorld


class SceneEntityNode(WorldNode):
    """A node bound to ``world.scene[entity]``."""

    def __init__(self, world: IsaacSimWorld, name: str, entity: str | None = None):
        self.world, self.name = world, name
        self.entity: Any = world.scene[entity or name]


def unbounded_obs_space(world: IsaacSimWorld, metadata: Mapping[str, Mapping[str, Any]] | None = None,
                        **shapes: tuple[int, ...]) -> DictSpace:  # fmt: skip
    """A dict of unbounded float boxes, one per key, each (num_envs, *shape); ``metadata``: the space metadata of some
    of the keys (see unienv_isaacsim.space_metadata)."""
    metadata = metadata or {}
    if unknown := set(metadata) - set(shapes):
        raise KeyError(f"metadata for keys without a shape: {sorted(unknown)}")
    boxes = {k: BoxSpace(PyTorchComputeBackend, -torch.inf, torch.inf, torch.float32, device=world.device,
                         shape=(world.num_envs, *s), metadata=metadata.get(k)) for k, s in shapes.items()}
    return DictSpace(PyTorchComputeBackend, boxes, device=world.device)
