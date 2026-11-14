"""IsaacLabVecEnv: a WorldEnv over an IsaacLabWorld with Isaac Lab's batched-env step semantics.

UniEnv's WorldEnv leaves resetting finished envs to the caller and reads the observation before the reward. RL
libraries for Isaac Lab (rsl_rl, rl_games, FlashSAC's loop) expect DirectRLEnv's behaviour instead, which this class
provides:

    step(action):
        set_next_action(action)
        for each update substep: pre-step hooks, world.step() (one physics step each), post-step hooks
        episode_length_buf += 1
        termination, truncation, reward, info      (read before any reset, in this order)
        reset the envs that terminated or truncated (world.reset(mask) + the nodes' reset hooks; no after_reset,
                                                     so no sim.forward, as DirectRLEnv.step)
        observation                                 (built once, after the resets: a reset env's first observation)

``reset()`` is WorldEnv's: the full lifecycle including after_reset (world: write to sim, forward, camera
re-renders). ``step``'s ``info`` merges the nodes' info dicts.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from unienv_interface.world import WorldNode
from unienv_interface.world.env_composer import WorldEnv
from unienv_interface.world.nodes.combined_node import CombinedWorldNode

from .world import IsaacLabWorld


class IsaacLabVecEnv(WorldEnv):
    world: IsaacLabWorld

    def __init__(self, world: IsaacLabWorld, nodes: WorldNode | Iterable[WorldNode]):
        super().__init__(world, nodes, render_mode=None)
        self._first_reset = False  # the world is built in its constructor; reload is not used
        # physics steps per env step: the world's substep index runs 0 .. decimation - 1 within a step
        self.decimation = self._n_update_substeps * self._n_world_substeps
        world.decimation = self.decimation

    @property
    def num_envs(self) -> int:
        return self.world.num_envs

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.world.episode_length_buf

    def _merged_info(self) -> dict[str, Any]:
        info = self.node.get_info() or {}
        if not isinstance(self.node, CombinedWorldNode):
            return dict(info)
        merged: dict[str, Any] = {}
        for node_info in info.values():
            merged.update(node_info)
        return merged

    def _signal(self, has: bool, get, dtype: torch.dtype) -> torch.Tensor:
        return get() if has else torch.zeros(self.num_envs, dtype=dtype, device=self.device)

    def _run_resets(self, mask: torch.Tensor) -> None:
        self.world.reset(mask=mask)
        for p in self._get_priority_order("reset"):
            self.node.reset(priority=p, mask=mask)

    def step(self, action: Any):
        """One env step (``decimation`` physics steps); returns (observation, reward, terminated, truncated, info)
        with the observation after the resets of the envs that finished."""
        if self.node.action_space is not None:
            self.node.set_next_action(action)
        pre, post = self._get_priority_order("pre_step"), self._get_priority_order("post_step")
        k = 0
        for _ in range(self._n_update_substeps):
            self.world.substep = k
            for p in pre:
                self.node.pre_environment_step(self._update_dt, priority=p)
            for _ in range(self._n_world_substeps):
                self.world.substep = k
                self.world.step()
                k += 1
            for p in post:
                self.node.post_environment_step(self._update_dt, priority=p)
        self.world.episode_length_buf += 1

        n = self.node
        terminated = self._signal(n.has_termination_signal, n.get_termination, torch.bool)
        truncated = self._signal(n.has_truncation_signal, n.get_truncation, torch.bool)
        reward = self._signal(n.has_reward, n.get_reward, torch.float32)
        info = self._merged_info()
        done = terminated | truncated
        if done.any():
            self._run_resets(done)
        obs = n.get_observation() if n.observation_space is not None else None
        return obs, reward, terminated, truncated, info
