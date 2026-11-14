"""IsaacLabWorld: UniEnv's World over Isaac Lab's SimulationContext and InteractiveScene.

It does what isaaclab.envs.DirectRLEnv does around its task code (DirectRLEnv.__init__, step's physics loop, reset,
close), with one World.step() being one physics step. The scene is built in the constructor, because nodes need the
scene's entity handles when they are made and an Isaac stage cannot be rebuilt, so reload is not supported.

Counters the nodes share live here, as in DirectRLEnv: ``episode_length_buf`` (env steps since each env's reset; the
env adapter increments it once per env step), ``decimation`` (physics steps per env step), ``substep`` (the physics
step within the current env step, 0-based) and ``step_count`` (physics steps since start; drives the render interval).
"""

from __future__ import annotations

import builtins
from typing import Any, Callable

import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.ui import ViewportCameraController
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.utils.stage import attach_stage_to_usd_context, use_stage
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.version import get_isaac_sim_version
from unienv_interface.backends.pytorch import PyTorchComputeBackend
from unienv_interface.world import World


class IsaacLabWorld(World):
    backend = PyTorchComputeBackend

    def __init__(self, sim_cfg: SimulationCfg, scene_cfg: InteractiveSceneCfg, *, seed: int | None = None,
                 viewer_cfg: ViewerCfg | None = ViewerCfg(), num_rerenders_on_reset: int = 0,
                 setup_scene: Callable[[InteractiveScene], None] | None = None):
        """Create the simulation and the scene and start the simulation.

        Args:
            sim_cfg: the simulation settings; ``sim_cfg.render_interval`` (physics steps between renders) is read at
                every step, so it can be changed afterwards.
            scene_cfg: the scene (its ``num_envs`` is the world's batch size).
            seed: seeds replicator, torch, numpy and random (DirectRLEnv.seed), before the simulation exists.
            viewer_cfg: the viewport camera when the app renders; None: no viewport camera controller.
            num_rerenders_on_reset: renders after a full reset when there are RTX sensors (cameras).
            setup_scene: called with the scene before the simulation starts (DirectRLEnv._setup_scene), e.g. to
                clone environments or add lights.
        """
        self._is_closed = False
        if seed is not None:
            configure_seed(seed)
            try:
                import omni.replicator.core as rep
                rep.set_global_seed(seed)
            except ModuleNotFoundError:
                pass
        if SimulationContext.instance() is not None:
            raise RuntimeError("A simulation context already exists; Isaac Lab allows one per process.")
        self.sim_cfg = sim_cfg
        self.sim = SimulationContext(sim_cfg)
        if "cuda" in self.sim.device:
            torch.cuda.set_device(self.sim.device)
        with use_stage(self.sim.get_initial_stage()):
            self.scene = InteractiveScene(scene_cfg)
            if setup_scene is not None:
                setup_scene(self.scene)
            attach_stage_to_usd_context()
        self.viewport_camera_controller = None
        if viewer_cfg is not None and self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, viewer_cfg)  # reads num_envs, scene, sim
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            with use_stage(self.sim.get_initial_stage()):
                self.sim.reset()  # PLAY: assets and sensors initialize their handles
            self.scene.update(dt=self.physics_dt)
        self.num_rerenders_on_reset = num_rerenders_on_reset

        self.world_timestep = self.physics_dt
        self.batch_size = self.scene.num_envs
        self.step_count = 0
        self.decimation = 1  # physics steps per env step; IsaacLabVecEnv sets it from the nodes' timesteps
        self.substep = 0
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # the envs being reset (long, (n,)), set by reset() for the nodes' reset hooks
        self.reset_ids = torch.arange(self.num_envs, device=self.device)

    # -- properties ---------------------------------------------------------------------------------------------------

    @property
    def device(self) -> str:
        return self.sim.device

    @property
    def num_envs(self) -> int:
        return self.scene.num_envs

    @property
    def physics_dt(self) -> float:
        return self.sim_cfg.dt

    @property
    def env_origins(self) -> torch.Tensor:
        return self.scene.env_origins

    def is_control_timestep_compatible(self, control_timestep: float | None) -> bool:
        """Whether a period is a whole number of physics steps (the base class's float modulo can fail on it)."""
        if control_timestep is None:
            return True
        n = control_timestep / self.world_timestep
        return abs(n - round(n)) < 1e-6 and round(n) >= 1

    # -- World interface ----------------------------------------------------------------------------------------------

    def step(self) -> float:
        """One physics step: write the assets' buffers to the simulation, step physics, render every
        ``sim_cfg.render_interval`` physics steps when something renders (GUI or RTX sensors), update the scene's
        buffers (DirectRLEnv.step's inner loop)."""
        self.step_count += 1
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if self.step_count % self.sim_cfg.render_interval == 0 and (self.sim.has_gui() or self.sim.has_rtx_sensors()):
            self.sim.render()
        self.scene.update(dt=self.physics_dt)
        return self.physics_dt

    def reset(self, *, seed: int | None = None, mask: torch.Tensor | None = None, **kwargs: Any) -> None:
        """Reset the scene's assets and sensors and the episode counters of the envs in ``mask`` (bool, (N,); None:
        all), as DirectRLEnv._reset_idx's base part. Sets ``reset_ids`` for the nodes."""
        if seed is not None:
            configure_seed(seed)
        self.reset_ids = torch.arange(self.num_envs, device=self.device) if mask is None else \
            mask.nonzero().squeeze(-1)
        self.scene.reset(self.reset_ids)
        self.episode_length_buf[self.reset_ids] = 0

    def after_reset(self, *, seed: int | None = None, mask: torch.Tensor | None = None, **kwargs: Any) -> None:
        """After a full reset: push the written states to the simulation and update its kinematics, then re-render
        cameras (DirectRLEnv.reset). Not called for the resets inside an env step, as in DirectRLEnv.step."""
        self.scene.write_data_to_sim()
        self.sim.forward()
        if self.sim.has_rtx_sensors():
            for _ in range(self.num_rerenders_on_reset):
                self.sim.render()

    def reload(self, *, seed: int | None = None, mask: torch.Tensor | None = None, **kwargs: Any) -> None:
        raise NotImplementedError("IsaacLabWorld builds its scene once, in its constructor")

    def close(self) -> None:
        """Release the scene and the simulation context (DirectRLEnv.close); a new world can be made afterwards."""
        if self._is_closed:
            return
        del self.scene
        if self.viewport_camera_controller is not None:
            del self.viewport_camera_controller
        if get_isaac_sim_version().major >= 5 and self.sim_cfg.create_stage_in_memory:
            import omni.physx
            omni.physx.get_physx_simulation_interface().detach_stage()
            self.sim.stop()
            self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
        self._is_closed = True
