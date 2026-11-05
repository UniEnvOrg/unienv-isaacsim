"""Smoke test of unienv_isaaclab: 4 cubes dropped onto a ground plane, one task node at 30 Hz over 120 Hz physics.

Checks: the world builds and steps; an env step is decimation physics steps; envs that finish are reset inside step
and return their new episode's first observation; the episode counter restarts; reset() and close() work. (Isaac Sim
supports one simulation per app process in practice: making a second world after close() hangs in Kit.)

    python scripts/smoke.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from unienv_interface.backends.pytorch import PyTorchComputeBackend  # noqa: E402
from unienv_interface.space import BoxSpace  # noqa: E402

from unienv_isaaclab.nodes import RigidObjectNode  # noqa: E402
from unienv_isaaclab.vec_env import IsaacLabVecEnv  # noqa: E402
from unienv_isaaclab.world import IsaacLabWorld  # noqa: E402

PHYSICS_HZ, DECIMATION, EPISODE = 120, 4, 10  # episode: env steps
START_Z = 0.5


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    cube = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube",
                          spawn=sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                                                    collision_props=sim_utils.CollisionPropertiesCfg()),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, START_Z)))


class DropCube(RigidObjectNode):
    """Drops the cube from START_Z (+ env index cm) at each reset; observation: its height; reward: -height;
    episodes end after EPISODE env steps (env 0 also terminates at its 3rd step)."""

    reset_priorities = {50}
    pre_environment_step_priorities = {50}
    has_reward = has_termination_signal = has_truncation_signal = True

    def __init__(self, world):
        super().__init__(world, "drop", entity="cube")
        self.control_timestep = DECIMATION * world.physics_dt
        self.observation_space = BoxSpace(PyTorchComputeBackend, -10.0, 10.0, torch.float32, shape=(world.num_envs, 1),
                                          device=world.device)
        self.action_space = BoxSpace(PyTorchComputeBackend, -1.0, 1.0, torch.float32, shape=(world.num_envs, 1),
                                     device=world.device)
        self.pre_steps = 0

    def set_next_action(self, action):
        self.action = action

    def pre_environment_step(self, dt, *, priority=0):
        self.pre_steps += 1

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        ids = self.world.reset_ids
        pose = self.entity.data.default_root_state[ids, :7].clone()
        pose[:, :3] += self.world.env_origins[ids]
        pose[:, 2] = START_Z + 0.01 * ids.float()
        self.write_root_state(pose, torch.zeros(len(ids), 6, device=self.world.device), ids)

    def height(self):
        return self.entity.data.root_pos_w[:, 2:3]

    def get_observation(self):
        return self.height().clone()

    def get_reward(self):
        return -self.height()[:, 0]

    def get_termination(self):
        out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.world.device)
        out[0] = self.world.episode_length_buf[0] == 3
        return out

    def get_truncation(self):
        return self.world.episode_length_buf >= EPISODE


def run_once() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, render_interval=DECIMATION)
    world = IsaacLabWorld(sim_cfg, SceneCfg(num_envs=4, env_spacing=1.0), seed=0)
    node = DropCube(world)
    env = IsaacLabVecEnv(world, [node])
    assert env.decimation == DECIMATION, env.decimation
    _, obs, _ = env.reset()
    assert torch.allclose(obs[:, 0], START_Z + 0.01 * torch.arange(4, device=world.device), atol=1e-5), obs
    for t in range(1, 2 * EPISODE + 1):
        obs, reward, terminated, truncated, _ = env.step(torch.zeros(4, 1, device=world.device))
        done = terminated | truncated
        if done.any():  # these envs were reset inside step: their observation is the new episode's first
            ids = done.nonzero().squeeze(-1)
            assert torch.allclose(obs[ids, 0], START_Z + 0.01 * ids.float(), atol=1e-5), (t, obs)
            assert (world.episode_length_buf[ids] == 0).all()
        assert (reward < 0).all()
        print(f"step {t}: heights {[round(float(h), 3) for h in obs[:, 0]]} done {done.tolist()}", flush=True)
    assert world.step_count == 2 * EPISODE * DECIMATION, world.step_count
    assert node.pre_steps == 2 * EPISODE, node.pre_steps  # a 30 Hz node's pre-step runs once per env step
    env.close()


if __name__ == "__main__":
    import os
    run_once()
    print("SMOKE PASS", flush=True)
    os._exit(0)  # Kit's own shutdown can hang
