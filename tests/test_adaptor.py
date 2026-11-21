"""IsaacSimWorld, IsaacSimVecEnv and the generic nodes on 4 envs (needs a GPU): a cube dropped onto the ground and a
one-joint arm, one task node at 30 Hz over 120 Hz physics.

Isaac Sim supports one simulation per process in practice (a second world after close() hangs in Kit), so both tests
share one world.
"""

import isaaclab.sim as sim_utils
import pytest
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

from unienv_isaacsim import (
    ContactForceNode,
    IsaacSimVecEnv,
    IsaacSimWorld,
    JointPositionTargetNode,
    JointStateNode,
    RigidObjectNode,
    RootPoseNode,
)

PHYSICS_HZ, DECIMATION, EPISODE = 120, 4, 30  # episode: env steps
START_Z, CUBE, MASS = 0.5, 0.1, 0.1
N = 4

ARM_URDF = """<robot name="arm">
  <link name="base"><inertial><mass value="1"/><inertia ixx="1e-3" iyy="1e-3" izz="1e-3" ixy="0" ixz="0" iyz="0"/>
    </inertial></link>
  <link name="link"><inertial><origin xyz="0.1 0 0"/><mass value="0.1"/>
    <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    <visual><origin xyz="0.1 0 0"/><geometry><box size="0.2 0.02 0.02"/></geometry></visual></link>
  <joint name="hinge" type="revolute"><parent link="base"/><child link="link"/><axis xyz="0 0 1"/>
    <limit lower="-1.5" upper="1.5" effort="100" velocity="10"/></joint>
</robot>"""


def scene_cfg(urdf_path: str) -> InteractiveSceneCfg:
    @configclass
    class SceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(size=(CUBE,) * 3, rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=MASS),
                                      collision_props=sim_utils.CollisionPropertiesCfg(),
                                      activate_contact_sensors=True),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, START_Z)))
        cube_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Cube")
        arm = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Arm",
            spawn=sim_utils.UrdfFileCfg(  # a force drive; without joint_drive the importer makes it an acceleration one
                asset_path=urdf_path, fix_base=True,
                joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                    gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=10.0, damping=1.0))),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.4, 0.3)),
            actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=["hinge"], stiffness=10.0, damping=1.0)})

    return SceneCfg(num_envs=N, env_spacing=1.0)


class DropCube(RigidObjectNode):
    """Drops the cube from START_Z (+ env index cm) at each reset; reward: -height; episodes end after EPISODE env
    steps (env 0 also terminates at every 3rd step)."""

    reset_priorities = {50}
    pre_environment_step_priorities = {50}
    has_reward = has_termination_signal = has_truncation_signal = True

    def __init__(self, world):
        super().__init__(world, "drop", entity="cube")
        self.control_timestep = DECIMATION * world.physics_dt
        self.pre_steps = 0

    def pre_environment_step(self, dt, *, priority=0):
        self.pre_steps += 1

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        ids = self.world.reset_ids
        pose = self.entity.data.default_root_state[ids, :7].clone()
        pose[:, :3] += self.world.env_origins[ids]
        pose[:, 2] = START_Z + 0.01 * ids.float()
        self.write_root_state(pose, torch.zeros(len(ids), 6, device=self.world.device), ids)

    def get_reward(self):
        return -self.entity.data.root_pos_w[:, 2]

    def get_termination(self):
        out = torch.zeros(N, dtype=torch.bool, device=self.world.device)
        out[0] = self.world.episode_length_buf[0] == 3
        return out

    def get_truncation(self):
        return self.world.episode_length_buf >= EPISODE


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    urdf = tmp_path_factory.mktemp("arm") / "arm.urdf"
    urdf.write_text(ARM_URDF)
    world = IsaacSimWorld(sim_utils.SimulationCfg(dt=1.0 / PHYSICS_HZ, render_interval=DECIMATION),
                          scene_cfg(str(urdf)), seed=0)
    nodes = [DropCube(world), JointStateNode(world, "arm_state", entity="arm"),
             JointPositionTargetNode(world, "arm_target", entity="arm"),
             RootPoseNode(world, "cube_pose", entity="cube"), ContactForceNode(world, "cube_contact")]
    env = IsaacSimVecEnv(world, nodes)
    yield env
    env.close()


def action(env, target: float) -> torch.Tensor:
    """The hinge target; with one action node, UniEnv passes the action through without a dict."""
    return torch.full((N, 1), target, device=env.device)


def test_step_resets_finished_envs(env):
    """An env step is DECIMATION physics steps; envs that finish are reset inside step and return their new
    episode's first observation; the episode counter restarts."""
    world, drop = env.world, env.node.get_node("drop")
    steps0, pre0 = world.step_count, drop.pre_steps
    assert env.decimation == world.decimation == DECIMATION
    _, obs, _ = env.reset()
    z = obs["cube_pose"]["root_pose"][:, 2]
    assert torch.allclose(z, START_Z + 0.01 * torch.arange(N, device=world.device), atol=1e-5), z
    for t in range(1, 2 * EPISODE + 1):
        obs, reward, terminated, truncated, _ = env.step(action(env, 0.0))
        done = terminated | truncated
        if done.any():
            ids, z = done.nonzero().squeeze(-1), obs["cube_pose"]["root_pose"][:, 2]
            assert torch.allclose(z[ids], START_Z + 0.01 * ids.float(), atol=1e-5), (t, z)
            assert (world.episode_length_buf[ids] == 0).all()
        assert (reward < 0).all()
    assert world.step_count - steps0 == 2 * EPISODE * DECIMATION
    assert drop.pre_steps - pre0 == 2 * EPISODE  # a 30 Hz node's pre-step runs once per env step


def test_nodes(env):
    """The joint target moves the hinge and the joint state reads it; the resting cube's pose and contact force."""
    env.reset()
    for _ in range(20):  # 0.67 s: the hinge has settled and the cubes of envs 1-3 have landed (env 0's is
        obs, *_ = env.step(action(env, 0.5))  # dropped again every 3 steps)
    assert torch.allclose(obs["arm_state"]["joint_position"], torch.full((N, 1), 0.5, device=env.device), atol=0.02)
    z = obs["cube_pose"]["root_pose"][1:, 2]
    assert torch.allclose(z, torch.full_like(z, CUBE / 2), atol=5e-3), z
    force_z = obs["cube_contact"]["net_force"][1:, 0, 2]
    assert torch.allclose(force_z, torch.full_like(force_z, MASS * 9.81), rtol=0.1), force_z
