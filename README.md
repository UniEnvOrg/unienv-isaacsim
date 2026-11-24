# unienv-isaacsim

Isaac Lab adaptor for [UniEnv](https://github.com/UniEnvOrg/UniEnv)'s node system (package `unienv_isaacsim`), the
counterpart of [unienv-genesis](https://github.com/UniEnvOrg/unienv-genesis) for Genesis. Written against Isaac Lab
2.3.2 and Isaac Sim 5.1.0.

- `IsaacSimWorld`: owns Isaac Lab's `SimulationContext` and `InteractiveScene` and steps them one physics step at a
  time. It is built once in its constructor (an Isaac stage cannot be rebuilt), so `reload` is not supported.
- `IsaacSimVecEnv`: a `WorldEnv` with Isaac Lab's batched-env behaviour: `step` resets the envs that finished and
  returns their first observation of the new episode (rewards and dones are computed before that reset), the
  interface rsl_rl, rl_games and FlashSAC loops expect.
- Nodes over the scene's entities, keyed by their name in the scene:

  | node | observation / action |
  |---|---|
  | `JointStateNode` | `joint_position`, `joint_velocity` (N, n) |
  | `JointPositionTargetNode` | action: joint position targets (N, n), for the articulation's actuators |
  | `RootPoseNode` | `root_pose` (N, 7: position, wxyz), world frame |
  | `ContactForceNode` | `net_force` (N, bodies, 3), world frame |

  Their bases (`ArticulationNode`, `RigidObjectNode`, `ContactSensorNode`) hold the entity handle plus name lookups and
  state writes, for task nodes to subclass.

  Their spaces carry UniEnv space metadata (`unienv_isaacsim.space_metadata`): joint names (resolved from the
  `joint_names` patterns) with units from PhysX's DOF types (revolute: rad, prismatic: m), the root pose's
  representation (world frame: env origins are not subtracted) and the contact forces' frame and bodies.

Priorities follow UniEnv (higher runs first): scene geometry 80-100, robots and objects 50-80, tasks 0-50,
sensors below 0.

## Usage

```python
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app  # before any isaaclab / unienv_isaacsim import

import isaaclab.sim as sim_utils
from unienv_isaacsim import IsaacSimVecEnv, IsaacSimWorld, JointPositionTargetNode, JointStateNode

world = IsaacSimWorld(sim_utils.SimulationCfg(dt=1 / 120), MySceneCfg(num_envs=64, env_spacing=1.0), seed=0)
nodes = [JointStateNode(world, "robot_state", entity="robot"),
         JointPositionTargetNode(world, "robot_target", entity="robot", control_timestep=4 / 120)]
env = IsaacSimVecEnv(world, nodes)

context, obs, info = env.reset()
# obs = {"robot_state": {"joint_position": ..., "joint_velocity": ...}}
obs, reward, terminated, truncated, info = env.step(targets)  # (64, n) joint targets; 4 physics steps
```

`MySceneCfg` is an ordinary Isaac Lab `InteractiveSceneCfg` with an articulation named `robot`. As everywhere in
UniEnv, observations and actions are dicts keyed by node name, except that a single node's value is passed as-is.

## Install and test

Into an env that has Isaac Lab (it is not installable from PyPI; see Isaac Lab's installation guide):

```bash
pip install unienv-isaacsim
```

From a checkout, with the tests:

```bash
pip install -e '.[test]'
pytest  # needs a GPU; launches Isaac Sim headless
UNIENV_ISAACSIM_NO_SIM=1 pytest  # without Isaac Lab: only the tests that do not need it
```
