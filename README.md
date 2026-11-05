# unienv_isaaclab

Isaac Lab adaptor for [UniEnv](https://github.com/UniEnvOrg/UniEnv)'s node system, the counterpart of
`genesis_adaptor` for Genesis.

- `IsaacLabWorld` (`unienv_isaaclab/world`): owns Isaac Lab's `SimulationContext` and `InteractiveScene` and steps
  them one physics step at a time. It is built once in its constructor (an Isaac stage cannot be rebuilt), so
  `reload` is not supported.
- `nodes/`: thin generic nodes over the scene's entities (`ArticulationNode`, `RigidObjectNode`,
  `ContactSensorNode`).
- `IsaacLabVecEnv` (`unienv_isaaclab/vec_env.py`): a `WorldEnv` with Isaac Lab's batched-env behaviour: `step`
  resets the envs that finished and returns their first observation of the new episode (rewards and dones are
  computed before that reset), the interface rsl_rl, rl_games and FlashSAC loops expect.

Priorities follow UniEnv (higher runs first): scene geometry 80-100, robots and objects 50-80, tasks 0-50,
sensors below 0.

Install into an env that has Isaac Lab and UniEnv: `pip install --no-deps -e .`
Smoke test (needs a GPU): `python scripts/smoke.py --headless`
