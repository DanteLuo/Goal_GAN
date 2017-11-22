from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from ant_env_modified_baseline import AntEnvModBase
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from rllab.envs.mujoco.ant_env import AntEnv
from ant_maze_modified_baseline import AntMazeEnvModBase
from uniform_sampler import UniformSampler
import pickle
from rllab import spaces
from cached_property import cached_property
from rllab.misc.overrides import overrides
from uniform_sampler import UniformSampler
import numpy as np
import time
from rllab.envs.mujoco.hopper_env import HopperEnv


# env = TfEnv(normalize(AntEnvMod()))
# env = TfEnv(normalize(AntMazeEnvMod(maze_size_scaling=5.)))
# env.wrapped_env.wrapped_env.set_goal(goal=[0.5,0.5])
# env.wrapped_env.wrapped_env.turn_l2_rew(True)
# sampler = UniformSampler(env,False)
# print(sampler.sample_goals(2))
# env = HopperEnv(alive_coeff=0.9)
#
# # for i in range(1):
# #     state, reward, done, info = env.step(env.action_space.sample())
# #     print(reward,done)
# #     env.render()
#
# # env = AntEnvMod()
#
# # print([pickle.loads(pickle.dumps(env)).wrapped_env.wrapped_env.goal_set for _ in range(10)])
# envs = [pickle.loads(pickle.dumps(env)) for _ in range(10)]
# for i in range(10):
#     print(envs[i].alive_coeff)
#
# env2 = HopperEnv(alive_coeff=0.8)
# envs2 = [pickle.loads(pickle.dumps(env2)) for _ in range(10)]
#
#
# for i in range(10):
#     print(envs2[i].alive_coeff)

env = AntMazeEnvModBase(seed=55)
print(env.seed)
envs = [pickle.loads(pickle.dumps(env)) for _ in range(1)]
envs[0].reset()
print(envs[0]._goal_range)
# env = AntMazeEnvModBase(seed=66)
# print(env.seed)
# print([pickle.loads(pickle.dumps(env)).goal_set for _ in range(10)])


