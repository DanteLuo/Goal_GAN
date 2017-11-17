from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from ant_env_modified import AntEnvMod
from rllab.envs.mujoco.maze.ant_maze_env import AntMazeEnv
from rllab.envs.mujoco.ant_env import AntEnv
from ant_maze_modified import AntMazeEnvMod


# env = TfEnv(normalize(AntEnvMod(goal=[1,1])))
env = TfEnv(normalize(AntMazeEnvMod()))
env.wrapped_env.wrapped_env.set_goal(goal=[2,2])

for i in range(100):
    state, reward, done, info = env.step(env.action_space.sample())
    print(reward,done)
    env.render()

