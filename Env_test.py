from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from ant_env_modified import AntEnvModified


env = TfEnv(normalize(AntEnvModified()))

for i in range(1000):
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()

