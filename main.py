from ant_env_modified import AntEnvMod
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from uniform_sampler import UniformSampler
from rllab.misc import ext
import tensorflow as tf


# stub(globals())
# ext.set_seed(1)

env = TfEnv(normalize(AntEnvMod()))
sampler = UniformSampler(env=env,is_maze=False)
goals = sampler.sample_goals(1)
print(goals)
env.wrapped_env.wrapped_env.set_goal(goals)

policy = GaussianMLPPolicy(name='policy',
                           env_spec=env.spec)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(env=env,
            policy=policy,
            baseline=baseline,
            n_itr=2,
            max_path_length=5,
            discount=0.998,
            gae_lambda=0.995)



sess = tf.Session()
sess.__enter__()

for id in range(1000):
    # goals = sampler.sample_goals(1)
    # print(goals)
    # env.wrapped_env.wrapped_env.set_goal(goals)
    # print(env.wrapped_env.wrapped_env.goal)

    algo.train(sess=sess)


sess.close()

