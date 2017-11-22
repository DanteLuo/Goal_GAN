from ant_env_modified_baseline import AntEnvMod
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.instrument import run_experiment_lite, stub
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from uniform_sampler import UniformSampler
from rllab.misc import ext
import tensorflow as tf
from rllab.envs.mujoco.ant_env import AntEnv
import pickle


# stub(globals())
# ext.set_seed(1)

env = AntEnv()
env.action_noise = 1.
# sampler = UniformSampler(env=env.wrapped_env.wrapped_env,is_maze=False)

print(pickle.loads(pickle.dumps(env)).action_noise)
# goals = sampler.sample_goals(1)
# print(goals[0])
# env.wrapped_env.wrapped_env.set_goal(goals[0])

# policy = GaussianMLPPolicy(name='policy',
#                            env_spec=env.spec,
#                            hidden_nonlinearity=tf.nn.tanh)
#
# baseline = LinearFeatureBaseline(env_spec=env.spec)
#
# algo = TRPO(env=env,
#             policy=policy,
#             baseline=baseline,
#             n_itr=2,
#             max_path_length=5,
#             discount=0.998,
#             gae_lambda=0.995,
#             force_batch_sampler=True)
#
#
# sess = tf.Session()
# sess.__enter__()
#
# for id in range(1000):
#     # goals = sampler.sample_goals(1)
#     # print(goals)
#     # env.wrapped_env.wrapped_env.set_goal(goals)
#     # print(env.wrapped_env.wrapped_env.goal)
#
#     algo.train(sess=sess)
#
#
# sess.close()

