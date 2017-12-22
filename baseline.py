from ant_env_modified_baseline import AntEnvModBase
from ant_maze_modified_baseline import AntMazeEnvModBase
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from trpo_rollouts import TRPO_mod
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
import tensorflow as tf
import numpy as np
import pickle


# stub(globals())
# ext.set_seed(1)

env = TfEnv(normalize(AntEnvModBase()))


policy = GaussianMLPPolicy(name='policy',
                           env_spec=env.spec,
                           hidden_nonlinearity=tf.nn.tanh)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO_mod(env=env,
                policy=policy,
                baseline=baseline,
                n_itr=5,
                max_path_length=500,
                discount=0.998,
                gae_lambda=0.995,
                batch_size=6000)

algo.initialize(0.0,1.0)

sess = tf.Session()
sess.__enter__()

for it in range(1):

    records = {}
    sess.run(tf.global_variables_initializer())

    for id in range(300):

        algo.train(sess=sess)

        goals,label,rewards = algo.label_goals()

        records[id] = (goals,label,rewards)

        env = TfEnv(normalize(AntEnvModBase(seed=np.random.randint(0,100000))))

        algo.set_env(env)

    pickle_out = open("dict.records_BL_One_{0}".format(str(it+1)), "wb")
    pickle.dump(records,pickle_out)
    pickle_out.close()

sess.close()

env = TfEnv(normalize(AntEnvModBase(is_l2=True)))

policy2 = GaussianMLPPolicy(name='policy2',
                           env_spec=env.spec,
                           hidden_nonlinearity=tf.nn.tanh)

baseline2 = LinearFeatureBaseline(env_spec=env.spec)

algo2 = TRPO_mod(env=env,
                policy=policy2,
                baseline=baseline2,
                n_itr=5,
                max_path_length=500,
                discount=0.998,
                gae_lambda=0.995,
                batch_size=6000)

algo2.initialize(0.0, 1.0)

sess = tf.Session()
sess.__enter__()

for it in range(1):

    records = {}
    sess.run(tf.global_variables_initializer())

    for id in range(300):
        algo2.train(sess=sess)

        goals, label, rewards = algo2.label_goals()

        records[id] = (goals, label, rewards)

        env = TfEnv(normalize(AntEnvModBase(seed=np.random.randint(0,100000),is_l2=True)))

        algo2.set_env(env)

    pickle_out = open("dict.records_BL_Two_Modified_{0}".format(str(it+3)), "wb")
    pickle.dump(records, pickle_out)
    pickle_out.close()

sess.close()
