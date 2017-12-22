from ant_env_modified_baseline import AntEnvModBase
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
from trpo_rollouts import TRPO_mod
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from ls_gan_model import LSGAN
from utils import ReplayBuffer
import tensorflow as tf
import numpy as np
import pickle


# Train initial policy

env = TfEnv(normalize(AntEnvModBase(seed=100099)))


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

# set Rmin and Rmax
Rmin = 0.2
Rmax = 0.5
algo.initialize(Rmin,Rmax)

records = {}

replay_buffer = ReplayBuffer()

goal_GAN = LSGAN()

sess = tf.Session()
sess.__enter__()
goal_GAN.build_model(sess=sess)
sess.run(tf.global_variables_initializer())

for it in range(30):

    algo.train(sess=sess,take_points=True)

    env = TfEnv(normalize(AntEnvModBase(seed=np.random.randint(0,100000))))

    algo.set_env(env)

replay_buffer.add_goal(algo.init_goal_set,np.ones(len(algo.init_goal_set)))

# init GAN
goals, labels = replay_buffer.sample_goals(len(algo.init_goal_set))
goal_GAN.train(goals,labels,int(len(algo.init_goal_set)))
# this is clean the goal_info
algo.label_goals()
replay_buffer_records = {}

# Goal GAN training
for it in range(300):

    goals = goal_GAN.sample_goals(8)
    goals_rb,_ = replay_buffer.sample_goals(4)
    goals = np.append(goals,goals_rb,axis=0)
    env = TfEnv(normalize(AntEnvModBase(goals=goals,is_baseline=False)))
    algo.set_env(env)
    algo.train(sess=sess)
    goals_new, labels_new, rewards_new = algo.label_goals()
    labels_new = np.reshape(labels_new,[len(labels_new),1])
    goal_GAN.train(goals_new,labels_new,len(goals_new))
    replay_buffer.add_goal(goals_new,labels_new)

    records[it] = (goals_new, labels_new, rewards_new)
    replay_buffer_records[it] = (replay_buffer.goals,replay_buffer.label)

pickle_out = open('dict.Goal_GAN_fixed_2_5_dot3-1','wb')
pickle.dump(records,pickle_out)
pickle_out.close()
pickle_out = open('dict.Goal_GAN_RB_2_5_dot3-1','wb')
pickle.dump(replay_buffer_records,pickle_out)
pickle_out.close()

sess.close()
