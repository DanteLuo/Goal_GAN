from sandbox.rocky.tf.algos.trpo import TRPO
from rllab.misc.overrides import overrides
import numpy as np


class TRPO_mod(TRPO):
    '''
    Trust Region Policy Optimization with Rollout recorded
    '''
    @overrides
    def process_samples(self, itr, paths):
        for path in paths:
            x = path["observations"][0][32]
            y = path["observations"][0][33]
            if (x, y) in self.goal_info:
                average_reward, n = self.goal_info[(x, y)]
            else:
                average_reward, n = 0, 0
            new_reward = sum(path["env_infos"]["info"])
            self.goal_info[(x,y)] = (average_reward + 1 / (n + 1) * (new_reward - average_reward),
                                      n + 1)

            if self.take_points and len(self.init_goal_set) <= self.init_goal_size:
                for _ in range(20):
                    id = np.random.randint(0, len(path['observations']))
                    goal = np.reshape(path['observations'][id][:2],[1,2])
                    if self.min_dist(goal) >= 0.3:
                        self.init_goal_set = np.append(self.init_goal_set, goal,axis=0)

        return self.sampler.process_samples(itr, paths)

    def train(self, sess=None, take_points=False):
        self.take_points = take_points
        super(TRPO_mod, self).train(sess)


    def initialize(self, Rmin, Rmax, init_goal_size=12):
        self.Rmin = Rmin
        self.Rmax = Rmax
        self.goal_info = {}
        self.init_goal_set = np.array([[0.,0.]])
        self.init_goal_size = init_goal_size

    def label_goals(self):
        label = list()
        goals = list()
        rewards = list()
        for goal in self.goal_info.keys():
            goals.append(goal)
            rewards.append(self.goal_info[goal][0])
            if self.goal_info[goal][0] >= self.Rmin and self.goal_info[goal][0] <= self.Rmax:
                label.append(1)
            else:
                label.append(0)

        # clear the goal records each round
        self.goal_info = {}
        return np.asarray(goals),np.asarray(label),np.asarray(rewards)

    def get_visited_place(self, num_places):
        return self.init_goal_set[1:num_places+1,:]

    def min_dist(self, goal):
        delta_dist = self.init_goal_set - goal
        distance = min(np.sqrt((delta_dist[:, 0]) ** 2 + (delta_dist[:, 1]) ** 2))
        return distance

    def set_env(self, env):
        self.env = env

