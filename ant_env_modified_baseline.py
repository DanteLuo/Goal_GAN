from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.base import Step
from rllab import spaces
from cached_property import cached_property
from rllab.misc.overrides import overrides
from utils import UniformSampler
from rllab.core.serializable import Serializable
import numpy as np


class AntEnvModBase(AntEnv, Serializable):
    '''
    Modified the output of the observation and the done criterion
    '''

    def __init__(self, seed=42, goals=None, is_baseline=True, goal_space=[1,1], scaling=10, m=0.5, is_l2=False, num_goals=12, *args, **kwargs):
        self.__m = m
        self.goal_space = goal_space
        self.scaling = scaling
        self.seed = seed
        self.is_l2 = is_l2
        self.num_goals = num_goals
        self.goal = [1,0]
        np.random.seed(seed)
        if is_baseline:
            self.goal_set = UniformSampler(self,is_maze=False).sample_goals(num_goals)
        else:
            self.goal_set = goals
        self.reset()
        super(AntEnvModBase, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

    def _distance_to_goal(self):
        '''
        Calculate the Euclidean distance from the current position to goal

        :return: distance to goal
        '''
        comvel = self.get_body_comvel("torso")
        d_to_goal = np.sqrt((comvel[0]-self.goal[0])**2+(comvel[1]-self.goal[1])**2)
        return d_to_goal


    def _vector_to_goal(self):
        '''
        Calculate the vector from CoM to goal right now is in two dimensional

        :return: vector from CoM to goal ndarray (2,)
        '''
        comvel = self.get_body_comvel("torso")
        return np.array([(self.goal[0]-comvel[0]),(self.goal[1]-comvel[1])])


    def set_goal_id(self,goal_id):
        self.goal = self.goal_set[goal_id]

    @cached_property
    @overrides
    def observation_space(self):
        ub = 1e6 * np.ones(37)
        return spaces.Box(ub * -1, ub)


    def _get_current_obs(self,full_obs):
        next_obs = full_obs[:29]  # robot position and velocity
        next_obs = np.append(next_obs, full_obs[-3:])  # robot CoM

        # adding extra state
        d_to_goal = self._distance_to_goal()
        vec_to_goal = self._vector_to_goal()
        next_obs = np.append(next_obs, [self.goal[0], self.goal[1],
                                        vec_to_goal[0], vec_to_goal[1], d_to_goal])
        return next_obs


    def step(self, action):
        full_obs, _, _, _ = super(AntEnvModBase,self).step(action)

        next_obs = self._get_current_obs(full_obs)
        x, y = self.get_body_com("torso")[:2]
        reward = 0
        reward_l2 = 0
        done = False
        if abs(x-self.goal[0])<self.__m and abs(y-self.goal[1])<self.__m:
            reward = 1
            reward_l2 = 1
            done = True
        if self.is_l2:
            reward = -self._distance_to_goal()

        return Step(next_obs,float(reward),done,info=reward_l2)


    def reset(self, init_state=None):
        full_obs = super(AntEnvModBase,self).reset()
        corp_obs = self._get_current_obs(full_obs)
        self.goal = self.goal_set[np.random.choice(self.num_goals), :]
        return corp_obs
