from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.base import Step
import numpy as np


class AntEnvMod(AntEnv):
    '''
    Modified the output of the observation and the done criterion
    '''

    def __init__(self, goal=None, m=0.1, *args, **kwargs):

        if goal is None:
            print('Please specify goal for the agent!')
            return
        self.goal  = goal
        self.__m = m
        super(AntEnvMod, self).__init__(*args, **kwargs)


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


    def set_goal(self, new_goal):
        self.goal = new_goal


    def step(self, action):
        full_obs, _, _, _ = super(AntEnvMod,self).step(action)

        next_obs = full_obs[:29]  # robot position and velocity
        next_obs = np.append(next_obs, full_obs[-3:])  # robot CoM

        # adding extra state
        d_to_goal = self._distance_to_goal()
        ## the vector to goal is two dimensional or three dimensional
        vec_to_goal = self._vector_to_goal()
        next_obs = np.append(next_obs, [self.goal[0],self.goal[1],
                            vec_to_goal[0],vec_to_goal[1],d_to_goal])
        x, y = self.get_body_com("torso")[:2]
        reward = 0
        done = False
        if abs(x)<5*self.__m and abs(y)<5*self.__m:
            reward = 1
            done = True

        return Step(next_obs,float(reward),done)

