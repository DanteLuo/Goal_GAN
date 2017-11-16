from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.base import Step
import numpy as np


class AntEnvModified(AntEnv):
    '''
    Modified the output of the observation and the done criterion
    '''

    def __init__(self, goal, epsilon=0.1, *args, **kwargs):
        self.goal  = goal
        self.__epsilon = epsilon
        super(AntEnvModified, self).__init__(*args, **kwargs)


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
        ob, _, _, _ = super(AntEnvModified,self).step(action)

        # adding extra state
        d_to_goal = self._distance_to_goal()
        ## the vector to goal is two dimensional or three dimensional
        vec_to_goal = self._vector_to_goal()
        ob = np.append(ob, [self.goal[0],self.goal[1],
                            vec_to_goal[0],vec_to_goal[1],d_to_goal])

        reward = 0.
        done = False
        if d_to_goal < self.__epsilon:
            reward = 1.
            done = True

        return Step(ob,reward,done)

