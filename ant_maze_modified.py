from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.base import Step
import numpy as np


class AntMazeEnvMod(MazeEnv):
    MODEL_CLASS = AntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 4
    MAZE_SIZE_SCALING = 5.
    MAZE_MAKE_CONTACTS = False
    MAZE_STRUCTURE = [
        [1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]

    MANUAL_COLLISION = False

    def __init__(self, *args, **kwargs):
        super(AntMazeEnvMod,self).__init__(*args, **kwargs)
        self.is_l2 = False

    def turn_l2_rew(self, is_l2=True):
        self.is_l2 = is_l2

    def set_goal(self, goal):
        '''
        :param goal: goal in absolute maze axis without size_scaling fall in [0, 1]
        :return:
        '''
        minx = goal[0] - 0.5
        maxx = goal[0] + 0.5
        miny = goal[1] - 0.5
        maxy = goal[1] + 0.5

        self._goal_range = minx, maxx, miny, maxy

    def step(self, action):
        full_obs, reward, done, _ = super(AntMazeEnvMod,self).step(action)
        next_obs = full_obs[:29] # robot position and velocity
        next_obs = np.append(next_obs,full_obs[-3:]) # robot CoM

        minx,maxx,miny,maxy = self._goal_range
        goal_x = (minx+maxx)/2.
        goal_y = (miny+maxy)/2.
        x, y = self.wrapped_env.get_body_com("torso")[:2]
        # adding extra state
        d_to_goal = np.sqrt((goal_x-x)**2+(goal_y-y)**2)
        vec_to_goal = [goal_x-x,goal_y-y]
        next_obs = np.append(next_obs, [goal_x, goal_y,
                            vec_to_goal[0], vec_to_goal[1], d_to_goal])
        if self.is_l2:
            reward = -d_to_goal
        return Step(next_obs,reward,done)


