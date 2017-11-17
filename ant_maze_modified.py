from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.base import Step
import numpy as np


class AntMazeEnvMod(MazeEnv):
    MODEL_CLASS = AntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 2
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

    def set_goal(self,goal=[1,1]):
        size_scaling = self.MAZE_SIZE_SCALING
        minx = goal[0] * size_scaling - size_scaling * 0.5 - self._init_torso_x
        maxx = goal[0] * size_scaling + size_scaling * 0.5 - self._init_torso_x
        miny = goal[1] * size_scaling - size_scaling * 0.5 - self._init_torso_y
        maxy = goal[1] * size_scaling + size_scaling * 0.5 - self._init_torso_y

        self._goal_range = minx, maxx, miny, maxy


    def step(self,action):
        full_obs, reward, done, _ = super(AntMazeEnvMod,self).step(action)
        next_obs = full_obs[:29] # robot position and velocity
        next_obs = np.append(next_obs,full_obs[-3:]) # robot CoM

        minx,maxx,miny,maxy = self._goal_range
        goal_x = (minx+maxx)/2.
        goal_y = (miny+maxy)/2.
        x, y = self.wrapped_env.get_body_com("torso")[:2]
        # adding extra state
        d_to_goal = np.sqrt((goal_x-x)**2+(goal_y-y)**2)
        ## the vector to goal is two dimensional or three dimensional
        vec_to_goal = [goal_x-x,goal_y-y]
        next_obs = np.append(next_obs, [goal_x, goal_y,
                            vec_to_goal[0], vec_to_goal[1], d_to_goal])
        return Step(next_obs,reward,done)


