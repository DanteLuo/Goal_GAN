from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.base import Step
from rllab import spaces
from cached_property import cached_property
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from utils import UniformSampler
import numpy as np


class AntMazeEnvModBase(MazeEnv, Serializable):
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

    def __init__(self, goals=None, is_baseline=True, num_goals=12, is_l2=False, *args, **kwargs):
        Serializable.quick_init(self, locals())

        Serializable.quick_init(self, locals())
        self.is_l2 = is_l2
        self.num_goals = num_goals
        if is_baseline:
            self.goal_set = UniformSampler(self,is_maze=True).sample_goals(num_goals)
        else:
            self.goal_set = goals
        self.reset()
        super(AntMazeEnvModBase, self).__init__(*args, **kwargs)

    @cached_property
    @overrides
    def observation_space(self):
        ub = 1e6 * np.ones(37)
        return spaces.Box(ub * -1, ub)

    def set_goal_id(self,goal_id):
        self.goal = self.goal_set[goal_id]

    def _get_current_obs(self, full_obs):
        next_obs = full_obs[:29]  # robot position and velocity
        next_obs = np.append(next_obs, full_obs[-3:])  # robot CoM

        minx, maxx, miny, maxy = self._goal_range
        goal_x = (minx + maxx) / 2.
        goal_y = (miny + maxy) / 2.
        x, y = self.wrapped_env.get_body_com("torso")[:2]
        # adding extra state
        d_to_goal = np.sqrt((goal_x - x) ** 2 + (goal_y - y) ** 2)
        vec_to_goal = [goal_x - x, goal_y - y]
        next_obs = np.append(next_obs, [goal_x, goal_y,
                                        vec_to_goal[0], vec_to_goal[1], d_to_goal])

        return next_obs, [goal_x,goal_y]

    def _set_goal(self, goal):
        '''
        :param goal: goal in absolute maze axis without size_scaling fall in [0, 1]
        :return:
        '''
        minx = goal[0] - 0.5 - self._init_torso_x
        maxx = goal[0] + 0.5 - self._init_torso_x
        miny = goal[1] - 0.5 - self._init_torso_y
        maxy = goal[1] + 0.5 - self._init_torso_y

        self._goal_range = minx, maxx, miny, maxy

    def step(self, action):
        full_obs, reward, done, info = super(AntMazeEnvModBase,self).step(action)
        next_obs, goal = self._get_current_obs(full_obs)
        if self.is_l2:
            reward = -next_obs[-1]

        info['goal'] = self.goal
        return Step(next_obs,float(reward),done,**info)

    def reset(self, init_state=None):
        full_obs = super(AntMazeEnvModBase,self).reset()
        corp_obs, _ = self._get_current_obs(full_obs)
        self.goal = self.goal_set[np.random.choice(self.num_goals), :]
        return corp_obs


