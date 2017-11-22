from rllab.envs.mujoco.maze.maze_env import MazeEnv
from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.base import Step
from rllab import spaces
from cached_property import cached_property
from rllab.misc.overrides import overrides
from rllab.core.serializable import Serializable
from uniform_sampler import UniformSampler
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

    def __init__(self, seed=42, num_goals=10, is_l2=False, *args, **kwargs):
        Serializable.quick_init(self, locals())

        Serializable.quick_init(self, locals())
        self.is_l2 = is_l2
        self.num_goals = num_goals
        self.seed = seed
        np.random.seed(seed)
        self.goal_set = UniformSampler(self,is_maze=True).sample_goals(num_goals)
        super(AntMazeEnvModBase, self).__init__(*args, **kwargs)
        self.reset()

    @cached_property
    @overrides
    def observation_space(self):
        ub = 1e6 * np.ones(37)
        return spaces.Box(ub * -1, ub)

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
        print(goal)
        minx = goal[0] - 0.5 - self._init_torso_x
        maxx = goal[0] + 0.5 - self._init_torso_x
        miny = goal[1] - 0.5 - self._init_torso_y
        maxy = goal[1] + 0.5 - self._init_torso_y

        self._goal_range = minx, maxx, miny, maxy

    def step(self, action):
        full_obs, reward, done, _ = super(AntMazeEnvModBase,self).step(action)
        next_obs, goal = self._get_current_obs(full_obs)
        if self.is_l2:
            reward = -next_obs[-1]
        return Step(next_obs,reward,done,kwarg1=goal)

    def reset(self, init_state=None):
        np.random.seed(None)
        self._set_goal(self.goal_set[np.random.choice(self.num_goals),:])
        full_obs = super(AntMazeEnvModBase,self).reset()
        corp_obs, _ = self._get_current_obs(full_obs)

        return corp_obs


