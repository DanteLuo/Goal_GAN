from ant_maze_modified import AntMazeEnvMod
import numpy as np


class UniformSampler():
    def __init__(self, env, is_maze=True):
        '''
        :param env: env to sampling
        :param is_maze: is_maze
        '''
        self.goal_space = {}
        if is_maze:
            self._get_goal_set(env.wrapped_env.wrapped_env.MAZE_STRUCTURE)
            self.scaling = env.wrapped_env.wrapped_env.MAZE_SIZE_SCALING
            self.env_size = np.array(env.wrapped_env.wrapped_env.MAZE_STRUCTURE).shape
        else: # free space case
            self.goal_space[(0,0)] = 'g'
            self.scaling = env.wrapped_env.wrapped_env.scaling
            self.env_size = [1,1]

    def _get_goal_set(self, structure):
        '''
        Get all the possible goals with in the structure.
        :param structure: the structure of the env
        :return:
        '''
        for i in range(len(structure)):
            for j in range(len(structure[0])):
                if str(structure[i][j]) != '1':
                    self.goal_space[(i,j)] = 'g'

    def sample_goals(self, num_goals):
        '''
        :param num_goals: number of goals
        :return:
        '''
        goals = list()
        goal_ids = np.random.randint(0, len(list(self.goal_space.keys())), num_goals)
        for goal_id in goal_ids:
            goal_pre = list(self.goal_space.keys())[goal_id]
            delta_x = np.random.random_sample()
            delta_y = np.random.random_sample()
            x = (goal_pre[0]+delta_x)/self.env_size[0]*self.scaling
            y = (goal_pre[1]+delta_y)/self.env_size[1]*self.scaling
            goals.append([x, y])

        return np.asarray(goals)
