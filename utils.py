import numpy as np


class UniformSampler():
    def __init__(self, env, is_maze=True):
        '''
        :param env: env to sampling
        :param is_maze: is_maze
        '''
        self.goal_space = {}
        if is_maze:
            self._get_goal_set(env.MAZE_STRUCTURE)
            self.scaling = env.MAZE_SIZE_SCALING
            self.env_size = np.array(env.MAZE_STRUCTURE).shape
        else: # free space case
            self.goal_space[(0,0)] = 'g'
            self.scaling = env.scaling
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
            x = int(((goal_pre[0]+delta_x)/self.env_size[0]-0.5)*self.scaling)
            y = int(((goal_pre[1]+delta_y)/self.env_size[1]-0.5)*self.scaling)
            goals.append([x, y])

        return np.asarray(goals)


class ReplayBuffer():
    def __init__(self, epsilon=0.3):
        '''
        :param epsilon: critic param for adding a new goal
        '''
        self.goals = np.array([[0.,0.]]) # nx2 matrix
        self.rewards = np.array([0.])
        self.label = np.array([1])
        self.epsilon = epsilon

    def add_goal(self, goals, label):
        '''
        Adding goals to replay buffer from goals list that are epsilon apart from the current goal.

        :param goals: the list of goals to be added
        :return: N/A
        '''
        goals = np.array(goals)
        label = np.array(label)
        for id,goal in enumerate(goals):
            goal = np.reshape(goal,[1,2])
            if self.goals.size == 0:
                self.goals = np.append(self.goals, goal,axis=0)
                self.label = np.append(self.label, label[id])
                continue
            min_dist, min_dist_id = self.min_distance(goal)
            if min_dist > self.epsilon:
                self.goals = np.append(self.goals,goal,axis=0)
                self.label = np.append(self.label, label[id])
            elif self.label[min_dist_id] != label[id]:
                self.label[min_dist_id] = label[id]

    def sample_goals(self,num_goals):
        '''
        Sample # of goals from replay buffer uniformly

        :param num_goals: number of goals needed
        :return: a list of goals
        '''
        goals_ids = np.random.randint(0,len(self.goals),num_goals)
        sample_goals = self.goals[goals_ids,:]
        sample_labels = self.label[goals_ids]
        return sample_goals, np.reshape(sample_labels,[len(sample_labels),1])

    def min_distance(self, goal):
        '''
        Calculate the minimum distance of the goal to the replay buffer.
        :param goal: the goal to be evaluated
        :return: the minimum distance
        '''
        delta_dist = self.goals-goal
        distance = min(np.sqrt((delta_dist[:,0])**2+(delta_dist[:,1])**2))
        id_num = np.argmin(np.sqrt((delta_dist[:,0])**2+(delta_dist[:,1])**2))
        return distance, id_num
