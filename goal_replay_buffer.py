import numpy as np


class ReplayBuffer():
    def __init__(self, epsilon=0.5):
        '''
        :param epsilon: critic param for adding a new goal
        '''
        self.goals = np.array([[0.,0.]]) # nx2 matrix
        self.epsilon = 0.5

    def add_goal(self, goals):
        '''
        Adding goals to replay buffer from goals list that are epsilon apart from the current goal.

        :param goals: the list of goals to be added
        :return: N/A
        '''
        goals = np.array(goals)
        for goal in goals:
            goal = np.reshape(goal,[1,2])
            if self.goals.size == 0:
                self.goals = np.append(self.goals, goal,axis=0)
                continue
            min_dist = self.min_distance(goal)
            if min_dist > self.epsilon:
                self.goals = np.append(self.goals,goal,axis=0)

    def sample_goals(self,num_goals):
        '''
        Sample # of goals from replay buffer uniformly

        :param num_goals: number of goals needed
        :return: a list of goals
        '''
        goals_ids = np.random.randint(0,len(self.goals),num_goals)
        sample_goals = self.goals[goals_ids,:]
        return sample_goals

    def min_distance(self, goal):
        '''
        Calculate the minimum distance of the goal to the replay buffer.
        :param goal: the goal to be evaluated
        :return: the minimum distance
        '''
        delta_dist = self.goals-goal
        distance = min(np.sqrt((delta_dist[:,0])**2+(delta_dist[:,1])**2))
        return distance
