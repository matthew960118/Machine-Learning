import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_grqqdy=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_grqqdy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            stat_actions = self.q_table[observation,:]
            
            action_index = stat_actions.argmax()
            action = stat_actions.index(action_index)
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table[s,a]
        if s_ != "terminal":
            q_traget = r * self.gamma * self.q_table[s_,:].max()
        else:
            q_traget = r
        self.q_table[s, a] += self.learning_rate * (q_traget-q_predict)
        print(self)
        
    def test(self, state):

    def check_state_exist(self, state):
        if state not in self.q_table.index:
        	self.q_table = pd.concat(
            	[
                	self.q_table,
                	pd.Series([0] * len(self.actions), index=self.q_table.columns, name=str(state)).to_frame()
            	],ignore_index=True
        )
