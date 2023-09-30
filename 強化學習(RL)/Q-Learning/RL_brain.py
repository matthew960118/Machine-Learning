import numpy as np
import pandas as pd


class QLearningTable: 
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_grqqdy=0.9, qTableCSVPath=None):
        self.actions = actions
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_grqqdy
        if qTableCSVPath == None:
          self.q_table = pd.DataFrame(columns=self.actions, dtype=float)
        else:
          self.q_table = pd.read_csv(qTableCSVPath)
          self.q_table.index = self.q_table.index.astype(str)
          self.q_table.columns =self.q_table.columns.astype(int)
          print(self.q_table)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:
            state_actions = self.q_table.loc[str(observation), :]
            action = np.random.choice(state_actions[state_actions == np.max(state_actions)].index)
        else:
            action = np.random.choice(self.actions)
        return action


    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_==s:
            ...
        else:
            if s_ != "terminal":
                q_target = r + self.gamma * self.q_table.loc[s_, :].max()
            else:
                q_target = r
            self.q_table.loc[s, a] = (1-self.learning_rate)*self.q_table.loc[s, a] + self.learning_rate * q_target
            #print(self.q_table)
 
    def getQTable(self):
      return self.q_table
    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
        	self.q_table = pd.concat(
            	[
                	self.q_table,
                	pd.Series([0] * len(self.actions),
                            index=self.q_table.columns, 
                            name=str(state)).to_frame().T
            	]
        )
    

