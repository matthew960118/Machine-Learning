import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import numpy as np
import ResNet

class RL_brain():
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=128,
            e_gerrdy_increment=None,
        ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_gerrdy_increment
        self.epsilon = 0 if e_gerrdy_increment is not None else self.epsilon_max
        
        self.memory = np.ndarray((self.memory_size,))
        
    def store_memory(self,s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter%self.memory_size
        transition = np.hstack((s,[a, r], s_),)
        
        self.memory[index, :] = transition

        self.memory_counter += 1
    def _build_net_(self):
        
        eval_net = ResNet.resent18(self.n_actions)