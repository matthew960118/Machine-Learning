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
        self.optimizer = tf.optimizers.SGD(learning_rate)
        
        self.memory = np.ndarray((self.memory_size,))
        self.learn_step_counter = 0
        
    def store_memory(self,s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter%self.memory_size
        transition = np.hstack((s,[a, r], s_),)
        
        self.memory[index, :] = transition

        self.memory_counter += 1
    
    def _build_net_(self):
        self.eval_net = ResNet.resent18(self.n_actions)
        self.target_net = ResNet.resent18(self.n_actions)
    
    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)  # Add an additional dimension to make it (batch_size, n_features)
        if np.random.uniform() < self.epsilon:
            q_values = self.eval_net(state)
            action = tf.argmax(q_values, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.n_actions)
        return action
    
    def learn(self):
        if self.learn_step_counter%self.replace_target_iter:
            w = self.eval_net.get_weights
            self.target_net.set_weights(w)

        batch_memory = np.random.choice(self.memory,self.batch_size,replace=False)
        
        batch_state_ = batch_memory[:,-self.n_features-1:-1]
        batch_state = batch_memory[:,:self.n_features]
        reward = batch_memory[:,self.n_features+1]
        action = tf.cast(batch_memory[:,self.n_features],dtype=tf.int32)
        done = batch_memory[:,-1]
        
        with tf.GradientTape() as tape:
            q_eval = self.eval_net(batch_state)
            q_next = self.target_net(batch_state_)
            
            q_target = tf.identity(q_eval)
            i = tf.expand_dims(action,axis=1)
            b = tf.expand_dims(np.arange(self.batch_size),axis=1)
            i = tf.concat([b,i],axis=1)
            # tensor_scatter_nd_update
            update = tf.where(done, reward, tf.stop_gradient(reward + self.gamma * tf.reduce_max(q_next, axis=1)))
            
            q_target = tf.tensor_scatter_nd_update(q_target, i, update)
            
            loss = tf.losses.MSE(q_eval,q_target)
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        
        self.learn_step_counter += 1
        self.epsilon =self.epsilon + self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max