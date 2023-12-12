import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import numpy as np
import ResNet
import conv

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
        
        # self.memory = np.ndarray((self.memory_size,))
        
        self.memory_buffer = {
            'states': np.zeros((memory_size, 210,160,3), dtype=np.uint8),
            'rewards': np.zeros((memory_size, 1), dtype=np.float32),
            'actions': np.zeros((memory_size, 1), dtype=np.int32),
            'next_states': np.zeros((memory_size,210,160,3), dtype=np.uint8),
            'done':np.ndarray((memory_size),dtype=np.bool_)
        }
        self.learn_step_counter = 0
        
        self._build_net_()
    
    def data_perprocessing(self,s):
        return tf.image.resize(s,[55,37])
    
    
    def store_memory(self,s,a,r,s_,done):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter%self.memory_size
        self.memory_buffer['states'][index]=s
        self.memory_buffer['rewards'][index]=r
        self.memory_buffer['actions'][index]=a
        self.memory_buffer['next_states'][index]=s_
        self.memory_buffer['done'][index]=done
        
        self.memory_counter += 1
    def get_memories(self, indices):
        s = np.ndarray((len(indices),210,160,3),dtype=np.uint8)
        a = np.ndarray(len(indices),dtype=np.float32)
        r = np.ndarray(len(indices),dtype=np.int32)
        s_ = np.ndarray((len(indices),210,160,3),dtype=np.uint8)
        done = np.ndarray(len(indices),dtype=np.bool_)
        
        for i,index in enumerate(indices):
            if index < 0 or index >= min(self.memory_counter, self.memory_size):
                raise ValueError("Invalid memory index")
            s[i] = np.array(self.memory_buffer['states'][index])/255
            a[i] = self.memory_buffer['actions'][index]
            r[i] = self.memory_buffer['rewards'][index]
            s_[i] = np.array(self.memory_buffer['next_states'][index])/255
            done[i] = self.memory_buffer['done'][index]
        return s,r,a,s_,done
    
    def _build_net_(self):
        # self.eval_net = ResNet.resent18(self.n_actions)
        # self.target_net = ResNet.resent18(self.n_actions)
        
        self.eval_net = conv.bulid_net((self.batch_size,210,180,3),nactions=self.n_actions)
        self.target_net = conv.bulid_net((self.batch_size,210,180,3),nactions=self.n_actions)
    
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
            w = self.eval_net.get_weights()
            self.target_net.set_weights(w)

        batch_index = np.random.choice(self.memory_size,self.batch_size,replace=False)
        
        batch_state,reward,action,batch_state_,done = self.get_memories(batch_index)
        
        
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