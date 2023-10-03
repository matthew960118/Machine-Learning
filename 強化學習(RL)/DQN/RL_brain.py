import tensorflow as tf , datetime, tensorboard
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, metrics, Input
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CustomTensorBoard(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, histogram_freq=0):
        super().__init__()
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.file_writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with self.file_writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=epoch)
            self.file_writer.flush()


class DeepQNetwork:
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

        self.loss = tf.losses.MeanSquaredError()
        self.optinizer = optimizers.Adam(self.lr)

        self.learn_step_counter = 0

        self.memory = np.zeros((self.memory_size,n_features*2+2))

        self._build_net()

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = CustomTensorBoard(log_dir=log_dir, histogram_freq=1)

    def _build_net(self):
        self.eval_Network = keras.Sequential([
            Input(shape = (1)),
            layers.Dense(10,activation=tf.nn.relu),
            layers.Dense(self.n_actions)
        ])

        self.target_Network = keras.Sequential([
            Input(shape = (1)),
            layers.Dense(10,activation=tf.nn.relu),
            layers.Dense(self.n_actions)
        ])
    def store_transition(self,s,a,r,s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s,[a, r], s_),)

        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1
    
    def choose_action(self, observation):
        observation = np.array([observation])
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_Network(observation)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0,self.n_actions)

        return action
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_Network.set_weights(self.eval_Network.get_weights())
            print("replaced")
        
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]
        s_ = batch_memory[:,-self.n_features]
        s = batch_memory[:,self.n_features]

        with tf.GradientTape() as tape:
            q_next = self.target_Network(s_)
            q_eval = self.eval_Network(s)
            q_target = tf.identity(q_eval)

            batch_index = np.arange(self.batch_size,dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:,self.n_features+1]

            indices = tf.stack([batch_index, eval_act_index], axis=1)
            updates = reward + self.gamma * tf.reduce_max(q_next, axis=1)
            q_target = tf.tensor_scatter_nd_update(q_target, indices, updates)

            loss = self.loss(q_target, q_eval)
        
        gradients = tape.gradient(loss,self.eval_Network.trainable_variables)
        self.optinizer.apply_gradients(zip(gradients,self.eval_Network.trainable_variables))

        self.tensorboard_callback.on_epoch_end(self.learn_step_counter, {'loss': loss})
        self.epsilon =self.epsilon + self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max
        self.learn_step_counter +=1