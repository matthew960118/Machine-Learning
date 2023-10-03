import tensorflow as tf , datetime, tensorboard
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, metrics, Input
import numpy as np
import os, random
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
            e_greedy_increment=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.memory = deque(maxlen=self.memory_size)
        self.memory_counter = 0  # Corrected typo
        self.learn_step_counter = 0

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.tensorboard_writer = tf.summary.create_file_writer(logdir=log_dir)

        self.eval_network = self._build_net()
        self.target_network = self._build_net()

        self.winstep = 0
        self.wincounter = 0

    def _build_net(self):
        nn_count = 8
        network = Sequential([
            Input(shape = (self.n_features)),
            layers.Dense(nn_count, activation=tf.nn.relu),
            layers.Dense(self.n_actions)
        ])
        network.compile(optimizer=optimizers.Adam(self.lr),loss = tf.losses.MeanSquaredError())
        return network
    
    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)  # Add an additional dimension to make it (batch_size, n_features)
        if np.random.uniform() < self.epsilon:
            q_values = self.eval_network(state)
            action = tf.argmax(q_values, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.n_actions)
        return action

    def store_transition(self, s, a, r, s_, done):
        self.memory.append((s,a,r,s_,done))

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_network.set_weights(self.eval_network.get_weights())
            print("replaced")

        batch_indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False)
        batch_memory = [self.memory[i] for i in batch_indices]
        batch_memory = np.array(batch_memory)

        reward = batch_memory[:,self.n_features+1]
        state_ = batch_memory[:,-(self.n_features+1)]
        state = batch_memory[:,self.n_features-1]

        q_next = self.target_network.predict(state_)
        q_eval = self.eval_network.predict(state)

        q_target = tf.identity(q_eval)

        update = reward + self.gamma*tf.reduce_max(q_next,axis=1)
        i = tf.expand_dims(tf.argmax(q_next,axis=1,output_type=tf.int32),axis=1)
        b = tf.expand_dims(np.arange(self.batch_size),axis=1)
        i = tf.concat([b,i],axis=1)

        q_target = tf.tensor_scatter_nd_update(q_target, i, update)
        print(q_target.shape)

        self.eval_network.fit(state, q_target,epochs=1,verbose=-1)
        # for state, action, reward, state_, done in batch_memory:
        #     if not done:
        #         target = reward + self.gamma*np.amax(self.target_network.predict([state_])[0])
        #     else:
        #         target = reward
        #     target_f = self.target_network.predict([state])
        #     target_f[0][action]=target
        #     state = np.ndarray([state],dtype=np.int32)

        #     print(target_f.shape, state.shape,state)
        #     self.eval_network.fit(state,target_f,epochs=1)

        # with self.tensorboard_callback.as_default():
        #     tf.summary.scalar('loss', loss, step=self.learn_step_counter)
        # self.tensorboard_callback.flush()

        with self.tensorboard_writer.as_default():
            # tf.summary.scalar('Loss', loss, step=self.learn_step_counter)
            tf.summary.scalar('Win Step', self.winstep, step=self.learn_step_counter)
            tf.summary.scalar('Win Count', self.wincounter, step=self.learn_step_counter)
            tf.summary.scalar('e_greedy', self.epsilon, step=self.learn_step_counter)

        self.learn_step_counter += 1
        self.epsilon =self.epsilon + self.epsilon_increment if self.epsilon<self.epsilon_max else self.epsilon_max

    def save_model(self, model_path):
        # 儲存評估網路的權重
        self.eval_network.save_weights(model_path)

    def load_model(self, model_path):
        # 載入評估網路的權重
        self.eval_network.load_weights(model_path)
        self.target_network.load_weights(model_path)