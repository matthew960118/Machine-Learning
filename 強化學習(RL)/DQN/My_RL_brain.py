import tensorflow as tf , datetime, tensorboard
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, metrics, Input
import numpy as np
import os

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
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 3))
        self.memory_counter = 0  # Corrected typo
        self.learn_step_counter = 0

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.tensorboard_writer = tf.summary.create_file_writer(logdir=log_dir)

        self.optimizer = optimizers.SGD(self.lr)

        self._build_net()

        self.winstep = 0
        self.wincounter = 0
        self.eqisode = 0

    def _build_net(self):
        nn_count = 4
        self.eval_network = Sequential([
            Input(shape = (self.n_features)),
            layers.Dense(nn_count, activation=tf.nn.relu),
            layers.Dense(nn_count, activation=tf.nn.relu),
            layers.Dense(self.n_actions)
        ])

        self.target_network = Sequential([
            Input(shape = (self.n_features)),
            layers.Dense(nn_count, activation=tf.nn.relu),
            layers.Dense(nn_count, activation=tf.nn.relu),
            layers.Dense(self.n_actions)
        ])

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)  # Add an additional dimension to make it (batch_size, n_features)
        if np.random.uniform() < self.epsilon:
            q_values = self.eval_network(state)
            action = tf.argmax(q_values, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.n_actions)
        return action

    def store_transition(self, s, a, r, s_,done):
        index = self.memory_counter % self.memory_size
        transition = np.hstack((s, [a, r], s_, done))
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_network.set_weights(self.eval_network.get_weights())
            print("replaced")

        batch_indices = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        batch_memory = self.memory[batch_indices]

        batch_state = batch_memory[:, :self.n_features]
        batch_state_ = batch_memory[:, -self.n_features-1:-1]

        reward = batch_memory[:, self.n_features + 1]
        action = tf.cast(batch_memory[:,self.n_features],dtype=tf.int32)
        done = batch_memory[:,-1]

        with tf.GradientTape() as tape:
            q_eval = self.eval_network(batch_state)
            q_next = self.target_network(batch_state_)

            q_eval = tf.cast(q_eval,tf.float32)

            # gamma_term = tf.broadcast_to(self.gamma * tf.expand_dims(tf.reduce_max(q_next, axis=1), axis=1),(self.batch_size,4))
            # gamma_term = tf.cast(gamma_term, tf.float32)

            # q_target = q_eval + tf.stop_gradient(tf.cast(tf.expand_dims(reward,axis=1), tf.float32) + gamma_term - q_eval)

            q_target = tf.identity(q_eval)
            i = tf.expand_dims(action,axis=1)
            b = tf.expand_dims(np.arange(self.batch_size),axis=1)
            i = tf.concat([b,i],axis=1)
            # tensor_scatter_nd_update
            update = tf.where(done, reward, tf.stop_gradient(reward + self.gamma * tf.reduce_max(q_next, axis=1)))
            
            q_target = tf.tensor_scatter_nd_update(q_target, i, update)

            # if self.learn_step_counter>1000:
            #     print(q_next,tf.argmax(q_next,axis=1,output_type=tf.int32),reward + self.gamma*tf.reduce_max(q_next,axis=1))
            #     exit()
            loss = tf.losses.MeanSquaredError()(q_target, q_eval)
        
        gradients = tape.gradient(loss, self.eval_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_network.trainable_variables))

        # with self.tensorboard_callback.as_default():
        #     tf.summary.scalar('loss', loss, step=self.learn_step_counter)
        # self.tensorboard_callback.flush()

        with self.tensorboard_writer.as_default():
            tf.summary.scalar('Loss', loss, step=self.learn_step_counter)
            tf.summary.scalar('Win Step', self.winstep, step=self.eqisode)
            tf.summary.scalar('Win Count', self.wincounter, step=self.eqisode)
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