from RL_brainNew import DeepQNetwork
import gym
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, metrics, Input


class Eval_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network')
        self.layer1 = layers.Dense(10, activation='relu')
        self.logits = layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


class Target_Model(tf.keras.Model):
    def __init__(self, num_actions):
        super().__init__('mlp_q_network_1')
        self.layer1 = layers.Dense(10, trainable=False, activation='relu')
        self.logits = layers.Dense(num_actions, trainable=False, activation=None)

    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        layer1 = self.layer1(x)
        logits = self.logits(layer1)
        return logits


def update():
    step = 0
    for eqisode in range(300):
      observation, info = env.reset()
    
      while True:
          env.render()
          
          action = RL.choose_action(observation)

          observation_, reward, done, truncated, info = env.step(action)

          RL.store_transition(observation, action, reward, observation_)

          if step > 200 and step%5 == 0:
              RL.learn()

          observation = observation_

          if done:
              break
          step += 1

if __name__ == "__main__":
    n_action = 4
    env = gym.make("FrozenLake-v1",render_mode="human")
    eval_model = Eval_Model(n_action)
    target_model = Target_Model(n_action)
    RL = DeepQNetwork(
        n_actions=n_action,
        n_features=1,
        eval_model=eval_model,
        target_model=target_model
        )
    update()
    RL.plot_cost()