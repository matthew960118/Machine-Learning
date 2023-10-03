from My_RL_brain import DeepQNetwork
import gym
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, metrics, Input

RL =DeepQNetwork(4,1,0.001)

x = RL.store_transition(1,2,0,2)
x = RL.store_transition(1,2,3,4)

print(x)