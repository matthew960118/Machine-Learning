from My_RL_brain import DeepQNetwork
import gym
import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, optimizers, Sequential, metrics, Input

def update():
    step = 0
    for eqisode in range(10000):
        observation, info = env.reset()
        winstep = 0
        for i in range(100):

        # while True:
            step += 1
            winstep +=1
            env.render()
            
            action = RL.choose_action(observation)

            observation_, reward, done, t, info = env.step(action)
            if done :
                reward = 1

            if reward ==-100:
                done = True
            RL.store_transition(observation, action, reward, observation_, done)

            if step > 200 and step%5 == 0:
                RL.learn()

            observation = observation_
            if done:
                RL.eqisode = eqisode
                if reward== 1:
                    RL.winstep = winstep
                    RL.wincounter+=1
                    winstep = 0
                # RL.save_model("model.h5")
                break
        


if __name__ == "__main__":
    n_action = 4
    env = gym.make('CliffWalking-v0',render_mode="ansi")
    RL = DeepQNetwork(
        n_actions=n_action,
        n_features=1,
        learning_rate=0.01,
        e_greedy_increment=0.001
        )
    update()
    # RL.load_model("model.h5")
    RL.tensorboard_writer.flush()
    RL.tensorboard_writer.close()