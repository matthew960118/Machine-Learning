import gym
from RL_brain import QLearningTable

def update():
    for episode in range(1000):
        observation, info = env.reset()
        count = 0
        for i in range(100):
            count +=1
            env.render()

            action = RL.choose_action(str(observation))
            modifyReward = 0
            observation_, reward, done, truncated, info = env.step(int(action))
            # if done and reward!=1:
            #     reward = -1
            if not done:
                RL.learn(str(observation), action, 0, str(observation_))
            elif reward ==1:
                RL.learn(str(observation), action, 1, str(observation_))
            else:
                RL.learn(str(observation), action, -1, str(observation_))
            observation = observation_

            if done:
                print("Episode finished after {} timesteps".format(episode+1))
                break
    print('game over')
    RL.getQTable().to_csv("q-table", index=True, mode='w')

if __name__ == "__main__":
    env = gym.make("FrozenLake-v1",render_mode="human",map_name="8x8")
    RL = QLearningTable(actions=[0,1,2,3],qTableCSVPath='./q-table')
    update()
    