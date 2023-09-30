import RL_brain
import gym

def update():
    step = 0
    for episode in range(10):
        obs, info = env.reset()
        
        while True:
            env.rander()
            
            action = RL.choose_action(obs)
            
            obs_, reword, done, info = env.step(action)
            
            RL.store_memory(obs, action, reword, obs_)
            
            if step>100 and step%10==0:
                RL.learn()
                
            obs = obs_
            
            if done:
                break
            
            step+=1
            
if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    RL =RL_brain()
    update()