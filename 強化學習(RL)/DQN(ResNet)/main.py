from RL_brain import RL_brain
import gym
import ale_py

def update():
    step = 0
    for episode in range(10):
        obs, info = env.reset()
        
        while True:
            env.render()
            print(obs)
            action = RL.choose_action(obs)
            
            obs_, reword, done, info, _ = env.step(action)
            
            RL.store_memory(obs, action, reword, obs_)
            
            if step>100 and step%10==0:
                RL.learn()
                
            obs = obs_
            
            if done:
                break
            
            step+=1
            
if __name__ == "__main__":
    env = gym.make('Breakout-v4',render_mode='human')
    
    n_action = 4
    
    RL = RL_brain(
        n_actions=n_action,
        n_features=1,
        learning_rate=0.01,
        )
    update()