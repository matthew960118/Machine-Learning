from RL_brain import RL_brain
import gym
import ale_py

def update():
    step = 0
    for episode in range(10):
        obs, info = env.reset()
        
        while True:
            env.render()
            
            obs = RL.data_perprocessing(obs)
            
            action = RL.choose_action(obs)

            obs_, reword, done, info, _ = env.step(action)
            obs_  = RL.data_perprocessing(obs_)
            RL.store_memory(obs, action, reword, obs_,done)
            
            if step>500 and step%10==0:
                print(step)
                RL.learn()
            if step%1000==0:
                RL.save(name=step%1000)
            obs = obs_
            
            if done:
                break
            
            step+=1
            
if __name__ == "__main__":
    env = gym.make('Breakout-v4',render_mode="human")
    
    n_action = 4
    
    RL = RL_brain(
        n_actions=n_action,
        n_features=1,
        learning_rate=0.01,
        )
    update()