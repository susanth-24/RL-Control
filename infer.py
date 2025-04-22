import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from Core.env import PHControlEnv
#Create the environment
env = PHControlEnv(target_pH=8)  #Initial setpoint
env = make_vec_env(lambda: env, n_envs=1)
from stable_baselines3 import DDPG
#Initialize the agent
model = DDPG("MlpPolicy", env, verbose=1)
# Load and test the trained model
env = PHControlEnv(target_pH=5, inference_mode=True)
env = make_vec_env(lambda: env, n_envs=1)

model = DDPG.load("ddpg_ph_control_dynamic_setpoint_soft_sensor_v2_quasi_trail5_dynamic_ph_5_8_actions.zip")
env.env_method('change_setpoint', new_target_pH=5) 

obs = env.reset()
#env.target_pH= 2
actionss = []
statess = []
spp=[]
setpoints = [5,8]
steps_per_setpoint = 100
current_step = 0

for _ in range(200):
    # Change setpoint every 100 steps
    if current_step % steps_per_setpoint == 0:
        setpoint_index = (current_step // steps_per_setpoint) % len(setpoints)
        new_setpoint = setpoints[setpoint_index]
        env.env_method('change_setpoint', new_target_pH=new_setpoint)
        print(f"Setpoint changed to {new_setpoint} at step {current_step}")
    
    # Predict action using the model
    action, _states = model.predict(obs)
    
    # Step in the environment with the action
    obs, reward, done, info = env.step(action)
    spp.append(new_setpoint)
    # Track the pH level and actions for plotting
    statess.append(obs[0][1])  # Append the pH level
    actionss.append(action[0])  # Append the action
    
    # Print the observation for debugging purposes
    print(obs)
    
    current_step += 1

np.save('dynamic_soft_sensor_5_8_states.npy', statess)
np.save('dynamic_soft_sensor_5_8_setpoint.npy', spp)


# Plot the results
plt.plot(statess, label='pH Level')
plt.plot(spp,label='SetPoint')
plt.xlabel('Time (steps)')
plt.ylabel('pH Level')
plt.title('pH Level Tracking with Dynamic Setpoints')
plt.legend()
plt.show()

