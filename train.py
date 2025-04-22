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

total_timesteps = 20000
setpoints = [5,8]
steps_per_setpoint = 10000  # Setpoint changes every 100 steps


# Load the model
# We can load the model weights here to transfer leanning
# model = DDPG.load("ddpg_ph_control_dynamic_setpoint_v1_quasi_trail3")

# Initialize variables
current_step = 0
setpoint_index = 0

env.reset()
#Start training and change the setpoint every 100 steps
while current_step < total_timesteps:
    env.reset()
    if current_step % steps_per_setpoint == 0:
        #Switch the setpoint
        setpoint_index = (current_step // steps_per_setpoint) % len(setpoints)
        new_setpoint = setpoints[setpoint_index]
          #Update the environment's target pH
        env.env_method('change_setpoint', new_target_pH=new_setpoint)
        print(f"Setpoint changed to {new_setpoint} at step {current_step}")
    
    #Train the model for the next 100 steps
    model.learn(total_timesteps=steps_per_setpoint, reset_num_timesteps=False)
    
    #Increment the step count
    current_step += steps_per_setpoint


#Save the trained model
model.save("ddpg_ph_control_dynamic_setpoint_soft_sensor_v2_quasi_trail5_dynamic_ph_5_8_actions.zip")
