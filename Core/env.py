import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

#The following function calculates the reward based on pH and pH Setpoint
def calculate_reward_1(pH3, pH3_set):
    #Calculate the absolute difference between actions
    # action_diff = abs(a_t - a_t_plus_1)
     
    # if action_diff > 0.05:
    #     reward_multiplier = k
    # else:
    #     reward_multiplier = 2 * k
    
    #calculate the reward
    reward =  (0.1 - abs(pH3 - pH3_set)*3)
    
    return reward

#LSTM Soft Sensor model definition
class LSTMSoftSensor(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMSoftSensor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

class PHControlEnv(gym.Env):
    def __init__(self, target_pH=9.0, render_mode=None, inference_mode=False):
        super(PHControlEnv, self).__init__()
        self.target_pH = target_pH
        self.render_mode = render_mode
        self.inference_mode = inference_mode
        
        #Action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), 
                                            high=np.array([14.0, 14.0, 0.211]), 
                                            dtype=np.float32)
        
        #Initial conditions of the model
        self.G = 0.01
        self.F_B = 0.001
        self.pH = -np.log10(self.G + np.sqrt(self.G**2 + 4*1e-14)) + np.log10(2)  # Initial pH value
        
        #Wait action is action 1
        self.wait_action = 2
        self.k = 1e-4  #Controller gain
        self.time_step = 0.1 
        #LSTM model setup
        self.lstm_model = LSTMSoftSensor()
        self.lstm_model.load_state_dict(torch.load('lstm_soft_sensor_weights_with_action.pth'))
        self.lstm_model.eval()
        
        self.action_history = [-np.inf] * 4 
        self.pH_history = [-np.inf] * 4  #Initialize with placeholders
        self.step_counter = 0  #Counter for steps
        self.predicted_pH = None  #To store predicted pH value
        self.V = 11  #Tank volume
        self.F_A = 0.111  #Flow rate for acid (fixed)
        self.C_A = 0.001  #Acid concentration
        self.C_B = 0.001  #Base concentration
        self.K_w = 1e-14  #Water dissociation constant
        self.pH_tolerance = 0.1
        
        self.consecutive_steps_within_target = 0
        
        self.steps_to_maintain = 1000
        self.pH_min = 3.162196286682878
        self.pH_max = 9.532618603153724 
        self.action_min = 1.1920929e-07
        self.action_max = 1

    def normalize(self, value):
        return (value - self.pH_min) / (self.pH_max - self.pH_min)
    
    def normalize_action(self, value):
        return (value - self.action_min) / (self.action_max - self.action_min)

    def denormalize(self, value):
        return value * (self.pH_max - self.pH_min) + self.pH_min
        
    def step(self, action):
        min_F_B = 0.001
        max_F_B = 0.211
        self.F_B = action[0] * (max_F_B - min_F_B) + min_F_B 
        self.update_G(self.F_B)
        
        self.pH = -np.log10(self.G + np.sqrt(self.G**2 + 4*1e-14)) + np.log10(2)
        
        self.pH_history.append(self.pH)
        if len(self.pH_history) > 4:
            self.pH_history.pop(0) 

        self.action_history.append(action[0])
        if len(self.action_history) > 4:
            self.action_history.pop(0)
        
    
        self.step_counter += 1
        
        if self.step_counter == 5:
            normalized_pH_history = [self.normalize(pH) for pH in self.pH_history[-4:]]
            normalized_action_history = [self.normalize_action(action_here) for action_here in self.action_history[-4:]]
            
            lstm_input = torch.tensor(
                list(zip(normalized_pH_history, normalized_action_history)), 
                dtype=torch.float32
            ).view(1, 4, 2)  

            #Predict the next pH value using the LSTM model
            self.predicted_pH = self.lstm_model(lstm_input).item()
            self.predicted_pH = self.denormalize(self.predicted_pH)
            
            self.pH = self.predicted_pH
            self.step_counter = 0
        else:
            self.predicted_pH = None
        print(self.pH,self.F_B,self.G)
        reward = calculate_reward_1(self.pH , self.target_pH)

        if self.pH < 0 or self.pH > 14:
            done = True 
        if abs(self.pH - self.target_pH) < self.pH_tolerance:
            self.consecutive_steps_within_target += 1
        else:
            self.consecutive_steps_within_target = 0 

        done = self.consecutive_steps_within_target >= self.steps_to_maintain
        if self.inference_mode:
            done = False
        self.state = np.array([self.target_pH, self.pH, self.F_B])
        return self.state, reward, done, {}
    
    def seed(self, seed=None):
        np.random.seed(seed)

    def change_setpoint(self, new_target_pH):
        """
        Method to change the target pH setpoint dynamically.
        
        Parameters:
        - new_target_pH (float): The new setpoint to track.
        """
        self.target_pH = new_target_pH
        print(f"Target pH setpoint changed to: {self.target_pH}")

    def update_G(self, F_B): 
        dG_dt = -(self.F_A + F_B) * self.G + self.C_A * self.F_A - self.C_B * F_B
        self.G += dG_dt * self.time_step 

    def reset(self):
        #Reset the environment to the initial state
        self.pH_history = [-np.inf] * 4
        self.action_history = [-np.inf] * 4 
        self.step_counter = 0 
        self.predicted_pH = None 
        self.G = 0
        self.pH = 7.0
        self.F_B = 0.1 
        
        #Return the initial state
        self.state = np.array([self.target_pH, self.pH, self.F_B])
        return self.state

    def render(self, mode='human'):
        if self.render_mode is not None:
            print(f"Target pH: {self.target_pH}, Predicted pH: {self.state[1]}, Flow Rate F_B: {self.F_B}")
