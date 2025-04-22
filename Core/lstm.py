import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#LSTM model
class LSTMSoftSensor(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMSoftSensor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

data = pd.read_csv('train3.csv')

pH_data = data['PH_VAL'].values
action_data = data['ACTION'].values

pH_min, pH_max = np.min(pH_data), np.max(pH_data)
action_min, action_max = np.min(action_data), np.max(action_data)
print(pH_min)
print(pH_max)
