# 18.11.24

import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        input_size: Dimensione dello stato osservato.
        output_size: Numero di azioni possibili.
        """
        super(DQN, self).__init__()
        
        # Define layer sizes
        hidden1_size = 128
        hidden2_size = 64
        
        # First layer with batch normalization
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size)
        
        # Second layer with batch normalization
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size)
        
        # Third layer with batch normalization
        self.fc3 = nn.Linear(hidden2_size, hidden2_size)
        self.bn3 = nn.BatchNorm1d(hidden2_size)
        
        # Output layer
        self.output_layer = nn.Linear(hidden2_size, output_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x