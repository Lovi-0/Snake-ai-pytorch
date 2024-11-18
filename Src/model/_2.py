# 18.11.24

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        input_size: Dimensione dello stato osservato.
        output_size: Numero di azioni possibili.
        """
        super(DQN, self).__init__()
        
        # Primo blocco denso con LayerNorm
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.LayerNorm(256)  # Usa LayerNorm per supportare batch_size = 1
        
        # Secondo blocco
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)
        
        # Terzo blocco
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.LayerNorm(64)
        
        # Output per l'azione
        self.output_layer = nn.Linear(64, output_size)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.output_layer(x)
