# 18.11.24

import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        """
        input_size: Dimensione dello stato osservato (es: posizione, direzione, distanze).
        output_size: Numero di azioni possibili (4: su, giù, sinistra, destra).
        """
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.network(x)
