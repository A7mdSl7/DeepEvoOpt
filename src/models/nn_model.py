import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hyperparams):
        super(MLP, self).__init__()
        
        # Hyperparameters
        self.hidden1 = int(hyperparams.get('hidden1', 128))
        self.hidden2 = int(hyperparams.get('hidden2', 64))
        self.dropout_rate = float(hyperparams.get('dropout', 0.5))
        
        # Layers
        self.fc1 = nn.Linear(28 * 28, self.hidden1)
        self.fc2 = nn.Linear(self.hidden1, self.hidden2)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc3 = nn.Linear(self.hidden2, 10) # 10 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
