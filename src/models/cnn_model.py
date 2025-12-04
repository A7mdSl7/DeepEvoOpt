import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, hyperparams):
        super(CNN, self).__init__()
        
        # Hyperparameters
        self.conv1_out = int(hyperparams.get('conv1_out', 16))
        self.kernel1 = int(hyperparams.get('kernel1', 3))
        self.conv2_out = int(hyperparams.get('conv2_out', 32))
        self.kernel2 = int(hyperparams.get('kernel2', 3))
        self.fc1_out = int(hyperparams.get('fc1_out', 128))
        self.dropout_rate = float(hyperparams.get('dropout', 0.5))
        
        # Layers
        self.conv1 = nn.Conv2d(1, self.conv1_out, kernel_size=self.kernel1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(self.conv1_out, self.conv2_out, kernel_size=self.kernel2, padding=1)
        
        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            x = self.pool(F.relu(self.conv1(dummy_input)))
            x = self.pool(F.relu(self.conv2(x)))
            self.flattened_size = x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.flattened_size, self.fc1_out)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(self.fc1_out, 10) # 10 classes for Fashion-MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flattened_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
