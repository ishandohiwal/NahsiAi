import torch
import torch.nn as nn

class CustomBrainNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomBrainNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.relu(self.layer2(out))
        out = self.layer3(out)
        return out
