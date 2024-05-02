import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError


class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 400)
        self.layernorm1 = nn.LayerNorm(400)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(400, 200)
        self.layernorm2 = nn.LayerNorm(200)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(200, 100)
        self.layernorm3 = nn.LayerNorm(100)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.layernorm1(self.fc1(x)))
        out = self.dropout1(out)
        out = self.relu(self.layernorm2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.relu(self.layernorm3(self.fc3(out)))
        out = self.dropout3(out)
        out = self.fc4(out)
        return out
