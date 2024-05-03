import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError


class Model(nn.Module):
    def __init__(self, num_features, time_steps, hidden_size1=10, hidden_size2=25, output_size=1):
        super(Model, self).__init__()
        self.num_features = num_features
        self.time_steps = time_steps
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Liste von Fully-Connected-Schichten erstellen
        self.feature_layers = nn.ModuleList([nn.Linear(time_steps, hidden_size1) for _ in range(num_features)])

        self.fc_almost_final = nn.Linear(num_features * hidden_size1, hidden_size2)
        self.fc_final = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        if x.dim() == 3:
            b, t, c = x.shape
        else:
            b = 1
            t, c = x.shape
            x = x.unsqueeze(0)
        feature_activations = []

        for feature_idx in range(self.num_features):
            feature_data = x[:, :, feature_idx]
            feature_data = feature_data.view(b, t)
            out = self.relu(self.feature_layers[feature_idx](feature_data))  # Korrektur hier
            feature_activations.append(out)

        if x.dim() == 3:
            tmp = torch.cat(feature_activations, dim=1)  # Konkateniere die Ausgaben entlang der Dimension 1
        else:
            tmp = torch.cat(feature_activations, dim=0)

        tmp = self.relu(self.fc_almost_final(tmp))
        out = self.fc_final(tmp)
        return out