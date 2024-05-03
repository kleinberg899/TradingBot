import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError


class Model(nn.Module):
    def __init__(self, input_size, num_layers=20, hidden_size1=25, hidden_size2=25, output_size=1):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        # Liste von Fully-Connected-Schichten erstellen
        self.fc_layers = nn.ModuleList([nn.Linear(input_size, hidden_size1) for _ in range(num_layers)])
        self.fc_almost_final = nn.Linear(hidden_size1 * num_layers, hidden_size2)
        self.fc_final = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = [self.relu(fc(x)) for fc in
               self.fc_layers]  # Durchlaufe alle Schichten und führe den ReLU-Aktivierungsfunktion aus
        if out[0].dim() == 2:
            tmp = torch.cat(out, dim=1)  # Konkateniere die Ausgaben entlang der Dimension 1
        else:
            tmp = torch.cat(out, dim=0)
        tmp = self.relu(self.fc_almost_final(tmp))
        out = self.fc_final(tmp)
        return out