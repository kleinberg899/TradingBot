
import torch
from torchmetrics import MeanAbsoluteError

#from price_estimation.FeedforwardNN import Model
from price_estimation.Up_Down_Classifikation.More_Complex_Timeseries_NN import Model

context_size = 14
dist_target_from_context = 1
epochs = 200
iterations_per_stock = 100
batch_size = 16
feature_size = 28
learning_rate = 3e-4
col_position_of_target = 3


model = Model(feature_size, context_size)


model = Model(feature_size, context_size)


num_parameters = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_parameters}')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
}, '../../models/model_feed_forward_classifier.pth')
