
import torch
from torchmetrics import MeanAbsoluteError

#from price_estimation.FeedforwardNN import Model
from price_estimation.regression.More_Complex_Timeseries_NN import Model
context_size = 365
dist_target_from_context = 7
epochs = 200
iterations_per_stock = 25
batch_size = 64
feature_size = 28
learning_rate = 3e-4
col_position_of_target = 3


model = Model(feature_size, context_size)

loss_fn = MeanAbsoluteError()

num_parameters = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_parameters}')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

losses = []

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
}, '../../models/model_feed_forward.pth')
