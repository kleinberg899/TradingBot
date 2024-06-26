import pandas as pd
from pandas import read_csv
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import torch
import yfinance as yf
import datetime
import torch.nn as nn
from sklearn import preprocessing
from torch import tensor
import numpy as np
from torchmetrics.regression import MeanAbsolutePercentageError

# stocks to train the neural network

stocks = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE",# "CON.DE","SHL.DE",
    "1COV.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE",  "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
]

# Hyperparameters

context_size = 356
dist_target_from_context = 7
epochs = 200
iterations_per_stock = 25
batch_size = 64
input_size = 7 * context_size
learning_rate = 3e-4
col_position_of_target  =2

start_date = datetime.datetime(2017, 3, 1)
end_date = datetime.datetime(2022, 1, 4)

# drop these columns of the csv's to not include certain parameters in training data
columns_to_drop = ['Date']


# Returns batch of one single stock with target values
# For each stock a random starting point is chosen, x = stock[starting_point:starting_point+context_size]
# y = stock[starting_point+context_size+dist_target_from_context]
def get_batch(batch_size, data_tensor, context_size, dist_target_from_context, col_position_of_target=2):
    ix = torch.randint(data_tensor.shape[0] - context_size - dist_target_from_context, (batch_size,))
    price_normalisation = torch.stack([data_tensor[i + context_size - 1, col_position_of_target] for i in ix])
    if torch.any(price_normalisation == 0):
        print("TEILEN DURCH 0 DU HUND. BEHANDEL EXCEPTION")

    x = torch.stack([data_tensor[i:i + context_size] for i in ix]) / price_normalisation.unsqueeze(1).unsqueeze(2)
    y = torch.stack([data_tensor[i + context_size + dist_target_from_context, col_position_of_target] for i in ix]) / price_normalisation
    return x, y

def get_batch_and_normalization(batch_size, data_tensor, context_size, dist_target_from_context, col_position_of_target=2):
    ix = torch.randint(data_tensor.shape[0] - context_size - dist_target_from_context, (batch_size,))
    price_normalisation = torch.stack([data_tensor[i + context_size - 1, col_position_of_target] for i in ix])
    if torch.any(price_normalisation == 0):
        print("TEILEN DURCH 0 DU HUND. BEHANDEL EXCEPTION")

    x = torch.stack([data_tensor[i:i + context_size] for i in ix]) / price_normalisation.unsqueeze(1).unsqueeze(2)
    y = torch.stack([data_tensor[i + context_size + dist_target_from_context, col_position_of_target] for i in ix]) / price_normalisation
    return x, y, price_normalisation


# model architecture
model = nn.Sequential(
    nn.Linear(input_size, 200),
    nn.ReLU(),
    nn.Linear(200, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    # nn.Linear(100, 50),
    # nn.ReLU(),
    nn.Linear(50, 1),
)
loss_fn = MeanAbsolutePercentageError()

num_parameters = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_parameters}')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

losses = []

for epoch in range(epochs):

    # choose one random stock per epoch and turn it into a tensor after dropping columns_to_drop
    ix = torch.randint(0, len(stocks), (1,))
    print(stocks[ix])
    raw_data = pd.read_csv('data/stock_data/' + stocks[ix] + '.csv', parse_dates=['Date'])
    data = raw_data[(raw_data['Date'] >= start_date) & (raw_data['Date'] <= end_date)].drop(columns_to_drop, axis=1)
    data['Volume'] = data['Volume'] / 10000.
    data_tensor = torch.tensor(data.values[:, 1:].astype(float), dtype=torch.float32)
    for steps in range(iterations_per_stock):
        # sample a batch of data
        xb, yb = get_batch(batch_size, data_tensor, context_size, dist_target_from_context)

        # evaluate the loss
        B, T, C = xb.shape
        prediction = model(xb.view(B, T * C))
        yb = yb.view(-1, 1)
        loss = loss_fn(prediction, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = torch.tensor(losses[-50:]).mean().item()
    formatted_loss = "{:.5f}".format(mean_loss)

    print('epoch', epoch, ':', formatted_loss)

# Check accuracy on one random batch for every stock

def check_accuracy(model, test_start_date, test_end_date):
    test_loss_list = []

    for stock in stocks:
        test_raw_data = pd.read_csv('data/stock_data/' + stock + '.csv', parse_dates=['Date'])
        test_data = test_raw_data[(test_raw_data['Date'] >= test_start_date) & (
                test_raw_data['Date'] <= test_end_date)].drop(columns_to_drop, axis=1)
        test_data['Volume'] = test_data['Volume'] / 10000.
        test_data_tensor = torch.tensor(test_data.values[:, 1:].astype(float), dtype=torch.float32)
        test_x, test_y, normalization_tensor = get_batch_and_normalization(batch_size, test_data_tensor, context_size, dist_target_from_context, col_position_of_target)

        model.eval()

        with torch.no_grad():
            B, T, C = test_x.shape
            scores = model(test_x.view(B, T * C))
            test_loss = loss_fn(scores, test_y.view(-1, 1)).item()
            test_loss_list.append(test_loss)
            print('test loss on stock ', stock, '= ', test_loss)

        for i in range(1):
            print(i, "prediction:", scores[i].item(), "target:", test_y[i].item(), "diff:", scores[i].item()-test_y[i].item())
            print(i, "prediction:", (scores[i] * normalization_tensor[i]).item(), "target:", (test_y[i] * normalization_tensor[i]).item(),'diff:',(scores[i] * normalization_tensor[i]).item() -(test_y[i] * normalization_tensor[i]).item())
            print("-------------")
    median_loss = np.median(test_loss_list)
    mean_loss = np.mean(test_loss_list)
    variance_loss = np.var(test_loss_list)
    print('test loss median: ', median_loss)
    print('test loss mean: ', mean_loss)
    print('test loss variance: ', variance_loss)


check_accuracy(model, datetime.datetime(2022, 2, 4), datetime.datetime(2024, 3, 4))
losses_train_tensor = torch.tensor(losses).view(-1, iterations_per_stock).mean(dim=1)
index_tensor = torch.arange(0, losses_train_tensor.shape[0])
plt.plot(index_tensor, losses_train_tensor)
plt.ylim(0, 0.6)
plt.show()

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'models/model_feed_forward.pth')
