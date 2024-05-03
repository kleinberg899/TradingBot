import matplotlib.pyplot as plt
import pandas as pd
import torch
import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch
import datetime
import numpy as np
#!pip install torchmetrics
from torchmetrics.regression import MeanAbsolutePercentageError
import torch.nn.functional as F


# stocks to train the neural network

stocks = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE",
    "1COV.DE",  "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE",  "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
]

# Hyperparameters

from price_estimation.Up_Down_Classifikation.More_Complex_Timeseries_NN import Model

context_size = 14
dist_target_from_context = 1

batch_size = 16
feature_size = 28
learning_rate = 3e-4
col_position_of_target = 3


model = Model(feature_size, context_size)



start_date = datetime.datetime(2017, 3, 1)
end_date = datetime.datetime(2021, 12, 30)

# drop these columns of the csv's to not include certain parameters in training data
columns_to_drop = ['Date']

# Returns batch of one single stock with target values
# For each stock a random starting point is chosen, x = stock[starting_point:starting_point+context_size]
# y = stock[starting_point+context_size+dist_target_from_context]
def get_batch(batch_size, data_tensor, context_size, dist_target_from_context, col_position_of_target):
    ix = torch.randint(data_tensor.shape[0] - context_size - dist_target_from_context, (batch_size,))
    price_normalisation = torch.stack([data_tensor[i + context_size - 1, col_position_of_target] for i in ix])
    if torch.any(price_normalisation == 0):
        print("TEILEN DURCH 0 DU HUND. BEHANDEL EXCEPTION")
        x = torch.zeros((batch_size, context_size, data_tensor.shape[1]))
        y = torch.zeros(batch_size)
        return x, y, 1
    cols_to_normalize = [0,1,2,3,4,6,7,8,9]
    # Normalisieren Sie nur die ersten 3 Spalten in x
    x = torch.stack([data_tensor[i:i + context_size] for i in ix])
    x[:, :, cols_to_normalize] = x[:, :, cols_to_normalize] / price_normalisation.unsqueeze(1).unsqueeze(1)
    # Ziel normalisieren

    Y = torch.stack([data_tensor[i + context_size + dist_target_from_context, col_position_of_target] for i in
                     ix]) /price_normalisation

    Y_replaced = torch.where(Y < 0.999, torch.tensor(0),
                          torch.where(Y > 1.001, torch.tensor(2.0),
                                      torch.tensor(1.0))).long()

    return x, Y_replaced, price_normalisation




model = Model(feature_size, time_steps= context_size)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#loss_fn = torch.nn.MSELoss()
#loss_fn = MeanAbsolutePercentageError()
loss_fn = torch.nn.CrossEntropyLoss()



# Initialisiere den Optimizer und den Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#loss_fn = torch.nn.MSELoss()
loss_fn = MeanAbsolutePercentageError()

# Lade das gespeicherte Modell und die zugehörigen Parameter
checkpoint = torch.load('../../models/model_feed_forward_classifier.pth')

# Lade das Modell
model.load_state_dict(checkpoint['model_state_dict'])

# Setze den Optimizer auf den Zustand beim Speichern zurück
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Setze die Verluste auf den Zustand beim Speichern zurück
losses = checkpoint['losses']

# Schalte das Modell in den Evaluierungsmodus
#model.eval()
model.train()

num_parameters = sum(p.numel() for p in model.parameters())
print(f'Number of parameters loaded: {num_parameters}')

model.train()

num_parameters = sum(p.numel() for p in model.parameters())
print(f'Number of parameters loaded: {num_parameters}')

count = [0,0,0]
epochs = 1000
iterations_per_stock = 100

for epoch in range(epochs):
    print(epoch)
    # choose one random stock per epoch and turn it into a tensor after dropping columns_to_drop
    ix = torch.randint(0, len(stocks), (1,))
    #print(stocks[ix])
    raw_data = pd.read_csv('../../data/stock_data/' + stocks[ix] + '.csv')
    raw_data.rename(columns={raw_data.columns[0]: 'Date'}, inplace=True)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    data = raw_data[(raw_data['Date'] >= start_date) & (raw_data['Date'] <= end_date)].drop(columns_to_drop, axis=1)
    data_tensor = torch.tensor(data.values.astype(float), dtype=torch.float32)

    for steps in range(iterations_per_stock):
        # sample a batch of data
        xb, yb, _ = get_batch(batch_size, data_tensor, context_size, dist_target_from_context, col_position_of_target)
        # evaluate the loss
        B, T, C = xb.shape
        #print(yb.shape)
        prediction = model.forward(xb)

        loss = F.cross_entropy(prediction, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

# Check accuracy on one random batch for every stock
print('count', count)

def check_accuracy(model, test_start_date, test_end_date):
    test_loss_list = []
    c,f,total = 0,0,0
    totaly_false = 0
    didnt_know = 0
    for stock in stocks:
        test_raw_data = pd.read_csv('../../data/stock_data/' + stock + '.csv')
        test_raw_data.rename(columns={test_raw_data.columns[0]: 'Date'}, inplace=True)
        test_raw_data['Date'] = pd.to_datetime(test_raw_data['Date'])
        test_data = test_raw_data[(test_raw_data['Date'] >= test_start_date) & (
                test_raw_data['Date'] <= test_end_date)].drop(columns_to_drop, axis=1)
        test_data_tensor = torch.tensor(test_data.values.astype(float), dtype=torch.float32)
        test_x, test_y, normalization_tensor = get_batch(500, test_data_tensor, context_size,
                                                                           dist_target_from_context,
                                                                           col_position_of_target)
        model.eval()
        with torch.no_grad():
            B, T, C = test_x.shape
            scores = model.forward(test_x)
            test_loss = F.cross_entropy(scores, test_y)
            for i in range(scores.shape[0]):
              if torch.argmax(scores[i]).item() ==  test_y[i].item():
                c += 1
              elif (torch.argmax(scores[i]).item() ==  0 and test_y[i].item() == 2) or (torch.argmax(scores[i]).item() ==  2 and test_y[i].item() == 0):
                totaly_false += 1
              elif (torch.argmax(scores[i]).item() ==  1 and test_y[i].item() != 1):
                didnt_know += 1
              else:
                f += 1
              total += 1
            test_loss_list.append(test_loss)
            print('test loss on stock ', stock, '= ', test_loss)

        for i in range(1):
            print("-------------")
    mean_loss = np.mean(test_loss_list)
    print('test loss mean: ', mean_loss)
    print('Richtig:', c, 'Falsch', f, 'totaly_false', totaly_false, 'didnt_know', didnt_know)
    print('Richtig:', c/total, 'Falsch', f/total, 'totaly_false', totaly_false/total, 'didnt_know', didnt_know/total)

with torch.no_grad():
    check_accuracy(model, datetime.datetime(2022, 1, 2), datetime.datetime(2024, 3, 4))
    losses_train_tensor = torch.tensor(losses[:100000]).view(-1, 100).mean(dim=1)
    index_tensor = torch.arange(0, losses_train_tensor.shape[0])
    plt.plot(index_tensor, losses_train_tensor)
    plt.show()
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
}, '../../models/model_feed_forward_classifier.pth')