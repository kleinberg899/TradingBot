import matplotlib.pyplot as plt
import pandas as pd
import torch
import datetime
import numpy as np
from torchmetrics import MeanAbsoluteError
from torchmetrics.regression import MeanAbsolutePercentageError

from FeedforwardNN import Model

'''
def elliptic_paraboloid_loss(x, y, c_diff_sign=100, c_same_sign=99):
    # Convert inputs to tensors
    #x_tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    #y_tensor = torch.tensor(y, dtype=torch.float32, requires_grad=True)

    x_tensor = x.clone().detach().requires_grad_(True)
    y_tensor = y.clone().detach().requires_grad_(True)

    # Compute Mean Squared Error
    mse = torch.nn.MSELoss()(x_tensor, y_tensor)

    # Adjust inputs
    x_tensor_adjusted = x_tensor - 1
    y_tensor_adjusted = y_tensor - 1

    # Compute a rotated elliptic paraboloid
    t = torch.tensor(np.pi / 4, dtype=torch.float32)
    x_rot = (x_tensor_adjusted * torch.cos(t)) + (y_tensor_adjusted * torch.sin(t))
    y_rot = (x_tensor_adjusted * -torch.sin(t)) + (y_tensor_adjusted * torch.cos(t))
    z = ((x_rot**2) / c_diff_sign) + ((y_rot**2) / c_same_sign)

    # Multiply by MSE
    ell_loss = z.sum() * mse

    return ell_loss

'''

# stocks to train the neural network

stocks = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE",  # "CON.DE","SHL.DE",
    "1COV.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE", "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "JNJ", "V", "NVDA",
    "BAC", "PG", "HD", "UNH", "MA", "PYPL", "INTC", "CMCSA", "VZ", "DIS",
    "ADBE", "NFLX", "CRM", "ABT", "XOM", "CSCO", "PEP", "KO",  "WMT",
    "MRK", "MDT", "NKE", "ABBV", "TXN", "BMY", "T", "GILD", "IBM", "CVX",
    "MDLZ", "LMT", "UPS", "MO", "INTU", "FDX", "AMGN", "PM", "MMM", "DUK", 'AGS.BR', 'AZN.L', 'AIR.PA', 'ENGI.PA', 'AGS.BR', 'AZN.L', 'AIR.PA', 'ENGI.PA',
 'NOVN.SW', 'SOLB.BR', 'CS.PA', 'DG.PA', 'SIE.DE', 'TTE.PA',
 'SU.PA', 'SAN.PA', 'OR.PA', 'SAF.PA', 'UMI.BR', 'BP.L', 'HEIA.AS',
 'ISP.MI', 'ROG.SW', 'ENEL.MI', 'INGA.AS', 'MC.PA', 'ALO.PA',
 'GSK.L', 'VIV.PA', 'SAP.DE', 'IBE.MC', 'BNP.PA',
 'AI.PA', 'ABI.BR', 'HSBA.L', 'AED.BR', 'ASML.AS', 'ZURN.SW',
 'NESN.SW'
]

# Hyperparameters

context_size = 365
dist_target_from_context = 7
epochs = 200
iterations_per_stock = 25
batch_size = 64
input_size = 8 * context_size
learning_rate = 3e-4
col_position_of_target = 3



start_date = datetime.datetime(2010, 3, 1)
end_date = datetime.datetime(2021, 1, 4)

# drop these columns of the csv's to not include certain parameters in training data
columns_to_drop = ['Date','SMA_50','EMA_50','EMA_20','RSI_14','BBL_20_2.0','BBM_20_2.0','BBU_20_2.0','BBB_20_2.0','BBP_20_2.0','MACD_12_26_9','MACDh_12_26_9','MACDs_12_26_9','STOCHk_14_3_3','STOCHd_14_3_3','ATRr_14','OBV','ADX_14','DMP_14','DMN_14']

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
        return x, y
    # Normalisieren Sie nur die ersten 3 Spalten in x
    x = torch.stack([data_tensor[i:i + context_size] for i in ix])
    x[:, :, :4] = x[:, :, :4] / price_normalisation.unsqueeze(1).unsqueeze(1)  # Normalisierung der ersten 3 Spalten
    # Ziel normalisieren
    y = torch.stack([data_tensor[i + context_size + dist_target_from_context, col_position_of_target] for i in
                     ix])/ price_normalisation
    return x, y


def get_batch_and_normalization(batch_size, data_tensor, context_size, dist_target_from_context, col_position_of_target):
    ix = torch.randint(data_tensor.shape[0] - context_size - dist_target_from_context, (batch_size,))
    price_normalisation = torch.stack([data_tensor[i + context_size - 1, col_position_of_target] for i in ix])
    if torch.any(price_normalisation == 0):
        print("TEILEN DURCH 0 DU HUND. BEHANDEL EXCEPTION")
        x = torch.zeros((batch_size, context_size, data_tensor.shape[1]))
        y = torch.zeros(batch_size)
        return x, y
    # Normalisieren Sie nur die ersten 3 Spalten in x
    x = torch.stack([data_tensor[i:i + context_size] for i in ix])
    x[:, :, :4] = x[:, :, :4] / price_normalisation.unsqueeze(1).unsqueeze(1)  # Normalisierung der ersten 3 Spalten
    # Ziel normalisieren
    y = torch.stack([data_tensor[i + context_size + dist_target_from_context, col_position_of_target] for i in
                     ix])/ price_normalisation
    return x, y, price_normalisation


# Initialisiere das Modell
model = Model(input_size)

# Initialisiere den Optimizer und den Loss
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()
#loss_fn = torch.nn.MeanAbsoluteError()

# Lade das gespeicherte Modell und die zugehörigen Parameter
checkpoint = torch.load('models/model_feed_forward.pth')

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

for epoch in range(epochs):

    # choose one random stock per epoch and turn it into a tensor after dropping columns_to_drop
    ix = torch.randint(0, len(stocks), (1,))
    print(stocks[ix])
    raw_data = pd.read_csv('data/stock_data/' + stocks[ix] + '.csv')
    raw_data.rename(columns={raw_data.columns[0]: 'Date'}, inplace=True)
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    data = raw_data[(raw_data['Date'] >= start_date) & (raw_data['Date'] <= end_date)].drop(columns_to_drop, axis=1)
    data['Volume'] = data['Volume'] / 10000.
    data_tensor = torch.tensor(data.values[:, 1:].astype(float), dtype=torch.float32)
    for steps in range(iterations_per_stock):
        # sample a batch of data
        xb, yb = get_batch(batch_size, data_tensor, context_size, dist_target_from_context, col_position_of_target)

        # evaluate the loss
        B, T, C = xb.shape
        prediction = model(xb.reshape(B, -1))
        yb = yb.reshape(-1, 1)
        loss = loss_fn(prediction, yb)
        #print(stocks[ix], prediction[0].item(), yb[0].item())
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
        test_raw_data = pd.read_csv('data/stock_data/' + stock + '.csv')
        test_raw_data.rename(columns={test_raw_data.columns[0]: 'Date'}, inplace=True)
        test_raw_data['Date'] = pd.to_datetime(test_raw_data['Date'])
        test_data = test_raw_data[(test_raw_data['Date'] >= test_start_date) & (
                test_raw_data['Date'] <= test_end_date)].drop(columns_to_drop, axis=1)
        test_data['Volume'] = test_data['Volume'] / 10000.
        test_data_tensor = torch.tensor(test_data.values[:, 1:].astype(float), dtype=torch.float32)
        test_x, test_y, normalization_tensor = get_batch_and_normalization(batch_size, test_data_tensor, context_size,
                                                                           dist_target_from_context,
                                                                           col_position_of_target)

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


check_accuracy(model, datetime.datetime(2020, 2, 4), datetime.datetime(2024, 3, 4))
losses_train_tensor = torch.tensor(losses).view(-1, iterations_per_stock).mean(dim=1)
index_tensor = torch.arange(0, losses_train_tensor.shape[0])
plt.plot(index_tensor, losses_train_tensor)
plt.ylim(0, 0.1)
plt.show()

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'losses': losses
}, 'models/model_feed_forward.pth')
