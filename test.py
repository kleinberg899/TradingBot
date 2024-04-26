import pandas as pd
from pandas import read_csv
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import Bot_LinearRegression as Bot
import pandas as pd
import torch
import yfinance as yf
import datetime
import torch.nn as nn


def load_stock_data(stock):
    data = read_csv('data/stock_data' + '/' + stock + '.csv')

    df = pd.DataFrame(data)

    # Umwandlung der 'Date'-Spalte in ein Datetime-Objekt
    df['Date'] = pd.to_datetime(df['Date'])

    # Erstellen eines vollständigen Datumsbereichs
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max())

    # Neuindexierung des DataFrame mit dem vollständigen Datumsbereich
    df = df.set_index('Date').reindex(date_range).reset_index()
    df = df.apply(pd.to_numeric)

    # Funktion zur Aktualisierung der Werte
    def update_values(row):
        if pd.isna(row['Open']):
            row['Open'] = row['Close']
        if pd.isna(row['High']):
            row['High'] = row['Close']
        if pd.isna(row['Low']):
            row['Low'] = row['Close']
        if pd.isna(row['Close']):
            row['Close'] = 0
        if pd.isna(row['Adj Close']):
            row['Adj Close'] = row['Close']
        if pd.isna(row['Volume']):
            row['Volume'] = 0
        return row

    last_close = None
    for index, row in df.iterrows():
        if not pd.isna(row['Close']):
            last_close = row['Close']
        else:
            row[
                'Close'] = last_close if last_close is not None else 0  # Ensure last_close is converted to appropriate data type
        df.loc[index] = update_values(row)
    df = df.fillna(0.0)
    df.rename(columns={'index': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df




####

context_size = 190
dist_target_from_context = 7
epochs = 2000
iterations_per_stock = 20
batch_size = 32
input_size = 1 * context_size

start_date = datetime.datetime(2020, 3, 1)
end_date = datetime.datetime(2024, 1, 4)
columns_to_drop = ['Open', 'High', 'Low', 'Adj Close', 'Volume']



def get_batch(batch_size, data_tensor, context_size, dist_target_from_context):
    ix = torch.randint(data_tensor.shape[0] - context_size - dist_target_from_context, (batch_size,))
    x = torch.stack([data_tensor[i:i + context_size] for i in ix])
    y = torch.stack([data_tensor[i + context_size + dist_target_from_context, 0] for i in ix])
    return x, y


stocks = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
]



stock_dict = {}

for stock in stocks:
    stock_dict[stock] = load_stock_data(stock)
    print('loading',stock)

model = nn.Sequential(
    nn.Linear(input_size, 500),
    nn.ReLU(),
    nn.Linear(500, 250),
    nn.ReLU(),
    nn.Linear(250, 100),
    nn.ReLU(),
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 1),
)
loss_fn = nn.MSELoss()

num_parameters = sum(p.numel() for p in model.parameters())
print(f'Number of parameters: {num_parameters}')
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

losses = []

for epoch in range(epochs):
    ix = torch.randint(0, len(stocks), (1,))
    #raw_data = pd.read_csv('data/stock_data/' + stocks[ix] + '.csv', parse_dates=['Date'])
    #raw_data = load_stock_data(stocks[ix])
    # raw_data = pd.read_csv('data/stock_data/AIR.DE.csv', parse_dates=['Date'])
    #data = raw_data[(raw_data['Date'] >= start_date) & (raw_data['Date'] <= end_date)]
    raw_data = stock_dict[stocks[ix]]
    data = raw_data[(raw_data['Date'] >= start_date) & (raw_data['Date'] <= end_date)].drop(columns_to_drop, axis=1)
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
    #fds
    #print(loss.item())
    #print(torch.tensor(losses[-50:]).mean(0))
    print('epoch: ', epoch, torch.tensor(losses[-50:]).mean(0).item())


losses_train_tensor = torch.tensor(losses).view(-1, iterations_per_stock).mean(dim=1)
index_tensor = torch.arange(0, losses_train_tensor.shape[0])
plt.plot(index_tensor, losses_train_tensor)
plt.ylim(0, 1000)
plt.show()

