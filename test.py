import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import Bot_LinearRegression as Bot

test_data = pd.read_csv('data/stock_data/DTE.DE.csv', sep=',')
test_data['Date'] = pd.to_datetime(test_data['Date'])

reference_date = pd.Timestamp('2000-01-01')
test_data['Days'] = (test_data['Date'] - reference_date).dt.days
model = LinearRegression()
model.fit(test_data[['Days']], test_data['Close'])
# Vorhersage
predictions = model.predict(test_data[['Days']])



plt.figure(figsize=(10,6))
plt.scatter(test_data['Days'], test_data['Close'], label='Data Points')
plt.plot(test_data['Days'], predictions, color='red', label='Linear Regression')
plt.xlabel('Days since reference date')
plt.ylabel('Close Price')
plt.title('Linear Regression of Close Price over Time')
plt.legend()
plt.grid(True)
plt.show()



####

context_size = 32
dist_target_from_context = 8



def get_batch(batch_size):
    ix = torch.randint(data_tensor.shape[0] - context_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+context_size] for i in ix])
    y = torch.stack([data_tensor[i + context_size + dist_target_from_context, 5] for i in ix])
    return x, y

epochs = 100
iterations_per_stock = 100

stocks = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
    ]


for epoch in len(epochs):
  ix = torch.randint(0, len(stocks), (1,))
  data = torch.tensor(pd.Dataframe(download_stock_data(stocks[ix] +".DE", start_date, end_date)))

  adam = torch.optim.AdamW(model.parameters(), lr=1e-3)

  batch_size = 32
  for steps in range(iterations_per_stock):

      # sample a batch of data
      xb, yb = get_batch(batch_size)

      # evaluate the loss
      prediction, loss = model(xb, yb)
      adam.zero_grad(set_to_none=True)
      loss.backward()
      adam.step()

  print(loss.item())
