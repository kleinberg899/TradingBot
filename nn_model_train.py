## not running yet

epochs = 100
iterations_per_stock = 10

stocks = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
    ]

def get_batch(batch_size):
    ix = torch.randint(data_tensor.shape[0] - context_size, (batch_size,))
    x = torch.stack([data_tensor[i:i+context_size] for i in ix])
    y = torch.stack([data_tensor[i + context_size + dist_target_from_context, 5] for i in ix])
    return x, y


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
