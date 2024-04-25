import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

class Bot_LinearRegression():
  def __init__(self, path_to_data_folder):
    self.data_path = path_to_data_folder
      #holds model parameters

  def __call__(self, stock, start_date, end_date, prediction_date):

    stock_data = self.load_stock_data(stock)

    # Filtere Daten zwischen start_date und end_date
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= end_date)]
    num_rows = stock_data.shape[0]
    if num_rows == 0:
      return None


    # Umwandlung des Datums in Tage seit einem Referenzdatum
    reference_date = pd.Timestamp(start_date)
    prediction_date = pd.Timestamp(prediction_date)
    stock_data['Days'] = (stock_data['Date'] - reference_date).dt.days
    model = LinearRegression()
    model.fit(stock_data[['Days']], stock_data['Close'])
    # Vorhersage
    predictions = model.predict([[(prediction_date - reference_date).days]])
    return predictions[0]

  def load_stock_data(self, stock):
    stock_data = pd.read_csv(self.data_path + '/' + stock + '.csv')
    return stock_data
    # Bot läd die csv file des geg. Stocks in Speicher


  def plot_prediction(self, stock, start_date, end_date, prediction_date):
    ... # soll plt erstellen