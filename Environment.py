import Bot_LinearRegression as Bot
import pandas as pd

def string_to_date(string):
  year = string[0:5]
  string = string[6:]
  string.find('-')


class Environment:
  def __init__(self, path_to_data_folder):

    self.data_path = path_to_data_folder

    self.stocks = [
    "ADS.DE", "AIR.DE","ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
    ]
    self.bot = Bot.Bot_LinearRegression(path_to_data_folder)

  def get_price(self, stock, date):
    #date = pd.Timestamp(date)
    stock_data = pd.read_csv(self.data_path + '/' + stock + '.csv')
    close_value = stock_data.loc[stock_data['Date'] == date, 'Close'].values
    if len(close_value) > 0:
      return close_value[0]  # Rückgabe des Close-Werts, falls gefunden
    else:
      return None
