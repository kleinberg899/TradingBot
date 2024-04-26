import yfinance as yf
import datetime

def download_stock_data(symbol, start_date, end_date):
    try:
        # Daten von Yahoo Finance herunterladen
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print("Fehler beim Herunterladen der Daten:", e)
        return None

def save_to_csv(data, filename):
    try:
        # Daten in eine CSV-Datei speichern
        data.to_csv(filename)
        print("Daten erfolgreich in", filename, "gespeichert.")
    except Exception as e:
        print("Fehler beim Speichern der Daten:", e)



dax_symbole = [
    "ADS", "AIR", "ALV", "BAS", "BAYN", "BEI", "BMW", "BNR", "CBK", "CON",
    "1COV", "DTG", "DBK", "DB1", "DHL", "DTE", "EOAN", "FRE", "HNR1", "HEI",
    "HEN3", "IFX", "MBG", "MRK", "MTX", "MUV2", "P911", "PAH3", "QIA", "RHM",
    "RWE", "SAP", "SRT3", "SIE", "ENR", "SHL", "SY1", "VOW3", "VNA", "ZAL"
]

for symbol in dax_symbole:
  print(symbol)
  start_date = datetime.datetime(2019, 4, 24)
  end_date = datetime.datetime(2024, 5, 25)

  # Daten herunterladen
  data = download_stock_data(symbol +".DE", start_date, end_date)

  if data is not None:
      path = "data/stock_data/" + symbol + ".DE.csv"
      print(path)
      save_to_csv(data, path)

  else:
    print('Error: ', symbol)