import yfinance as yf
import pandas as pd
import datetime

def download_stock_data(symbol, start_date, end_date):
    try:
        # Daten von Yahoo Finance herunterladen
        data = yf.download(symbol, start=start_date, end=end_date)
        return data
    except Exception as e:
        print("Fehler beim Herunterladen der Daten für", symbol, ":", e)
        return None

def save_to_csv(data, filename):
    try:
        # Daten in eine CSV-Datei speichern
        data.to_csv(filename)
        print("Daten erfolgreich in", filename, "gespeichert.")
    except Exception as e:
        print("Fehler beim Speichern der Daten:", e)
        
def fill_missing_dates(df):
    copy_df = df.copy()
    desired_index = pd.date_range(start_date, end_date)
    df = copy_df.reindex(desired_index)
    df['Volume'] = df['Volume'].fillna(0)
    df = df.fillna(method='ffill')
    df = df.fillna(0)
    return df

dax_symbols = [
    "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE", "CON.DE",
    "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
    "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "P911.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
    "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
]

us_symbols = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "JPM", "JNJ", "V", "NVDA",
    "BAC", "PG", "HD", "UNH", "MA", "PYPL", "INTC", "CMCSA", "VZ", "DIS",
    "ADBE", "NFLX", "CRM", "ABT", "XOM", "CSCO", "PEP", "KO", "MRNA", "WMT",
    "MRK", "MDT", "NKE", "ABBV", "TXN", "BMY", "T", "GILD", "IBM", "CVX",
    "MDLZ", "LMT", "UPS", "MO", "INTU", "FDX", "AMGN", "PM", "MMM", "DUK"
]

eu_symbols = [
    "AI.PA", "AIR.PA", "ALO.PA", "ASML.AS", "CS.PA", "BNP.PA", "ENEL.MI", "ENGI.PA", "IBE.MC", "MC.PA",
    "INGA.AS", "ISP.MI", "OR.PA", "SAN.PA", "SU.PA", "DG.PA", "FP.PA", "VIV.PA", "SAF.PA", "TEP.L",
    "MCRO.L", "AI.PA", "ASML.AS", "AI.PA", "IBE.MC", "AIR.PA",
    "ABI.BR", "OR.PA", "SAN.PA", "ASML.AS", "CS.PA", "ABI.BR", "SIE.DE", "SAP.DE",
    "VIV.PA", "SU.PA", "BNP.PA", "ASML.AS", "FP.PA", "ASML.AS", "SAN.PA", "SAN.PA"
]

indice_symbols = [
    '^IRX',         # Euro interestrate
    '^TNX',         # Dollar interestrate
    'DAX',          # DAX
    '^STOXX50E',    # EURO STOXX 50
    '^GSPC',        # &P500
    'XWD.TO',       # MSCI World
]


base_path = "/content/sample_data/stock_data_test4/"

start_date = datetime.datetime(2020, 1, 2)
end_date = datetime.datetime(2020, 2, 5)

# Laden aller Indizes und Benennen der Spalten
indices_data = {}
for indice in indice_symbols:
    data = download_stock_data(indice, start_date, end_date)
    if data is not None:
        data = fill_missing_dates(data)
        indices_data[indice] = data.rename(columns={col: col + '_' + indice for col in data.columns})

# Kombinieren der Indizes in einen DataFrame
indices_df = pd.concat(indices_data.values(), axis=1)

# Durchlaufen der Listen der Symbole
for symbol_list in [dax_symbols, us_symbols, eu_symbols]:
    for symbol in symbol_list:
        print("Verarbeite Symbol:", symbol)
        # Daten für das Unternehmen herunterladen
        data = download_stock_data(symbol, start_date, end_date)
        if data is not None:
            df = pd.DataFrame(data)
            df = df.drop(['Adj Close', 'Open'], axis=1)
            df = fill_missing_dates(df)

            # Indizes an die Aktiendaten anhängen
            df_with_indices = pd.concat([df, indices_df], axis=1)

            # Spalte 'Date' sicherstellen
            df_with_indices['Date'] = df_with_indices.index

            # Speichern der Daten in CSV
            path = base_path + symbol + ".csv"
            save_to_csv(df_with_indices, path)
        else:
            print("Fehler beim Herunterladen der Daten für", symbol)
