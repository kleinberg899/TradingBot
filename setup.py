import yfinance as yf
import pandas as pd
import datetime
#!pip install pytrends
from pytrends.request import TrendReq
from pytrends.exceptions import ResponseError




base_path = "data/stock_data/"

start_date = datetime.datetime(2017, 1, 2)
end_date = datetime.datetime(2024, 4, 28)





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
    df = df.ffill()
    df = df.fillna(0)
    return df

import time

def get_google_trends(keyword, start_date, end_date, geo=''):
    try:
        print(keyword, start_date, end_date, geo)
        pytrends = TrendReq()
        timeframe = start_date.strftime('%Y-%m-%d') + ' ' + end_date.strftime('%Y-%m-%d')
        pytrends.build_payload(kw_list=[keyword], timeframe=timeframe, geo=geo)
        interest_over_time_df = pytrends.interest_over_time()
        print("\nResponse-Code: 200")
        interest_over_time_df.drop(columns=['isPartial'], inplace=True)  # Spalte "isPartial" entfernen
        return interest_over_time_df
    except ResponseError as e:
        if "429" in str(e):  # Überprüfen, ob der Fehlercode 429 ist
            print("\nError: Google returned a response with code 429. Retrying after 1 minute...")
            time.sleep(60)  # 1 Minute warten
            return get_google_trends(keyword, start_date, end_date, geo)  # Anfrage erneut senden
        else:
            print(f"\nError: {e}")
            print(" Filling column with zeros.")
            return pd.DataFrame(index=pd.date_range(start_date, end_date), columns=[keyword]).fillna(0)


def fill_missing_dates_trends(df, start_date, end_date):
    copy_df = df.copy()
    desired_index = pd.date_range(start_date, end_date)
    df = copy_df.reindex(desired_index)
    df = df.ffill()
    df = df.fillna(0)
    return df

def merge_with_google_trends(df, keyword, start_date, end_date, geo, col_name):
    print('Google request: ' + col_name, geo, keyword)
    company_name = company_names.get(keyword)
    if not company_name:
        print(f"Company name not found for keyword '{keyword}'. Filling column with zeros.")
        trends_df = pd.DataFrame(index=pd.date_range(start_date, end_date), columns=[col_name]).fillna(0)
    else:
        try:
            trends_df = get_google_trends(company_name, start_date, end_date, geo)
            trends_df = fill_missing_dates_trends(trends_df, start_date, end_date)
            trends_df.rename(columns={'{}'.format(company_name): col_name}, inplace=True)  # Rename the column
        except KeyError:
            print("Error: 'isPartial' column not found in Google Trends data. Filling column with zeros.")
            trends_df = pd.DataFrame(index=pd.date_range(start_date, end_date), columns=[col_name]).fillna(0)
    df_with_trends = pd.concat([df, trends_df], axis=1)
    return df_with_trends

def combine_indices_data(indices, start_date, end_date):
    indices_data = {}
    for indice in indices:
        data = download_stock_data(indice, start_date, end_date)
        if data is not None:
            data = fill_missing_dates(data)
            indices_data[indice] = data.rename(columns={col: col + '_' + indice for col in data.columns})

    # Combine the indices into a DataFrame
    indices_df = pd.concat(indices_data.values(), axis=1)
    return indices_df


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

eu_symbols = ['AGS.BR', 'UNA.AS', 'AZN.L', 'AIR.PA', 'MCRO.L', 'CSGN.SW', 'ENGI.PA',
 'NOVN.SW', 'SOLB.BR', 'CS.PA', 'RDSA.AS', 'DG.PA', 'SIE.DE', 'TTE.PA',
 'SU.PA', 'SAN.PA', 'OR.PA', 'SAF.PA', 'UMI.BR', 'BP.L', 'HEIA.AS',
 'ISP.MI', 'ROG.SW', 'ENEL.MI', 'FP.PA', 'INGA.AS', 'MC.PA', 'ALO.PA',
 'GSK.L', 'VIV.PA', 'SAP.DE', 'IBE.MC', 'BNP.PA', 'DSM.AS', 'RDSB.L',
 'AI.PA', 'ABI.BR', 'HSBA.L', 'TEP.L', 'AED.BR', 'ASML.AS', 'ZURN.SW',
 'NESN.SW']

company_names = {
    "ADS.DE": "Adidas",
    "AIR.DE": "Airbus",
    "ALV.DE": "Allianz",
    "BAS.DE": "BASF",
    "BAYN.DE": "Bayer",
    "BEI.DE": "Beiersdorf",
    "BMW.DE": "BMW",
    "BNR.DE": "Brenntag",
    "CBK.DE": "Commerzbank",
    "CON.DE": "Continental",
    "1COV.DE": "Covestro",
    "DTG.DE": "Telekom",
    "DBK.DE": "Deutsche Bank",
    "DB1.DE": "Deutsche Boerse",
    "DHL.DE": "DHL",
    "DTE.DE": "Deutsche Telekom",
    "EOAN.DE": "E.ON",
    "FRE.DE": "Fresenius",
    "HNR1.DE": "Henkel",
    "HEI.DE": "HeidelbergCement",
    "HEN3.DE": "Henkel",
    "IFX.DE": "Infineon Technologies",
    "MBG.DE": "Munich Re",
    "MRK.DE": "Merck",
    "MTX.DE": "MTU Aero Engines",
    "MUV2.DE": "Muenchener Rueckversicherungs-Gesellschaft",
    "P911.DE": "Porsche",
    "PAH3.DE": "Porsche",
    "QIA.DE": "Qiagen",
    "RHM.DE": "Rheinmetall",
    "RWE.DE": "RWE",
    "SRT3.DE": "Sartorius",
    "SIE.DE": "Siemens",
    "ENR.DE": "E.ON",
    "SHL.DE": "Schaeffler",
    "SY1.DE": "Symrise",
    "VOW3.DE": "Volkswagen",
    "VNA.DE": "Vonovia",
    "ZAL.DE": "Zalando",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "FB": "Facebook",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson",
    "V": "Visa",
    "NVDA": "NVIDIA",
    "BAC": "Bank of America",
    "PG": "Procter & Gamble",
    "HD": "Home Depot",
    "UNH": "UnitedHealth Group",
    "MA": "Mastercard",
    "PYPL": "PayPal",
    "INTC": "Intel",
    "CMCSA": "Comcast",
    "VZ": "Verizon Communications",
    "DIS": "Walt Disney",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "ABT": "Abbott Laboratories",
    "XOM": "Exxon Mobil",
    "CSCO": "Cisco Systems",
    "PEP": "PepsiCo",
    "KO": "Coca-Cola",
    "MRNA": "Moderna",
    "WMT": "Walmart",
    "MDT": "Medtronic",
    "NKE": "Nike",
    "ABBV": "AbbVie",
    "TXN": "Texas Instruments",
    "BMY": "Bristol Myers Squibb",
    "T": "AT&T",
    "GILD": "Gilead",
    "IBM": "IBM",
    "CVX": "Chevron",
    "MDLZ": "Mondelez International",
    "LMT": "Lockheed Martin",
    "UPS": "United Parcel Service",
    "MO": "Altria Group",
    "INTU": "Intuit",
    "FDX": "FedEx",
    "AMGN": "Amgen",
    "PM": "Philip Morris",
    "MMM": "3M",
    "DUK": "Duke Energy",
    "AGS.BR": "Ageas",
    "UNA.AS": "Unilever",
    "AZN.L": "AstraZeneca",
    "AIR.PA": "Airbus",
    "MCRO.L": "Micro Focus International",
    "CSGN.SW": "Credit Suisse Group",
    "ENGI.PA": "Engie",
    "NOVN.SW": "Novartis",
    "SOLB.BR": "Solvay",
    "CS.PA": "AXA",
    "RDSA.AS": "Royal Dutch Shell",
    "DG.PA": "Vinci",
    "SIE.DE": "Siemens",
    "TTE.PA": "Thales",
    "SU.PA": "Schneider Electric",
    "SAN.PA": "Sanofi",
    "OR.PA": "L'Oreal",
    "SAF.PA": "Safran",
    "UMI.BR": "Umicore",
    "BP.L": "BP",
    "HEIA.AS": "Heineken",
    "ISP.MI": "Intesa Sanpaolo",
    "ROG.SW": "Roche Holding",
    "ENEL.MI": "Enel",
    "FP.PA": "TotalEnergies",
    "INGA.AS": "ING Groep",
    "MC.PA": "Louis Vuitton",
    "ALO.PA": "Alstom",
    "GSK.L": "GlaxoSmithKline",
    "VIV.PA": "Vivendi",
    "SAP.DE": "SAP",
    "IBE.MC": "Iberdrola",
    "BNP.PA": "BNP Paribas",
    "DSM.AS": "DSM",
    "RDSB.L": "Royal Dutch Shell",
    "AI.PA": "Air Liquide",
    "ABI.BR": "Anheuser-Busch InBev",
    "HSBA.L": "HSBC Holdings",
    "TEP.L": "Centrica",
    "AED.BR": "Aedifica",
    "ASML.AS": "ASML Holding",
    "ZURN.SW": "Zurich Insurance Group",
    "NESN.SW": "Nestle",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "FB": "Facebook",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson",
    "MRK": "Merck & Co",
    "V": "Visa",
    "NVDA": "NVIDIA",
    "BAC": "Bank of America",
    "PG": "Procter & Gamble",
    "HD": "Home Depot",
    "UNH": "UnitedHealth Group",
    "MA": "Mastercard",
    "PYPL": "PayPal",
    "INTC": "Intel",
    "CMCSA": "Comcast",
    "VZ": "Verizon Communications",
    "DIS": "Walt Disney",
    "ADBE": "Adobe",
    "NFLX": "Netflix",
    "CRM": "Salesforce",
    "ABT": "Abbott Laboratories",
    "XOM": "Exxon Mobil",
    "CSCO": "Cisco Systems",
    "PEP": "PepsiCo",
    "KO": "Coca-Cola",
    "MRNA": "Moderna",
    "WMT": "Walmart",
    "MDT": "Medtronic",
    "NKE": "Nike",
    "ABBV": "AbbVie",
    "TXN": "Texas Instruments",
    "BMY": "Bristol Myers Squibb",
    "T": "AT&T",
    "GILD": "Gilead Sciences",
    "IBM": "IBM",
    "CVX": "Chevron",
    "MDLZ": "Mondelez International",
    "LMT": "Lockheed Martin",
    "UPS": "United Parcel Service",
    "MO": "Altria Group",
    "INTU": "Intuit",
    "FDX": "FedEx",
    "AMGN": "Amgen",
    "PM": "Philip Morris International",
    "MMM": "3M",
    "DUK": "Duke Energy"
}


dax_business_fields = {
    "ADS.DE": "Advertising",
    "AIR.DE": "Aviation",
    "ALV.DE": "Insurance",
    "BAS.DE": "Chemistry",
    "BAYN.DE": "Pharma",
    "BEI.DE": "Energy",
    "BMW.DE": "Automotive",
    "BNR.DE": "Finance",
    "CBK.DE": "Finance",
    "CON.DE": "Construction",
    "1COV.DE": "Chemistry",
    "DTG.DE": "Travel and Tourism",
    "DBK.DE": "Finance",
    "DB1.DE": "Finance",
    "DHL.DE": "Logistics",
    "DTE.DE": "Energy",
    "EOAN.DE": "Energy",
    "FRE.DE": "Real Estate",
    "HNR1.DE": "Insurance",
    "HEI.DE": "Technology",
    "HEN3.DE": "Industry",
    "IFX.DE": "Semiconductor",
    "MBG.DE": "Real Estate",
    "MRK.DE": "Pharma",
    "MTX.DE": "Insurance",
    "MUV2.DE": "Automotive",
    "P911.DE": "Insurance",
    "PAH3.DE": "Chemistry",
    "QIA.DE": "Energy",
    "RHM.DE": "Pharma",
    "RWE.DE": "Energy",
    "SAP.DE": "Software",
    "SRT3.DE": "Retail",
    "SIE.DE": "Industry",
    "ENR.DE": "Energy",
    "SHL.DE": "Technology",
    "SY1.DE": "Industry",
    "VOW3.DE": "Automotive",
    "VNA.DE": "Real Estate",
    "ZAL.DE": "Fashion"
}

us_business_fields = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "AMZN": "E-Commerce",
    "GOOGL": "Technology",
    "FB": "Technology",
    "TSLA": "Automotive",
    "JPM": "Finance",
    "JNJ": "Pharma",
    "V": "Finance",
    "NVDA": "Semiconductor",
    "BAC": "Finance",
    "PG": "Consumption",
    "HD": "Retail",
    "UNH": "Healthcare",
    "MA": "Finance",
    "PYPL": "Finance",
    "INTC": "Semiconductor",
    "CMCSA": "Media and Entertainment",
    "VZ": "Telecommunication",
    "DIS": "Entertainment",
    "ADBE": "Software",
    "NFLX": "Entertainment",
    "CRM": "Software",
    "ABT": "Medical Technology",
    "XOM": "Energy",
    "CSCO": "Technology",
    "PEP": "Consumption",
    "KO": "Consumption",
    "MRNA": "Pharma",
    "WMT": "Retail",
    "MRK": "Pharma",
    "MDT": "Medical Technology",
    "NKE": "Consumption",
    "ABBV": "Pharma",
    "TXN": "Semiconductor",
    "BMY": "Pharma",
    "T": "Telecommunication",
    "GILD": "Pharma",
    "IBM": "Technology",
    "CVX": "Energy",
    "MDLZ": "Consumption",
    "LMT": "Defense",
    "UPS": "Logistics",
    "MO": "Consumption",
    "INTU": "Finance",
    "FDX": "Logistics",
    "AMGN": "Biotechnology",
    "PM": "Consumption",
    "MMM": "Industry",
    "DUK": "Energy"
}

eu_business_fields = {
    "AGS.BR": "Finance",
    "UNA.AS": "Consumption",
    "AZN.L": "Pharma",
    "AIR.PA": "Aviation",
    "MCRO.L": "Technology",
    "CSGN.SW": "Finance",
    "ENGI.PA": "Energy",
    "NOVN.SW": "Pharma",
    "SOLB.BR": "Chemistry",
    "CS.PA": "Finance",
    "RDSA.AS": "Energy",
    "DG.PA": "Construction",
    "SIE.DE": "Industry",
    "TTE.PA": "Defense",
    "SU.PA": "Energy",
    "SAN.PA": "Pharma",
    "OR.PA": "Consumption",
    "SAF.PA": "Aerospace",
    "UMI.BR": "Industry",
    "BP.L": "Energy",
    "HEIA.AS": "Consumption",
    "ISP.MI": "Finance",
    "ROG.SW": "Pharma",
    "ENEL.MI": "Energy",
    "FP.PA": "Energy",
    "INGA.AS": "Finance",
    "MC.PA": "Luxury Goods",
    "ALO.PA": "Transportation",
    "GSK.L": "Pharma",
    "VIV.PA": "Media",
    "SAP.DE": "Software",
    "IBE.MC": "Energy",
    "BNP.PA": "Finance",
    "DSM.AS": "Chemistry",
    "RDSB.L": "Energy",
    "AI.PA": "Chemistry",
    "ABI.BR": "Consumption",
    "HSBA.L": "Finance",
    "TEP.L": "Energy",
    "AED.BR": "Real Estate",
    "ASML.AS": "Technology",
    "ZURN.SW": "Insurance",
    "NESN.SW": "Consumption"
}


indice_symbols = [
    '^IRX',         # Euro interestrate
    '^TNX',         # Dollar interestrate
    'DAX',          # DAX
    '^STOXX50E',    # EURO STOXX 50
    '^GSPC',        # &P500
    'XWD.TO',       # MSCI World
]

company_country_codes = {
    "ADS.DE": "DE",
    "AIR.DE": "DE",
    "ALV.DE": "DE",
    "BAS.DE": "DE",
    "BAYN.DE": "DE",
    "BEI.DE": "DE",
    "BMW.DE": "DE",
    "BNR.DE": "DE",
    "CBK.DE": "DE",
    "CON.DE": "DE",
    "1COV.DE": "DE",
    "DTG.DE": "DE",
    "DBK.DE": "DE",
    "DB1.DE": "DE",
    "DHL.DE": "DE",
    "DTE.DE": "DE",
    "EOAN.DE": "DE",
    "FRE.DE": "DE",
    "HNR1.DE": "DE",
    "HEI.DE": "DE",
    "HEN3.DE": "DE",
    "IFX.DE": "DE",
    "MBG.DE": "DE",
    "MRK.DE": "DE",
    "MTX.DE": "DE",
    "MUV2.DE": "DE",
    "P911.DE": "DE",
    "PAH3.DE": "DE",
    "QIA.DE": "DE",
    "RHM.DE": "DE",
    "RWE.DE": "DE",
    "SRT3.DE": "DE",
    "SIE.DE": "DE",
    "ENR.DE": "DE",
    "SHL.DE": "DE",
    "SY1.DE": "DE",
    "VOW3.DE": "DE",
    "VNA.DE": "DE",
    "ZAL.DE": "DE",
    "AAPL": "US",
    "MSFT": "US",
    "AMZN": "US",
    "GOOGL": "US",
    "FB": "US",
    "TSLA": "US",
    "JPM": "US",
    "JNJ": "US",
    "V": "US",
    "NVDA": "US",
    "BAC": "US",
    "PG": "US",
    "HD": "US",
    "UNH": "US",
    "MA": "US",
    "PYPL": "US",
    "INTC": "US",
    "CMCSA": "US",
    "VZ": "US",
    "DIS": "US",
    "ADBE": "US",
    "NFLX": "US",
    "CRM": "US",
    "ABT": "US",
    "XOM": "US",
    "CSCO": "US",
    "PEP": "US",
    "KO": "US",
    "MRNA": "US",
    "WMT": "US",
    "MDT": "US",
    "NKE": "US",
    "ABBV": "US",
    "TXN": "US",
    "BMY": "US",
    "T": "US",
    "GILD": "US",
    "IBM": "US",
    "CVX": "US",
    "MDLZ": "US",
    "LMT": "US",
    "UPS": "US",
    "MO": "US",
    "INTU": "US",
    "FDX": "US",
    "AMGN": "US",
    "PM": "US",
    "MMM": "US",
    "DUK": "US",
    "MRK": "US",
    "AGS.BR": "BE",
    "UNA.AS": "NL",
    "AZN.L": "GB",
    "AIR.PA": "FR",
    "MCRO.L": "GB",
    "CSGN.SW": "CH",
    "ENGI.PA": "FR",
    "NOVN.SW": "CH",
    "SOLB.BR": "BE",
    "CS.PA": "FR",
    "RDSA.AS": "NL",
    "DG.PA": "FR",
    "SIE.DE": "DE",
    "TTE.PA": "FR",
    "SU.PA": "FR",
    "SAN.PA": "FR",
    "OR.PA": "FR",
    "SAF.PA": "FR",
    "UMI.BR": "BE",
    "BP.L": "GB",
    "HEIA.AS": "NL",
    "ISP.MI": "IT",
    "ROG.SW": "CH",
    "ENEL.MI": "IT",
    "FP.PA": "FR",
    "INGA.AS": "NL",
    "MC.PA": "FR",
    "ALO.PA": "FR",
    "GSK.L": "GB",
    "VIV.PA": "FR",
    "SAP.DE": "DE",
    "IBE.MC": "ES",
    "BNP.PA": "FR",
    "DSM.AS": "NL",
    "RDSB.L": "GB",
    "AI.PA": "FR",
    "ABI.BR": "BE",
    "HSBA.L": "GB",
    "TEP.L": "GB",
    "AED.BR": "BE",
    "ASML.AS": "NL",
    "ZURN.SW": "CH",
    "NESN.SW": "CH"
}







symbol_lists = [dax_symbols, us_symbols, eu_symbols]


for symbol_list in symbol_lists:
    for symbol in symbol_list:
        print("\n#############################################################################################")
        print("#############################################################################################")
        print("#############################################################################################")
        print("#############################################################################################\n")
        
        print("Verarbeite Symbol:", symbol)

        
        data = download_stock_data(symbol, start_date, end_date)
        if data is not None:
            df = pd.DataFrame(data)
            df = df.drop(['Adj Close', 'Open'], axis=1)
            df = fill_missing_dates(df)
            df['Date'] = df.index
            df['Date'] = pd.to_datetime(df['Date'])
            df['Day_of_the_year'] = df['Date'].dt.dayofyear
            
            if symbol in dax_symbols:
                geo = 'DE'
            elif symbol in us_symbols:
                geo = 'US'
            else:
                geo = company_country_codes[symbol]
            df = merge_with_google_trends(df, symbol, start_date, end_date, geo, 'trends_lokal')
            geo = ''
            df = merge_with_google_trends(df, symbol, start_date, end_date, geo, 'trends_global')

            
            path = base_path + symbol + ".csv"
            save_to_csv(df, path)
        else:
            print("\nFehler beim Herunterladen der Daten für", symbol)

indice_df = combine_indices_data(indice_symbols, start_date, end_date)

if indice_df is not None:
  save_to_csv(indice_df, base_path + 'indices.csv')


all_labels = set(dax_business_fields.values()).union(set(us_business_fields.values()), set(eu_business_fields.values()))

# Sort the labels alphabetically
alphabetical_labels = sorted(all_labels)

# Output the alphabet and its length
print("Alphabet:", alphabetical_labels)
print("Length of Alphabet:", len(alphabetical_labels))

dax_df = pd.DataFrame(list(dax_business_fields.items()), columns=['Symbol', 'Industry'])
dax_df['Region'] = 'DE'

us_df = pd.DataFrame(list(us_business_fields.items()), columns=['Symbol', 'Industry'])
us_df['Region'] = 'US'

eu_df = pd.DataFrame(list(eu_business_fields.items()), columns=['Symbol', 'Industry'])
eu_df['Region'] = 'EU'

# Concatenate all DataFrames
combined_df = pd.concat([dax_df, us_df, eu_df])

for index, row in combined_df.iterrows():
    if row['Region'] == 'EU':
        combined_df.at[index, 'Region'] = company_country_codes[row['Symbol']]
combined_df.reset_index(drop=True, inplace=True)
print("\n")
path = base_path + 'buisness_field_&_origin' + ".csv"
save_to_csv(combined_df, path)
