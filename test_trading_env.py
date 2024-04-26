import datetime
import Simple_Environment as Env
from trading_bot_simple import Bot_LinearRegression
import pandas as pd
from Market_Environment import Market_Environment

stocks = [
            "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE",  # "CON.DE","P911.DE",
            "1COV.DE", "DTG.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
            "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
            "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SHL.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
        ]
path = 'data/stock_data'


def date_to_string(year, month, day):
    if month >= 10 and day >= 10:
        return str(year) + '-' + str(month) + '-' + str(day)
    elif month < 10 and day >= 10:
        return str(year) + '-0' + str(month) + '-' + str(day)
    elif month >= 10 and day < 10:
        return str(year) + '-' + str(month) + '-0' + str(day)
    elif month < 10 and day < 10:
        return str(year) + '-0' + str(month) + '-0' + str(day)
def main():
    print("Init:")
    trading_market = Market_Environment(path, 2020, 1, 2, 2021,1,1)
    bot1 = Bot_LinearRegression(path, stocks, trading_market)
    trading_market.create_agent(bot1, 100_000)
    print("Start:")
    print("----------")
    print("Balance")
    print(trading_market.portfolios[bot1].get_balance())
    print("Portfolio")
    print("{}")
    print("Portfolio Wert")
    print(trading_market.calculate_portfolio_value(bot1))
    print("----------------------")
    for i in range(100):
        print(i, trading_market.simulated_date)
        trading_market.simulate_step()
        print("Balance")
        print(trading_market.portfolios[bot1].get_balance())
        print("Portfolio")
        portfolio = trading_market.portfolios[bot1].get_portfolio()
        list = []
        for aktie, menge in portfolio.items():
            if menge != 0:
                list.append((aktie,menge))
        print(list)
        print("Portfolio Wert")
        print(trading_market.calculate_portfolio_value(bot1))
    print("----------------------")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
