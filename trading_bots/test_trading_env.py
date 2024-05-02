from trading_bot_linear_regression import Bot_LinearRegression
from Market_Environment import Market_Environment
from trading_bot_feedfoward import Bot_FeedForward
stocks = [
            "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE",# "DTG.DE",
            # "CON.DE", "P911.DE","SHL.DE",
            "1COV.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
            "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
            "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
        ]
path = '../data/stock_data'


def date_to_string(year, month, day):
    if month >= 10 and day >= 10:
        return str(year) + '-' + str(month) + '-' + str(day)
    elif month < 10 and day >= 10:
        return str(year) + '-0' + str(month) + '-' + str(day)
    elif month >= 10 and day < 10:
        return str(year) + '-' + str(month) + '-0' + str(day)
    elif month < 10 and day < 10:
        return str(year) + '-0' + str(month) + '-0' + str(day)

def print_bot_state(bot, env, show_portfolio = False):
    portfolio = env.portfolios[bot].get_portfolio()

    portfolio_value = env.calculate_portfolio_value(bot)

    list = []
    for aktie, menge in portfolio.items():
        if menge != 0:
            list.append((aktie, menge))

    print(f"{bot.name}, Portfolio Value: {env.calculate_portfolio_value(bot):.2f}, Balance: {env.get_my_balance(bot):.2f}")

    if show_portfolio:
        print(f"Balance: {env.get_my_balance(bot):.2f}, {list}\n")
def main():
    print("Init:")
    trading_market = Market_Environment(path, 2022, 2, 4, 2024,4,15)
    #bot1 = Bot_LinearRegression(path, stocks, trading_market)
    #trading_market.create_agent(bot1, 100_000)

    bot2 = Bot_FeedForward(path, stocks, trading_market)
    trading_market.create_agent(bot2, 100_000)

    bot3 = Bot_LinearRegression(path, stocks, trading_market, name = 'Bot_POLY_5', degree= 5)
    trading_market.create_agent(bot3, 100_000)

    #bot4 = Bot_LinearRegression(path, stocks, trading_market, name='Bot_POLY_15', degree=15)
    #trading_market.create_agent(bot4, 100_000)


    print("\n\n")
    print("----------")
    print("Start:")
    print("----------")
    for bot in trading_market.agents:
        print_bot_state(bot, trading_market)
    for i in range(900):
        print(i, trading_market.simulated_date)
        trading_market.simulate_step()
        for bot in trading_market.agents:
            if True:
                print_bot_state(bot, trading_market, show_portfolio=True)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
