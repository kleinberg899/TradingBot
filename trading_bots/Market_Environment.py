import pandas as pd
from pandas import read_csv
from trading_bots import Portfolio


def date_to_string(year, month, day):
    if month >= 10 and day >= 10:
        return str(year) + '-' + str(month) + '-' + str(day)
    elif month < 10 and day >= 10:
        return str(year) + '-0' + str(month) + '-' + str(day)
    elif month >= 10 and day < 10:
        return str(year) + '-' + str(month) + '-0' + str(day)
    elif month < 10 and day < 10:
        return str(year) + '-0' + str(month) + '-0' + str(day)


class Market_Environment:
    def __init__(self, path_to_data_folder, start_year, start_month, start_day, end_year, end_month, end_day):
        self.data_path = path_to_data_folder
        self.stocks = [
            "ADS.DE", "AIR.DE", "ALV.DE", "BAS.DE", "BAYN.DE", "BEI.DE", "BMW.DE", "BNR.DE", "CBK.DE",# "DTG.DE",
            # "CON.DE", "P911.DE","SHL.DE",
            "1COV.DE", "DBK.DE", "DB1.DE", "DHL.DE", "DTE.DE", "EOAN.DE", "FRE.DE", "HNR1.DE", "HEI.DE",
            "HEN3.DE", "IFX.DE", "MBG.DE", "MRK.DE", "MTX.DE", "MUV2.DE", "PAH3.DE", "QIA.DE", "RHM.DE",
            "RWE.DE", "SAP.DE", "SRT3.DE", "SIE.DE", "ENR.DE", "SY1.DE", "VOW3.DE", "VNA.DE", "ZAL.DE"
        ]
        self.agent_count = 0
        self.agents = []
        self.portfolios = {}

        self.dict_agent_id = {}

        self.start_date = pd.Timestamp(start_year, start_month, start_day)
        self.end_date = pd.Timestamp(end_year, end_month, end_day)
        self.simulated_date = pd.Timestamp(start_year, start_month, start_day)

        self.prices = {}
        for stock in self.stocks:
            self.prices[stock] = self.load_stock_data(stock)


    def get_data_df(self, stock):
        return self.prices[stock]

    def increment_date_day(self):
        # Über pd.Timedelta einen Tag hinzufügen
        self.simulated_date = self.simulated_date + pd.Timedelta(days=1)
        return self.simulated_date

    def get_simulated_date(self):
        return self.simulated_date

    def get_price(self, stock, date):
        # date = pd.Timestamp(date)
        stock_data = self.get_data_df(stock)
        num_rows = stock_data.shape[0]
        if num_rows <= 0:
            return None
        stock_data.rename(columns={stock_data.columns[0]: 'Date'}, inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        close_value = stock_data.loc[stock_data['Date'] == date, 'Close'].values
        if close_value.shape[0] > 0:
            return close_value[0]
        else:
            return 0

    def get_stock_data_df(self, stock, context_size):
        stock_data = self.get_data_df(stock)
        num_rows = stock_data.shape[0]
        if num_rows <= 0:
            return None

        end_date = self.simulated_date
        start_date = end_date - pd.Timedelta(days=context_size)
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        return stock_data.loc[(stock_data['Date'] <= end_date) & (stock_data['Date'] >= start_date)]

    def create_agent(self, bot, starting_balance):
        self.agents.append(bot)
        self.agent_count += 1
        self.dict_agent_id[bot] = self.agent_count
        portfolio = Portfolio.Portfolio(starting_balance, self.stocks)
        self.portfolios[bot] = portfolio

    def simulate_step(self):
        self.increment_date_day()
        for agent in self.agents:
            moves = agent(self.simulated_date)  # expects a list of moves, each move is a list, where move[0] = "buy" or "sell", move[1] = stock, move[2] = amount

            for move in moves:
                amount = move[2]
                stock = move[1]
                price = self.get_price(stock, self.simulated_date)
                if price is None:
                    print("system failed")
                    continue

                if move[0] == 'buy':
                    if self.portfolios[agent].increase_balance(-amount * price):
                        self.portfolios[agent].increase_stock_amount(stock, amount)
                    else:
                        ...
                        # print("Agent ", self.dict_agent_id[agent], "tried to make illegal move: ")
                        # print(["buy/sell", "stock", "amount"])
                        # print(move)
                elif move[0] == 'sell':
                    if self.portfolios[agent].increase_stock_amount(stock, -amount):
                        print("SELL!!!", agent.name, move[1], move[2], price,amount * price)
                        answer = self.portfolios[agent].increase_balance(amount * price)

    def get_my_portfolio(self, bot):
        return self.portfolios[bot].get_portfolio()

    def get_my_balance(self, bot):
        return self.portfolios[bot].get_balance()

    def load_stock_data(self, stock):
        data = read_csv(self.data_path + '/' + stock + '.csv')
        df = pd.DataFrame(data)
        return df

    def calculate_portfolio_value(self, agent):
        sum = 0
        portfolio = self.portfolios[agent]
        for stock in self.stocks:
            price = self.get_price(stock, self.simulated_date)
            sum += portfolio.stock_dict[stock] * price
        return sum + portfolio.get_balance()
