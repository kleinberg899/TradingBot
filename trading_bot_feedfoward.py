import pandas as pd
from pandas import read_csv
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError

# Warnungen unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)

from Market_Environment import Market_Environment


def normalize_prices(data_tensor, col_position_of_target=2):
    x = data_tensor.clone()
    price_normalisation = x[-1, col_position_of_target]
    if torch.any(price_normalisation == 0):
        print("TEILEN DURCH 0 DU HUND. BEHANDEL EXCEPTION")
        return None
    data_tensor[:, :3] = data_tensor[:, :3] / price_normalisation
    return data_tensor, price_normalisation.item()
class Bot_FeedForward:

    def __init__(self, path_to_data_folder, stock_list, environment, context_size=356, name='Feedforward_Bot'):
        self.data_path = path_to_data_folder
        self.stocks = stock_list
        self.environment = environment
        self.name = name

        self.context_size = context_size
        self.dist_target_from_context = 7
        self.input_size = 7 * self.context_size
        self.col_position_of_target = 2
        self.columns_to_drop = ['Date']
        self.input_size = 7 * context_size

        from FeedforwardNN import Model

        self.feedforward = Model(self.input_size)

        checkpoint = torch.load('models/model_feed_forward.pth')

        self.feedforward.load_state_dict(checkpoint['model_state_dict'])


    def __call__(self, current_date, ):

        output = []

        consider_buy = []
        consider_sell = []

        for stock in self.stocks:
            stock_df = self.environment.get_stock_data_df(stock, self.context_size - 1).copy()
            if stock_df is None or stock_df.shape[0] <= 0:
                continue
            if self.environment.get_price(stock, current_date) == 0:
                continue

            stock_df['Volume'] = stock_df['Volume'] / 10000.
            stock_df = stock_df.drop(self.columns_to_drop, axis=1)
            data_tensor = torch.tensor(stock_df.values[:, 1:].astype(float), dtype=torch.float32)
            data_tensor, norm_divisor = normalize_prices(data_tensor)
            prediction = self.feedforward(data_tensor.contiguous().view(-1)).item()
            if prediction >= 1.15:
                consider_buy.append((stock, prediction))
            if prediction <= 1.01:
                consider_sell.append((stock, prediction))
        consider_buy = sorted(consider_buy, key=lambda x: x[1], reverse=True)
        consider_sell = sorted(consider_sell, key=lambda x: x[1])

        #
        # buy behaviour
        spending = self.environment.get_my_balance(self) * 0.05
        num_stocks = 10
        if len(consider_buy) < 10:
            num_stocks = len(consider_buy)
        sum_of_yields = 0
        for i in range(num_stocks):
            sum_of_yields += (consider_buy[i])[1] - 1
        for i in range(num_stocks):
            stock = consider_buy[i][0]
            weight = (((consider_buy[i])[1] - 1) / sum_of_yields)
            rnd_factor = torch.randint(5, 15, (1,)).item() / 10
            price = self.environment.get_price(stock, current_date)
            amount = int((weight * rnd_factor * spending) / price)

            if amount > 0:
                output.append(['buy', stock, amount])

        # sell behaviour

        portfolio = self.environment.get_my_portfolio(self)

        for stock_est in consider_sell:
            if portfolio[stock_est[0]] == 0:
                continue
            else:
                amount = portfolio[stock_est[0]]
                output.append(['sell', stock_est[0], amount])
        return output
