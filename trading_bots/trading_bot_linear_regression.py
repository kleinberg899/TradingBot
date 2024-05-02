import pandas as pd
from pandas import read_csv
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# Warnungen unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)

from Market_Environment import Market_Environment


class Bot_LinearRegression():

    def __init__(self, path_to_data_folder, stock_list, environment, degree = 1, name = 'Linear_Reg_Bot'):
        self.data_path = path_to_data_folder
        self.stocks = stock_list
        self.degree = degree
        self.environment = environment
        self.name = name

    def __call__(self, current_date, ):

        output = []

        prediction_date_buy = current_date + pd.offsets.DateOffset(years=1)
        prediction_date_sell = current_date + pd.offsets.DateOffset(days=31)
        context_size = 356

        current_prices_buy = []
        stocks_buy = []
        predictions_buy = []
        yield_predictions_buy = []

        current_prices_sell = []
        stocks_sell = []
        predictions_sell = []
        yield_predictions_sell = []

        for stock in self.stocks:

            stock_df = self.environment.get_stock_data_df(stock, context_size).copy()

            if stock_df is None or stock_df.shape[0] <= 0:
                continue

            current_price = self.environment.get_price(stock, current_date)
            if current_price == 0:
                continue

            prediction_buy = self.linear_regression_poly(stock_df, current_date, context_size, prediction_date_buy, self.degree)

            if prediction_buy is not None:
                stocks_buy.append(stock)
                current_prices_buy.append(current_price)
                predictions_buy.append(prediction_buy)
                yield_predictions_buy.append((prediction_buy - current_price) / current_price)

            prediction_sell = self.linear_regression_poly(stock_df, current_date, context_size, prediction_date_sell, self.degree)
            if prediction_sell is not None:
                stocks_sell.append(stock)
                current_prices_sell.append(current_price)
                predictions_sell.append(prediction_sell)
                yield_predictions_sell.append((prediction_sell - current_price) / current_price)

        # sort all lists by largests predicted yields
        sorted_indices = sorted(range(len(yield_predictions_buy)), key=lambda k: yield_predictions_buy[k],
                                reverse=True)

        sorted_stocks_buy = [stocks_buy[i] for i in sorted_indices]
        sorted_predictions_buy = [predictions_buy[i] for i in sorted_indices]
        sorted_currents_buy = [current_prices_buy[i] for i in sorted_indices]
        sorted_yield_predictions_buy = [yield_predictions_buy[i] for i in sorted_indices]

        buy_df = pd.DataFrame({
            'Stock': sorted_stocks_buy,
            'Prediction': sorted_predictions_buy,
            'Current': sorted_currents_buy,
            'Yield_Prediction': sorted_yield_predictions_buy
        })

        sorted_indices = sorted(range(len(yield_predictions_sell)), key=lambda k: yield_predictions_sell[k],
                                reverse=True)

        sorted_stocks_sell = [stocks_sell[i] for i in sorted_indices]
        sorted_predictions_sell = [predictions_sell[i] for i in sorted_indices]
        sorted_currents_sell = [current_prices_sell[i] for i in sorted_indices]
        sorted_yield_predictions_sell = [yield_predictions_sell[i] for i in sorted_indices]

        sell_df = pd.DataFrame({
            'Stock': sorted_stocks_sell,
            'Prediction': sorted_predictions_sell,
            'Current': sorted_currents_sell,
            'Yield_Prediction': sorted_yield_predictions_sell
        })

        portfolio = self.environment.get_my_portfolio(self)

        for stock in self.stocks:
            if portfolio[stock] == 0:
                continue
            elif sell_df.shape[0] == 0:
                print(stock, "uuuff")
            else:
                prediction_for_stock = sell_df.loc[sell_df['Stock'] == stock, 'Yield_Prediction'].values[0]
                if prediction_for_stock <= 0.00:
                    output.append(['sell', stock, portfolio[stock]])

        # Buy Behaviour

        spending = self.environment.get_my_balance(self) * 0.05

        sum_of_best_yields = buy_df['Yield_Prediction'].head(5).sum()

        for i in range(10):
            yield_i = buy_df.iloc[i]['Yield_Prediction']
            if yield_i >= 0.2:
                stock = buy_df.iloc[i]['Stock']
                price = buy_df.iloc[i]['Current']
                randomfactor = torch.randint(5, 15, (1,)).item() / 10
                amount = ((yield_i / sum_of_best_yields) * randomfactor * spending) / price
                amount = int(amount)
                if amount > 0:
                    output.append(['buy', stock, amount])
        return output

    def linear_regression_poly(self, data, current_date, context_size, prediction_date, degree=1):

        start_date = current_date - pd.Timedelta(days=context_size)
        prediction_date = pd.Timestamp(prediction_date)

        difference_in_days_prediction = (prediction_date - start_date).days

        Y = data['Close'].values
        X = np.arange(1, data.shape[0] + 1)
        X = X.reshape(-1, 1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        poly_features = PolynomialFeatures(degree)
        X_poly = poly_features.fit_transform(X)

        # Lineare Regression anpassen
        model = LinearRegression()
        model.fit(X_poly, Y)
        prediction_day = np.array([difference_in_days_prediction]).reshape(-1, 1)
        prediction_day_poly = poly_features.fit_transform(prediction_day)
        predictions = model.predict(prediction_day_poly)

        return predictions[0]

    def plot_prediction(self, stock, start_date, end_date, prediction_date):
        ...  # soll plt erstellen
