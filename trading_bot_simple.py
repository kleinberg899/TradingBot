import pandas as pd
from pandas import read_csv
from scipy import stats
from sklearn.linear_model import LinearRegression
import warnings
import torch

# Warnungen unterdrücken
warnings.filterwarnings("ignore", category=UserWarning)

from Market_Environment import Market_Environment
class Bot_LinearRegression():

    def __init__(self, path_to_data_folder, stock_list, environment):
        self.data_path = path_to_data_folder
        self.stocks = stock_list

        self.environment = environment

    def __call__(self, current_date, ):

        output = []

        prediction_date_buy = current_date + pd.offsets.DateOffset(years=1)
        prediction_date_sell = current_date + pd.offsets.DateOffset(days=31)
        context_size = 352

        current_prices_buy = []
        stocks_buy = []
        predictions_buy = []
        yield_predictions_buy = []

        current_prices_sell = []
        stocks_sell = []
        predictions_sell = []
        yield_predictions_sell = []

        for stock in self.stocks:
            stock_data = self.environment.get_price_df(stock).copy()

            current_price = stock_data['Close'].iloc[-1]

            start_date = current_date + pd.Timedelta(days=-context_size)
            stock_data = stock_data[(stock_data['Date'] >= start_date) & (stock_data['Date'] <= current_date)]
            num_rows = stock_data.shape[0]
            if num_rows <= 0:
                continue
            prediction_buy = self.linear_regression(stock_data, start_date, prediction_date_buy)
            if prediction_buy is not None:
                stocks_buy.append(stock)
                current_prices_buy.append(current_price)
                predictions_buy.append(prediction_buy)
                yield_predictions_buy.append((prediction_buy - current_price) / current_price)

            prediction_sell = self.linear_regression(stock_data, start_date, prediction_date_sell)
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

        #Buy Behaviour

        spending = self.environment.get_my_balance(self) * 0.05

        sum_of_best_yields = buy_df['Yield_Prediction'].head(5).sum()

        for i in range(10):
            yield_i = buy_df.iloc[i]['Yield_Prediction']
            if yield_i >= 0.4:
                stock = buy_df.iloc[i]['Stock']
                price = buy_df.iloc[i]['Current']
                randomfactor = torch.randint(7,13,(1,)).item() / 5
                amount = ((yield_i/sum_of_best_yields) * randomfactor * spending)/price
                amount = int(amount)
                if amount > 0:
                    output.append(['buy', stock, amount])
        return output


    def linear_regression(self, stock_data, start_date, prediction_date):
        reference_date = pd.Timestamp(start_date)
        prediction_date = pd.Timestamp(prediction_date)
        stock_data.loc[:, 'Days'] = (stock_data['Date'] - reference_date).dt.days
        model = LinearRegression()
        model.fit(stock_data[['Days']], stock_data['Close'])
        days_difference = (prediction_date - reference_date).days
        predictions = model.predict([[days_difference]])
        return predictions[0]



    def plot_prediction(self, stock, start_date, end_date, prediction_date):
        ...  # soll plt erstellen
