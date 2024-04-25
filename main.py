import datetime
import Environment as Env
import Bot_LinearRegression

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

    start_year, start_month, start_day = 2019, 4, 24
    end_year, end_month, end_day = 2020, 4, 24

    start_date = date_to_string(start_year, start_month, start_day)
    end_date = date_to_string(end_year, end_month, end_day)

    prediction_date = date_to_string(end_year, end_month + 1, end_day+1)
    print(prediction_date)
    env = Env.Environment('data/stock_data')

    predictions = {}
    target = {}


    for stock in env.stocks:

        # price prediction by the bot
        profit = env.bot(stock, start_date, end_date, prediction_date)
        if profit is not None:
            predictions[stock] = profit
            # lookup of the correct price in data
            target[stock] = env.get_price(stock, prediction_date)

    sorted_dict = dict(sorted(predictions.items(), key=lambda item: item[1], reverse=True))

    print("Stock, prediction, target, difference")
    for key, value in sorted_dict.items():
        print(f"{key}, {value:.4f}, {target[key]:.4f}, {(target[key]-value):.4f}")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
