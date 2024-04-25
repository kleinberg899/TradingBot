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

    stocks = []
    predictions = []
    targets = []
    currents = []
    yield_predictions = []
    yield_targets = []

    
    for stock in env.stocks:

        # price prediction
        prediction = env.bot(stock, start_date, end_date, prediction_date)

        if prediction is not None:
            
            stocks.append(stock)
            predictions.append(prediction)

            # lookup of the correct price in data
            targets.append(env.get_price(stock, prediction_date))

            # compute relative price increase/decrease
            currents.append(env.get_price(stock, end_date))
            yield_predictions.append((prediction - currents[-1]) / currents[-1])
            yield_targets.append((targets[-1] - currents[-1]) / currents[-1])

    # sort all lists by largests predicted yields
    sorted_indices = sorted(range(len(yield_predictions)), key=lambda k: yield_predictions[k], reverse=True)

    sorted_stocks = [stocks[i] for i in sorted_indices]
    sorted_predictions = [predictions[i] for i in sorted_indices]
    sorted_targets = [targets[i] for i in sorted_indices]
    sorted_currents = [currents[i] for i in sorted_indices]
    sorted_yield_predictions = [yield_predictions[i] for i in sorted_indices]
    sorted_yield_targets = [yield_targets[i] for i in sorted_indices]


    result_df = pd.Dataframe({
        'Stock': sorted_stocks,
        'Prediction': sorted_predictions,
        'Target': sorted_targets,
        'Difference': [target - current for target, current in zip(sorted_targets, sorted_currents)],
        'Yield_Prediction': sorted_yield_predictions,
        'Yield_Target': sorted_yield_targets,
        'Yield_Difference': [yield_target - yield_prediction for yield_target, yield_prediction in zip(sorted_yield_targets, sorted_yield_predictions)]
    })


    print(result_df)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
