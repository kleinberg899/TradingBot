import numpy as np
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def linear_regression_poly(data, prediction_date, degree):
    prediction_date = pd.Timestamp(prediction_date)

    first_date = data.iloc[0]['Date']
    difference_in_days = (prediction_date - first_date).days

    Y = data['Close'].values

    X = np.arange(1, data.shape[0] + 1)
    X = X.reshape(-1, 1)

    # Normalisiere die Eingangsdaten
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Erzeuge polynomiale Merkmale
    poly_features = PolynomialFeatures(degree)
    X_poly = poly_features.fit_transform(X_normalized)

    # Lineare Regression anpassen
    model = LinearRegression()
    model.fit(X_poly, Y)

    # Vorhersage machen
    prediction_day = np.array([difference_in_days]).reshape(-1, 1)
    prediction_day_normalized = scaler.transform(prediction_day)
    prediction_day_poly = poly_features.transform(prediction_day_normalized)
    predictions = model.predict(prediction_day_poly)

    return predictions[0]

def load_stock_data(stock):
    data = read_csv('data/stock_data' + '/' + stock + '.csv')
    df = pd.DataFrame(data)
    return df


start_year, start_month, start_day = 2017, 3, 1
start_date = pd.Timestamp(start_year, start_month, start_day)

prediction_days = 150

days_data = 356 * 6

end_date = start_date + pd.offsets.DateOffset(days_data - prediction_days)
end_date_plot = start_date + pd.offsets.DateOffset(days_data)

prices_BMW = load_stock_data('BMW.DE')
prices_BMW['Date'] = pd.to_datetime(prices_BMW['Date'])  # Convert 'Date' column to timestamp format
prices_BMW = prices_BMW[['Date', 'Close']]
prices_BMW_train = prices_BMW[(prices_BMW['Date'] >= start_date) & (prices_BMW['Date'] <= end_date)]
prices_BMW_plot = prices_BMW[(prices_BMW['Date'] >= start_date) & (prices_BMW['Date'] <= end_date_plot)]
prediction_poly = []
prediction_poly2 = []
prediction_linear = []
real_value = []
indices = []

for i in range(days_data + 1):
    prediction_date = start_date + pd.Timedelta(days=i)
    real_value.append(prices_BMW.iloc[i]['Close'])
    prediction_linear.append(linear_regression_poly(prices_BMW, prediction_date, degree=1))
    prediction_poly.append(linear_regression_poly(prices_BMW, prediction_date, degree=5))
    prediction_poly2.append(linear_regression_poly(prices_BMW, prediction_date, degree=20))


print(prices_BMW_plot.shape[0])
print(len(real_value))
print(len(prediction_linear))
print(len(prediction_poly))
print(prices_BMW.iloc[0]['Date'])
print(prices_BMW.iloc[-1]['Date'])

# Zeitachse für die Vorhersagen und echten Aktienkurse
dates = prices_BMW_plot['Date']

# Grafische Darstellung der Vorhersagen und echten Aktienkurse
plt.figure(figsize=(10, 6))

# Aktuelle Aktienkurse plotten
plt.plot(dates, real_value, label='Echter Aktienkurs', color='blue')

# Vorhersagen der polynomiellen Regression plotten
plt.plot(dates, prediction_poly, label='Polynomiale Regression', color='red')

plt.plot(dates, prediction_poly2, label='Polynomiale Regression', color='purple')

# Vorhersagen der linearen Regression plotten
plt.plot(dates, prediction_linear, label='Lineare Regression', color='green')

# Vertikale Linie an die 3. letzte Position der x-Achse zeichnen
plt.axvline(x=dates.iloc[-prediction_days], linestyle='--', color='black', label='Vertikale Linie')

plt.title('Vorhersage der Aktienkurse')
plt.xlabel('Datum')
plt.ylabel('Schlusskurs')
plt.legend()
plt.grid(True)
plt.show()