import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import Bot_LinearRegression as Bot

test_data = pd.read_csv('data/stock_data/DTE.DE.csv', sep=',')
test_data['Date'] = pd.to_datetime(test_data['Date'])

reference_date = pd.Timestamp('2000-01-01')
test_data['Days'] = (test_data['Date'] - reference_date).dt.days
model = LinearRegression()
model.fit(test_data[['Days']], test_data['Close'])
# Vorhersage
predictions = model.predict(test_data[['Days']])



plt.figure(figsize=(10,6))
plt.scatter(test_data['Days'], test_data['Close'], label='Data Points')
plt.plot(test_data['Days'], predictions, color='red', label='Linear Regression')
plt.xlabel('Days since reference date')
plt.ylabel('Close Price')
plt.title('Linear Regression of Close Price over Time')
plt.legend()
plt.grid(True)
plt.show()

