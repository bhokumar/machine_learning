import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


linearRegressor = LinearRegression()
linearRegressor.fit(x, y)


from sklearn.preprocessing import PolynomialFeatures
polyRegressor = PolynomialFeatures(degree=4)
x_poly = polyRegressor.fit_transform(x)

poly_linearRegressor = LinearRegression()

poly_linearRegressor.fit(x_poly, y)



# plt.scatter(x, y, color='red')
# plt.plot(x, poly_linearRegressor.predict(x_poly), color='blue')
# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')
# plt.show()




# x_grid = np.arange(min(x), max(x), 0.1)
# x_grid = x_grid.reshape(len(x_grid), 1)
# plt.scatter(x, y, color='red')
# plt.plot(x_grid, poly_linearRegressor.predict(polyRegressor.fit_transform(x_grid)), color='blue')

# plt.title('Truth or Bluff (Polynomial Regression)')
# plt.xlabel('Position Level')
# plt.ylabel('Salary')

# plt.show()

# Predicting a new result with Linear Regression
print(linearRegressor.predict([[6.5]]))
print(poly_linearRegressor.predict(polyRegressor.fit_transform([[6.5]])))







