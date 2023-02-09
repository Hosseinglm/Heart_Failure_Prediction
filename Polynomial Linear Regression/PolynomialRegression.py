#Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns

#Import Dataset
data = pd.read_csv('Position_Salaries.csv')
data
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Poly Reg
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualization associated with Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Visualization of Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#Visualization associated with Polynomial Linear Regression
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg_2.predict(X_poly) , color='blue')
plt.title('Visualization of Polynomial Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()