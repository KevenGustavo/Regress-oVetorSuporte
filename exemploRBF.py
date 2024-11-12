import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


dataset = pd.read_csv('./Position_Salaries.csv')
X = dataset.iloc[:, 1].values.reshape(-1,1)
y = dataset.iloc[:, -1].values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(X,y)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Regressao SVR')
plt.xlabel('Nivel')
plt.ylabel('Salario')
plt.show()

sc_y.inverse_transform(regressor.predict(sc_X.transform([[7.5]])))