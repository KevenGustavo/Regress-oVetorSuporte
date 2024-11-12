import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures


#Ler arquivo csv
df=pd.read_csv('./Position_Salaries.csv')

#df.info()
df["Position"].value_counts()
plt.scatter(df['Level'],df['Salary'])
#plt.show()

X=df.iloc[:,1].values.reshape(-1,1)
y=df.iloc[:,-1].values

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

print(X_train.shape)
print(X_test.shape)

###Linear### 

Linear=LinearRegression()
Linear.fit(X_train,y_train)
y_pred = Linear.predict(X_test)

print('The r2 score',r2_score(y_test,y_pred))
print('The rmse',np.sqrt(mean_squared_error(y_test,y_pred)))
print('The mean absolute error',mean_absolute_error(y_test,y_pred))

plt.plot(X_test,y_pred,color='red')
plt.scatter(X,y)
plt.legend()
plt.show()
