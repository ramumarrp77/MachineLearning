# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 22:56:49 2019

@author: Ram Kumar R P
"""

import numpy as np
import pandas as pd


df= pd.read_csv("amazon.csv")
df = df.iloc[:,1:5]
df.head(5)

df.shape

df.dtypes

df.describe()

print(df.state.unique())

print(len(df.state.unique()))

df.isnull().sum()


'''Grouping by Years and seeing and plotting numbers'''

year = df.groupby('year')
year.get_group(2006)
len(year.get_group(2006))
len(year.get_group(1999))


import matplotlib.pyplot as plt
df1=year.number.mean().reset_index()
df1=df1.sort_values('number', ascending=False)
print(df1)
df1.plot(x='year', y='number', kind = 'line')
plt.show()


'''Grouping by months and plotting numbers '''
month = df.groupby('month')
month.get_group('Janeiro')
len(month.get_group('Julho'))
len(month.get_group('Dezembro'))




df2=month.number.mean().reset_index()
df2=df2.sort_values('number', ascending=False)
print(df2)
df2.plot(x='month', y='number', kind = 'bar')
plt.show()


'''Grouping by states and plotting numbers '''
state = df.groupby('state')
print(df.state.unique())

state.get_group('Acre')
len(state.get_group('Acre'))
len(state.get_group('Sergipe'))




df3=state.number.mean().reset_index()
df3=df3.sort_values('number', ascending=False)
print(df3)
df3.plot(x='state', y='number', kind = 'bar')
plt.show()



state_int = {'Acre':1,'Alagoas':2,'Amapa':3,'Amazonas':3,'Bahia':4 ,'Ceara':5 ,'Distrito Federal':6,
 'Espirito Santo':7, 'Goias':8 ,'Maranhao':9, 'Mato Grosso':10, 'Minas Gerais':11 ,'Pará':12,
 'Paraiba':13 ,'Pernambuco':14, 'Piau':15, 'Rio':16, 'Rondonia':17 ,'Roraima':18 ,'Santa Catarina':19,
 'Sao Paulo':21 ,'Sergipe':22, 'Tocantins':23}

month_int = {'Janeiro': 1, 'Fevereiro': 2, 'Março': 3, 'Abril': 4,'Maio': 5, 'Junho': 6, 'Julho': 7, 'Agosto': 8, 'Setembro': 9, 'Outubro': 10, 'Novembro': 11, 'Dezembro':12}

df['State'] = df['state'].map(state_int).astype(str)

df['Month'] = df['month'].map(month_int).astype(str)

df=df.drop(['month','state'], axis = 1)
print(df.head())



from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn import metrics

X = df.iloc[:,[0,2,3]]
#x = (X - np.min(X)) / (np.max(X) - np.min(X)).values
y=df.iloc[:,1]
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
reg= linear_model.LinearRegression()



reg.fit(X_train,y_train)

#To retrieve the intercept:
print(reg.intercept_)
#For retrieving the slope:
print(reg.coef_)
y_pred = reg.predict(X_test)
df4= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 100,random_state=0)

regressor.fit(X_train,y_train)

y_pred1 = regressor.predict(X_test)
df5= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred1))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred1)))