# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31  16:22:12 2019

@author: Ram Kumar R P
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("heart.csv")


df.shape
df.dtypes
df.head()
df.describe()
df.isnull().sum() 

df.target.value_counts()



sns.countplot(x="target",data=df)
plt.show()

disdf = df[df['target']==1]

sns.countplot(x='sex', data=disdf)
plt.show()


df.groupby('target').mean()

pd.crosstab(disdf.age,disdf.target).plot(kind="bar",figsize=(20,6))
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()



pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['red','blue' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df.head()


df = df.drop(columns = ['cp', 'thal', 'slope'])
df.head()


y = df.target.values
x_data = df.drop(['target'], axis = 1)



#x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values


x_train, x_test, y_train, y_test = train_test_split(x_data,y,test_size = 0.2,random_state=0)


accuracies = {}
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
acc = lr.score(x_test,y_test)*100

accuracies['Logistic Regression'] = acc
print(" Accuracy of Logistic Regression {:.2f}%".format(acc))

y_pred = lr.predict(x_test)
confusion_matrix(y_test, y_pred)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train, y_train)

acc = nb.score(x_test,y_test)*100
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))
y_pred = nb.predict(x_test)
confusion_matrix(y_test, y_pred)


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train, y_train)

acc = rf.score(x_test,y_test)*100
accuracies['Random Forest'] = acc
print("Accuracy of Random Forest: {:.2f}%".format(acc))
y_pred = rf.predict(x_test)
confusion_matrix(y_test, y_pred)

