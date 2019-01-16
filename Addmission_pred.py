# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:54:58 2019

@author: Suvhradip Ghosh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("Admission_Predict.csv")
x=df.iloc[: ,1:-1].values
y=df.iloc[: , 8].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print("Mean squared error: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))
print('Variance score: %.2f' % regressor.score(X_test, y_test))
import pickle
with open('model_pickle', 'wb') as d:
    pickle.dump(regressor, d)
    
with open('model_pickle', 'rb') as d:
    model =pickle.load(d)


