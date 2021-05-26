#Here we built a 3 layer deep neural network to predict number of emergency visits for eplilepsy patients
#the model has been built using tensorflow and Keras sequential model
#same parametres have been used as the other machine learning models
#We have used a batch size of 50 and total epochs = 1000
#Final mean squared error has been reported


from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
import keras.losses

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

print(tf.__version__)

xtrin = pd.read_csv("xtrin_final.csv")
xtest = pd.read_csv("xtest_final.csv")
ytrain = pd.read_csv("ytrain_final.csv", header = None)
ytest = pd.read_csv("ytest_final.csv", header = None)

xtrin = xtrin.drop(xtrin.columns[0], axis = 1)
xtest = xtest.drop(xtest.columns[0], axis = 1)
ytrain = ytrain.drop(ytrain.columns[0], axis = 1)
ytest = ytest.drop(ytest.columns[0], axis = 1)

#3 layer deep Neural Network with relu activation
model = keras.Sequential([
    keras.layers.Dense(13, activation = 'relu'),
    keras.layers.Dense(8, activation = 'relu'),
    keras.layers.Dense(1, activation = 'linear')
    ])

model.compile(optimizer= 'adam', loss= 'mean_absolute_percentage_error', metrics = ['accuracy'])

epinn = model.fit(xtrin.values, ytrain.values, epochs = 1000, batch_size = 50)

mean_squared_error(ytest.values, model.predict(xtest.values)) #27.70
