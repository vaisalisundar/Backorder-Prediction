# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 19:41:11 2020

@author: SubhiDev
"""

import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc

import tensorflow as tf

from tensorflow.keras.regularizers import l2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
data = pd.read_csv("BackOrders.csv",header=0)
data.shape
data.columns
data.index
data.head(10)
data.describe(include='all')
data.dtypes
data.nunique()
for col in ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']:
    data[col] = data[col].astype('category')
data.dtypes
data.drop('sku', axis=1, inplace=True)
data.isnull().sum()
print (data.shape)
data = data.dropna(axis=0)
print(data.isnull().sum())
print("----------------------------------")
print(data.shape)
print (data.columns)
categorical_Attributes = data.select_dtypes(include=['category']).columns
data = pd.get_dummies(columns=categorical_Attributes, data=data, prefix=categorical_Attributes, prefix_sep="_",drop_first=True)
print (data.columns, data.shape)
X, y = data.loc[:,data.columns!='went_on_backorder_Yes'].values, data.loc[:,'went_on_backorder_Yes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify = data['went_on_backorder_Yes'])
perceptron_model = Sequential()

perceptron_model.add(Dense(1, input_dim=21, activation='sigmoid', kernel_initializer='normal'))
perceptron_model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
perceptron_model_history = perceptron_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)
print(perceptron_model_history.history.keys())
plt.plot(perceptron_model_history.history['accuracy'])
plt.plot(perceptron_model_history.history['val_accuracy'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()
