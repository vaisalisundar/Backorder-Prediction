# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:06:46 2020

@author: Vaisali Sundar
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
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

data = pd.read_csv("BackOrders.csv",header=0)
data.shape
data.columns
data.index
data.head()
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
print (data.columns, data.shape)
X, y = data.loc[:,data.columns!='went_on_backorder_Yes'].values, data.loc[:,'went_on_backorder_Yes'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify = data['went_on_backorder_Yes'])