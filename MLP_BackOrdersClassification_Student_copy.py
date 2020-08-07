#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement
# 
#     Identify products at risk of backorder before the event occurs so that business has time to react.

# ### What is a Backorder?
#     Backorders are products that are temporarily out of stock, but a customer is permitted to place an order against future inventory. A backorder generally indicates that customer demand for a product or service exceeds a company’s capacity to supply it. Back orders are both good and bad. Strong demand can drive back orders, but so can suboptimal planning.

# ### Data
# 
# Data file contains the historical data for the 8 weeks prior to the week we are trying to predict. The data was taken as weekly snapshots at the start of each week. Columns are defined as follows:
# 
#     sku - Random ID for the product
# 
#     national_inv - Current inventory level for the part
# 
#     lead_time - Transit time for product (if available)
# 
#     in_transit_qty - Amount of product in transit from source
# 
#     forecast_3_month - Forecast sales for the next 3 months
# 
#     forecast_6_month - Forecast sales for the next 6 months
# 
#     forecast_9_month - Forecast sales for the next 9 months
# 
#     sales_1_month - Sales quantity for the prior 1 month time period
# 
#     sales_3_month - Sales quantity for the prior 3 month time period
# 
#     sales_6_month - Sales quantity for the prior 6 month time period
# 
#     sales_9_month - Sales quantity for the prior 9 month time period
# 
#     min_bank - Minimum recommend amount to stock
# 
#     potential_issue - Source issue for part identified
# 
#     pieces_past_due - Parts overdue from source
# 
#     perf_6_month_avg - Source performance for prior 6 month period
# 
#     perf_12_month_avg - Source performance for prior 12 month period
# 
#     local_bo_qty - Amount of stock orders overdue
# 
#     deck_risk - Part risk flag
# 
#     oe_constraint - Part risk flag
# 
#     ppap_risk - Part risk flag
# 
#     stop_auto_buy - Part risk flag
# 
#     rev_stop - Part risk flag
# 
#     went_on_backorder - Product actually went on backorder. This is the target value.

# ### Data Pre-processing
# #### Loading the required libraries

# In[1]:


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
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# #### Loading the data

# In[2]:


data = pd.read_csv('BackOrders.csv')


# #### Understand the Data

# See the number row and columns

# In[3]:


data.shape


# Display the columns

# In[4]:


data.columns


# Display the index

# In[5]:


data.index


# See the top rows of the data

# In[6]:


data.head()


# Shows a quick statistic summary of your data using describe.
# 
#     For object data (e.g. strings or timestamps), the result’s index will include count, unique, top, and freq. 
# 
#         The top is the most common value.
# 
#         The freq is the most common value’s frequency.

# In[7]:


data.describe(include='all')


# Display data type of each variable

# In[8]:


data.dtypes


# #### Get the unique values of each column

# In[9]:


data.nunique()


# #### Observations
# 
# potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, and went_on_backorder are categorical but is interpreted as object. 

# #### Convert all the attributes to appropriate type

# Data type conversion
# 
#     Using astype('category') to convert potential_issue, deck_risk, oe_constraint, ppap_risk, stop_auto_buy, rev_stop, and went_on_backorder attributes to categorical attributes.
# 

# In[10]:


cat_col = ['potential_issue', 'deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']
for i in cat_col:
     data[i] = data[i].astype('category')


# Display data type of each variable

# In[11]:


data.dtypes


# ##### Delete sku attribute

# In[12]:


data.drop('sku', axis=1, inplace=True)


# #### Missing Data
# 
#     Missing value analysis and dropping the records with missing values

# In[13]:


data.isnull().sum()


# Observing the number of records before and after missing value records removal

# In[14]:


print (data.shape)


# Since the number of missing values is about 5%. For initial analysis we ignore all these records

# In[15]:


data = data.dropna(axis=0)


# In[16]:


print(data.isnull().sum())
print("----------------------------------")
print(data.shape)


# #### Converting Categorical to Numeric
# 
# For some of the models all the independent attribute should be of type numeric and ANN model is one among them.
# But this data set has some categorial attributes.
# 
# 'pandas.get_dummies' To convert convert categorical variable into dummy/indicator variables
# 

# In[17]:


print (data.columns)


# ##### Creating dummy variables.
# 
# If we have k levels in a category, then we create k-1 dummy variables as the last one would be redundant. So we use the parameter drop_first in pd.get_dummies function that drops the first level in each of the category
# 

# In[18]:


categorical_Attributes = data.select_dtypes(include=['category']).columns


# In[19]:


data = pd.get_dummies(columns=categorical_Attributes, data=data, prefix=categorical_Attributes, prefix_sep="_",drop_first=True)


# In[20]:


print (data.columns, data.shape)


# #### Train-Test Split
# 
# Using sklearn.model_selection.train_test_split
# 
#     Split arrays or matrices into train and test subsets

# In[21]:


X, y = data.loc[:,data.columns!='went_on_backorder_Yes'].values, data.loc[:,'went_on_backorder_Yes'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify = data['went_on_backorder_Yes'])


# ### Perceptron Model

# In[22]:


perceptron_model = Sequential()

perceptron_model.add(Dense(1, input_dim=21, activation='relu', kernel_initializer='normal'))


# In[23]:


perceptron_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:


perceptron_model_history = perceptron_model.fit(X_train,y_train, epochs=100, batch_size=64,validation_split=0.2)


# ##### Plot

# In[25]:


print(perceptron_model_history.history.keys())


# In[26]:


plt.plot(perceptron_model_history.history['acc'])
plt.plot(perceptron_model_history.history['val_acc'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.show()


# In[ ]:





# In[ ]:





# ##### Predictions

# In[27]:


test_pred = perceptron_model.predict_classes(X_test)
train_pred = perceptron_model.predict_classes(X_train)

confusion_matrix_test= confusion_matrix(y_test, test_pred)
confusion_matrix_train = confusion_matrix(y_train, train_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# ##### Train Test Accuracy, True Negative Rate and True Positive Rate

# In[28]:


accuracy_train =(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
print("Train Accuracy: ", accuracy_train)

print("-----------------------")

accuracy_test = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])
print("Test Accuracy: ", accuracy_test)


# ### MLP with 2 layers
# 
#     1 hidden layer with 15 neurons

# In[29]:


perceptron_model2 = Sequential()

perceptron_model2.add(Dense(15, input_dim=21, activation='sigmoid', kernel_initializer='normal'))
perceptron_model2.add(Dense(1, activation='relu', kernel_initializer='normal'))


# In[30]:


perceptron_model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[31]:


perceptron_model1_history = perceptron_model2.fit(X_train,y_train, epochs=100, batch_size=64,validation_split=0.2)


# In[ ]:





# ##### Plot

# In[32]:


print(perceptron_model1_history.history.keys())


# In[33]:


plt.plot(perceptron_model1_history.history['acc'])
plt.plot(perceptron_model1_history.history['val_acc'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[34]:


plt.plot(perceptron_model1_history.history['loss'])
plt.plot(perceptron_model1_history.history['val_loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# #### Predictions

# In[35]:


train_model1_pred = perceptron_model2.predict_classes(X_train)
test_model1_pred = perceptron_model2.predict_classes(X_test)


# #### Getting evaluation metrics and evaluating model performance

# In[36]:


confusion_matrix_train = confusion_matrix(y_train, train_model1_pred)
confusion_matrix_test = confusion_matrix(y_test, test_model1_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[37]:


Accuracy_Train_M1 =(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
print("Train Accuracy: ",Accuracy_Train_M1)

print("-----------------------")

Accuracy_Test_M1 = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])
print("Test Accuracy: ",Accuracy_Test_M1)


# ### MLP with 2 layers
# 
#     1 hidden layer with 20 neurons

# In[38]:


perceptron_model3 = Sequential()

perceptron_model3.add(Dense(20, input_dim=21, activation='sigmoid', kernel_initializer='normal'))
perceptron_model3.add(Dense(1, activation='relu', kernel_initializer='normal'))


# In[39]:


perceptron_model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[40]:


perceptron_model3_history = perceptron_model3.fit(X_train,y_train, epochs=100, batch_size=64,validation_split=0.2)


# In[ ]:





# ##### Plot

# In[41]:


print(perceptron_model3_history.history.keys())


# In[42]:


plt.plot(perceptron_model3_history.history['acc'])
plt.plot(perceptron_model3_history.history['val_acc'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[43]:


plt.plot(perceptron_model3_history.history['loss'])
plt.plot(perceptron_model3_history.history['val_loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# #### Predictions

# In[44]:


train_model2_pred = perceptron_model3.predict_classes(X_train)
test_model2_pred = perceptron_model3.predict_classes(X_test)


# #### Getting evaluation metrics and evaluating model performance

# In[45]:


confusion_matrix_train = confusion_matrix(y_train, train_model2_pred)
confusion_matrix_test = confusion_matrix(y_test, test_model2_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[46]:


Accuracy_Train_M2 =(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])
print("Train Accuracy: ",Accuracy_Train_M2)

print("-----------------------")

Accuracy_Test_M2 = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])
print("Test Accuracy: ",Accuracy_Test_M2)


# ### MLP with 2 layers
# 
#     1 hidden layer with 25 neurons

# In[47]:


mlp_model3 = Sequential()

mlp_model3.add(Dense(25, input_dim=21, activation='tanh', kernel_initializer='normal'))
mlp_model3.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))


# In[48]:


mlp_model3.summary()


# In[49]:


mlp_model3.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


# In[50]:


model3_history = mlp_model3.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)


# ##### Plot

# In[51]:


print(model3_history.history.keys())


# In[55]:


plt.plot(model3_history.history['acc'])
plt.plot(model3_history.history['val_acc'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[56]:


plt.plot(model3_history.history['loss'])
plt.plot(model3_history.history['val_loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# #### Predictions

# In[57]:


train_model3_pred = mlp_model3.predict_classes(X_train)
test_model3_pred = mlp_model3.predict_classes(X_test)


# #### Getting evaluation metrics and evaluating model performance

# In[58]:


confusion_matrix_train = confusion_matrix(y_train, train_model3_pred)
confusion_matrix_test = confusion_matrix(y_test, test_model3_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[59]:


Accuracy_Train_M3 =(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train Accuracy: ",Accuracy_Train_M3)

print("-----------------------")

Accuracy_Test_M3 = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

print("Test Accuracy: ",Accuracy_Test_M3)


# Observation:
# 
#     Based on the TPR, 2 layer MLP with 20 nodes hidden layer is best

# ### MLP with 3 layers
# 
#     1st hidden layer with 25 neurons
#     2nd hidden layer with 15 neurons

# In[60]:


mlp_model4 = Sequential()

mlp_model4.add(Dense(25, input_dim=21, activation='tanh', kernel_initializer='normal'))
mlp_model4.add(Dense(15, activation='relu', kernel_initializer='normal'))
mlp_model4.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))


# In[61]:


mlp_model4.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[62]:


model4_history = mlp_model4.fit(X_train, y_train, epochs=150, batch_size=64, validation_split=0.2)


# ##### Plot

# In[ ]:


print(model4_history.history.keys())


# In[ ]:


plt.plot(model4_history.history['acc'])
plt.plot(model4_history.history['val_acc'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[ ]:


plt.plot(model4_history.history['loss'])
plt.plot(model4_history.history['val_loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# #### Predictions

# In[ ]:


train_model4_pred = mlp_model4.predict_classes(X_train)
test_model4_pred = mlp_model4.predict_classes(X_test)


# #### Getting evaluation metrics and evaluating model performance

# In[ ]:


confusion_matrix_train = confusion_matrix(y_train, train_model4_pred)
confusion_matrix_test = confusion_matrix(y_test, test_model4_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[ ]:


Accuracy_Train_M4 =(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train Accuracy: ",Accuracy_Train_M4)

print("-----------------------")

Accuracy_Test_M4 = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

print("Test Accuracy: ",Accuracy_Test_M4)


# ### MLP with 3 layers
# 
#     1st hidden layer with 25 neurons
#     2nd hidden layer with 20 neurons

# In[ ]:


from keras.regularizers import l2
mlp_model5 = Sequential()

mlp_model5.add(Dense(25, input_dim=21, activation='tanh', kernel_initializer='glorot_normal', kernel_regularizer= l2(0.03) ))
mlp_model5.add(Dense(15, activation='relu', kernel_initializer='normal'))
mlp_model5.add(Dense(1, activation='tanh', kernel_initializer='normal'))


# In[ ]:


mlp_model5.summary()


# In[ ]:


mlp_model5.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


model5_history = mlp_model5.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2)


# ##### Plot

# In[ ]:


print(model5_history.history.keys())


# In[ ]:


plt.plot(model5_history.history['acc'])
plt.plot(model5_history.history['val_acc'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[ ]:


plt.plot(model5_history.history['loss'])
plt.plot(model5_history.history['val_loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# #### Predictions

# In[ ]:


train_model5_pred = mlp_model5.predict_classes(X_train)
test_model5_pred = mlp_model5.predict_classes(X_test)


# #### Getting evaluation metrics and evaluating model performance

# In[ ]:


confusion_matrix_train = confusion_matrix(y_train, train_model5_pred)
confusion_matrix_test = confusion_matrix(y_test, test_model5_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[ ]:


Accuracy_Train_M5 =(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train Accuracy: ",Accuracy_Train_M5)
print("-----------------------")

Accuracy_Test_M5 = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

print("Test Accuracy: ",Accuracy_Test_M5)


# Observation:
# 
#     Model 4 performs better with 25 and 15 neurons
#     

# ## Find best parameter

# #### Function for creation of model

# In[ ]:


def model_def(h_activation, o_activation, kernel_init):
    model = Sequential()
    model.add(Dense(25,input_dim = 21,activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dense(15,activation='relu',kernel_initializer='glorot_normal'))
    model.add(Dense(1,activation='sigmoid',kernel_initializer='glorot_normal'))
    return model 
 


# #### Store the best parameters

# In[ ]:


best_params = {}


# ### Find the best Learning rate

# In[ ]:


lrs = [0.0001,0.001,0.01,0.1]


# In[ ]:


hist_loss = []

for lr in lrs:
    
    lr_model = model_def(h_activation='tanh', o_activation='sigmoid', kernel_init='normal')
    
    # Compile model
    optimizer = optimizers.Adam(lr=lr)
    
    lr_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    # Fit the model
    lr_model_history = lr_model.fit(X_train, y_train, validation_split=0.2, epochs=1, 
                                    shuffle=True, steps_per_epoch=50, validation_steps=50)
    
    hist_loss.append(lr_model_history.history['loss'])


# In[ ]:


# Get the lr and loss Dataframe
loss_lr = pd.DataFrame([lrs,hist_loss]).T  

#Give the coloumn names
loss_lr.columns=['lr', 'loss']

#Sort the values and reset the index
loss_lr=loss_lr.sort_values('loss').reset_index().drop('index',axis=1)
loss_lr


# In[ ]:


#pick the top lr
best_params['best_learning_rate'] = loss_lr.loc[:,'lr'][0]


# #### Build model with the best learning rate

# #### Got the below value as best learning rate after different experiments

# In[ ]:


bst_lr_model = model_def(h_activation='tanh', o_activation='sigmoid', kernel_init='normal')


# In[ ]:


bst_lr_model.summary()


# In[ ]:





# In[ ]:


bst_lr_model_history = bst_lr_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, shuffle=True)


# In[ ]:


print(bst_lr_model.history.history.keys())


# In[ ]:


plt.plot(bst_lr_model.history.history['accuracy'])
plt.plot(bst_lr_model.history.history['val_accuracy'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[ ]:


plt.plot(bst_lr_model.history.history['loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# #### Find the best Batch size 

# In[ ]:


batch_sizes=[32,64,128,256,512,1024]


# In[ ]:


history=[]

for batch_size in batch_sizes:
        
    bs_model = model_def(h_activation='tanh', o_activation='sigmoid', kernel_init='normal')
    
     # Compile model
    sgd = optimizers.SGD(lr=best_params['best_learning_rate'])
    bs_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    #Fit the model
    bs_model_history = bs_model.fit(X_train, y_train, batch_size=batch_size, epochs=10,
                                    validation_split=0.2, shuffle=True)
    
    history.append(bs_model_history)


# ###### Summarize history for train loss

# In[ ]:


for i in range(0, len(history)):
    plt.plot(history[i].history['loss'])
plt.title('Model Train Loss')
plt.ylabel('loss')
plt.xlabel('epochs')  
plt.legend(batch_sizes, loc='upper left')
plt.show()


# ###### Summarize history for test loss

# In[ ]:


for i in range(0, len(history)):
    plt.plot(history[i].history['val_loss'])
plt.title('Model Test Loss')
plt.ylabel('loss')
plt.xlabel('epoch')  
plt.legend(batch_sizes, loc='upper left')
plt.show()


# #### Build model with the best batch size

# In[ ]:





# In[ ]:


bst_bs_model = model_def(h_activation='tanh', o_activation='sigmoid', kernel_init='normal')


# In[ ]:


sgd = optimizers.SGD(lr=best_params['best_learning_rate'])
bst_bs_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])


# In[ ]:





# ##### Plot

# In[ ]:


print(bs_model_history.history.keys())


# In[ ]:


plt.plot(bst_bs_model.history.history['acc'])
plt.plot(bst_bs_model.history.history['val_acc'])
plt.title('Accuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# In[ ]:


plt.plot(bst_bs_model.history.history['loss'])
plt.plot(bst_bs_model.history.history['val_loss'])
plt.title('Loss Function Plot')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()


# #### Predictions

# In[ ]:


train_model5_pred = bst_bs_model.predict_classes(X_train)
test_model5_pred = bst_bs_model.predict_classes(X_test)


# #### Getting evaluation metrics and evaluating model performance

# In[ ]:


confusion_matrix_train = confusion_matrix(y_train, train_model5_pred)
confusion_matrix_test = confusion_matrix(y_test, test_model5_pred)

print(confusion_matrix_train)
print(confusion_matrix_test)


# #### Calculate Accuracy, True Positive Rate and True Negative Rates

# In[ ]:


Accuracy_Train_M5 =(confusion_matrix_train[0,0]+confusion_matrix_train[1,1])/(confusion_matrix_train[0,0]+confusion_matrix_train[0,1]+confusion_matrix_train[1,0]+confusion_matrix_train[1,1])

print("Train Accuracy: ",Accuracy_Train_M5)
print("-----------------------")

Accuracy_Test_M5 = (confusion_matrix_test[0,0]+confusion_matrix_test[1,1])/(confusion_matrix_test[0,0]+confusion_matrix_test[0,1]+confusion_matrix_test[1,0]+confusion_matrix_test[1,1])

print("Test Accuracy: ",Accuracy_Test_M5)


# ### Reference Links:
# 
# https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
# 
# https://keras.io/
