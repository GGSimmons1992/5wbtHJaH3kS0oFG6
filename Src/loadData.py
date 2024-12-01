#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

import sys
sys.path.insert(0, "../Src/")

get_ipython().run_line_magic('autosave', '5')


# In[2]:


def loadData(ticker):
    train = yf.download(ticker, start="2020-01-01", end="2024-03-01")[['Close','High','Low']].reset_index()
    test = yf.download(ticker, start="2024-03-01", end="2024-10-13")[['Close','High','Low']].reset_index()

    columnRenameDict = {
        'Date' : 'ds',
        'Close': 'y',
        'High': 'cap',
        'Low': 'floor'
    }
    
    train = train.rename(columns = columnRenameDict)
    test = test.rename(columns = columnRenameDict)

    train = createRollingAverageDF(train)
    test = createRollingAverageDF(test)
    
    return train,test


# In[3]:


def processDataForLSTM(data, timeStep=20):
    data = np.array(data)
    X, y = [], []
    for i in range(len(data) - timeStep - 1):
        inputValues = list(np.array(data[i:i + timeStep]).reshape(-1, 1))
        if (i + timeStep) < len(data):
            X.append(inputValues)
            y.append(data[i + timeStep])
        else:
            print(f"Index {i + timeStep} is out of bounds for data length {len(data)}")
    return np.array(X), np.array(y)
    


# In[4]:


def createRollingAverageDF(df):
    columns = [str(col) for col in df.columns]
    for col in columns:
        if col != 'ds':
            df[col] = df[col].rolling(window=5).mean()
            mean_value = np.nanmean(df[col])
            # Fill NaN values with the mean
            df[col] = np.where(np.isnan(df[col]), mean_value, df[col])
    return df


# In[5]:


def main():
    train,test = loadData("MSFT")
    display(train)
    display(test)
    XTrain,yTrain = processDataForLSTM(train['y'])
    XTest,yTest = processDataForLSTM(test['y'])
    print('XTrain.shape: ', XTrain.shape)
    print('yTrain.shape: ', yTrain.shape)
    print('XTest.shape: ', XTest.shape)
    print('yTest.shape: ', yTest.shape)


# In[6]:


if __name__ == '__main__':
    main()

