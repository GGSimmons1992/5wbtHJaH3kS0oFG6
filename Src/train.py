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
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pmdarima as pm
from statsmodels.tsa.api import AutoReg
import pickle
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import json
import os
from os.path import exists

import sys
sys.path.insert(0, "../Src/")
import loadData

get_ipython().run_line_magic('autosave', '5')


# In[2]:


def stabilize(data):
    p = 100
    while p > 0.05:
        adfTest = adfuller(data)
        p = adfTest[1]
        print(f'p: {adfTest[1]}')
        if p > 0.05:
            data = data.diff().dropna()
    return data


# In[3]:


def trainArima(data,params):
    arima = ARIMA(data, **params)
    arimaFit = arima.fit()
    return arimaFit


# In[4]:


def trainBestArima(data):
    print('Training ARIMA')
    data = stabilize(data)
    possibleP = np.arange(5)
    possibleD = np.arange(5)
    possibleQ = np.arange(5)

    train, dev = trainDevSplit(data)

    bestArima = None
    bestmse = np.inf
    bestP, bestQ = 0, 0

    for _ in range(20):
        p = np.random.choice(possibleP)
        d = np.random.choice(possibleD)
        q = np.random.choice(possibleQ)
        params = {'order': (p, d, q)}
        arima = ARIMA(train, order=(p, d, q))
        arimaFit = trainArima(train,params)
        forecast = arimaFit.forecast(len(dev))
        mse = mean_squared_error(dev, forecast)
        if mse < bestmse:
            bestmse = mse
            bestP, bestD, bestQ = p, d, q

    bestArimaParams = {
        'order': (int(bestP), int(bestD), int(bestQ))
    }
    with open('../Models/arima_params.json', 'w') as f:
        json.dump(bestArimaParams, f)
    
    arimaFit = trainArima(train,bestArimaParams)
    print(arimaFit.summary())
    residuals = arimaFit.resid[1:]

    fig,ax = plt.subplots(1,2)
    residuals.plot(title = 'Residuals', ax = ax[0])
    residuals.plot(title = 'Density', kind = 'kde', ax = ax[1])

    saveModel(arimaFit, 'arima')
    return arimaFit


# In[5]:


def saveModel(model,name):
    filename = f'../Models/{name}.pkl'
    if name in ['autoRegression','sarima','ExponentialSmoothing', 'prophet']:
        with open(filename, 'wb') as pkl:
            pickle.dump(model, pkl)
    else:
        model.save(filename)


# In[6]:


def trainSarima(train,saveSarima = False):
    print('Training Sarima')
    sarima = pm.auto_arima(train,stepwise=True,seasonal=True)
    print(sarima.summary())
    if (saveSarima):
        saveModel(sarima,'sarima')
    return sarima


# In[7]:


def trainAutoRegression(data,params):
    return AutoReg(data,**params).fit()


# In[8]:


def trainBestAutoRegression(data):
    print('Training AutoRegression')
    bestLag = 0
    bestCorr = 0
    for lag in range(1,11):
        corr = data.corr(data.shift(lag))
        if corr > bestCorr:
            bestLag = lag
            bestCorr = corr

    bestARParams = {
        'lags':bestLag
    }
    with open('../Models/ar_params.json', 'w') as f:
        json.dump(bestARParams, f)
    
    print(f'AR order = {bestLag}')
    ar_model = trainAutoRegression(data,bestARParams)
    saveModel(ar_model,'autoRegression')
    return ar_model
    


# In[9]:


def trainExponentialSmoothing(data,params):
    modelName = params['modelName']
    params = {
        'smoothing_level': params['smoothing_level']
    }
    constructors = {
        "SimpleExpSmoothing": SimpleExpSmoothing,
        "Holt": Holt,
        "ExponentialSmoothing": ExponentialSmoothing
    }
    model = constructors[modelName](data)
    return model.fit(**params)


# In[10]:


def trainBestExponentialSmoothing(data):
    print('Training ExponentialSmoothing')
    arrayData = np.asarray(data)
    trainDevCutOff = int(len(arrayData) * 0.8)
    train = arrayData[:trainDevCutOff]
    dev = arrayData[trainDevCutOff:]

    possibleModelNames = ["SimpleExpSmoothing", "Holt", "ExponentialSmoothing"]
    bestmse = np.inf

    bestModelParams = {
        'modelName': '',
        'smoothing_level': ''
    }
    for _ in range(20):
        smoothingLevel = np.random.uniform(0.1, 0.9)
        modelIndex = np.random.choice(len(possibleModelNames))
        params = {
            'modelName': possibleModelNames[modelIndex],
            'smoothing_level': smoothingLevel
        }
        
        model_fit = trainExponentialSmoothing(train,params)
        forecast = model_fit.forecast(len(dev))
        mse = mean_squared_error(dev, forecast)
        if mse < bestmse:
            bestmse = mse
            bestModelParams = params
            
    print(f'model chosen {bestModelParams["modelName"]}')
    print(f'smoothing parameter {bestModelParams["smoothing_level"]}')
    
    with open('../Models/exponentialSmoothing_params.json', 'w') as f:
        json.dump(bestModelParams, f)
    
    final_model_fit = trainExponentialSmoothing(train,bestModelParams)
    
    saveModel(final_model_fit, 'ExponentialSmoothing')
    return final_model_fit
    


# In[11]:


def trainProphet(data):
    print('Training Prophet')

    model = Prophet()
    model.fit(data)
    
    saveModel(model, 'prophet')
    return model
        


# In[12]:


def trainLSTM(data):
    train, dev = trainDevSplit(data)
    XData, yData = loadData.processDataForLSTM(data)
    XTrain, yTrain = loadData.processDataForLSTM(train)
    XDev, yDev = loadData.processDataForLSTM(dev)
    
    bestTrainScore = np.inf
    bestDevScore = np.inf
    bestLSTMUnits1 = 0
    bestLSTMUnits2 = 0
    bestLSTMUnits3 = 0
    bestEpochs = 0
    for trial in range(30):    
        lstmUnits1 = np.random.choice(range(1,128))
        lstmUnits2 = np.random.choice(range(1,128))
        lstmUnits3 = np.random.choice(range(1,128))
        epochs = np.random.choice(range(1,100))

        model = compileLSTM(XTrain,yTrain,lstmUnits1,lstmUnits2,lstmUnits3,epochs)
        
        devScore = model.evaluate(XDev,yDev)[1]
        if devScore < bestDevScore:
            bestTrainScore = model.evaluate(XTrain,yTrain)[1]
            bestDevScore = devScore
            bestLSTMUnits1 = lstmUnits1
            bestLSTMUnits2 = lstmUnits2
            bestLSTMUnits3 = lstmUnits3
            bestEpochs = epochs
    compileLSTM(XData,yData,bestLSTMUnits1,bestLSTMUnits2,lstmUnits3,bestEpochs)
    
    model_path = f'../Models/LSTM_.h5'
    model.save(model_path)
    bestModelParams = {
        'bestTrainScore': int(bestTrainScore),
        'bestDevScore': int(bestDevScore),
        'bestLSTMUnits1': int(bestLSTMUnits1),
        'bestLSTMUnits2': int(bestLSTMUnits2),
        'bestLSTMUnits3': int(bestLSTMUnits3),
        'bestEpochs': int(bestEpochs)
    }
    with open('../Models/lstm_params.json', 'w') as f:
        json.dump(bestModelParams, f)
    
    return model
        


# In[13]:


def trainDevSplit(data):
    totalRows = data.shape[0]
    trainDevCutoff = int(0.8 * totalRows)
    train = data.iloc[:trainDevCutoff]
    dev = data.iloc[trainDevCutoff:]
    return train,dev


# In[14]:


def compileLSTM(X,y,lstmUnits1,lstmUnits2,lstmUnits3,epochs):
    model = Sequential()
    model.add(LSTM(units=lstmUnits1, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=lstmUnits2, return_sequences=True))
    model.add(LSTM(units=lstmUnits3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['mean_squared_error'])
    model.fit(X, y, epochs=epochs, batch_size=32,verbose=0)
    return model


# In[15]:


def main():
    train, test = loadData.splitData(loadData.loadData('MSFT'))
    if exists('../Models/autoRegression.pkl') == False:
        trainBestAutoRegression(train['y'])
    if exists('../Models/arima.pkl') == False:
        trainBestArima(train['y'])
    if exists('../Models/sarima.pkl') == False:
        trainSarima(train['y'],True)
    if exists('../Models/ExponentialSmoothing.pkl') == False:
        trainBestExponentialSmoothing(train['y'])
    if exists('../Models/prophet.pkl') == False:
        trainProphet(train)
    
    if exists('../Models/LSTM_.h5') == False:
        trainLSTM(train['y'])


# In[16]:


if __name__ == '__main__':
    main()

