import json
import loadData
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import pandas as pd
import trainModel as tm
import pmdarima as pm
from IPython.display import display

def reshapeTo1DArray(data):
    return np.array(data).reshape(-1,)

def plotComparison(date,test,predict,folder,title):
    
    date = reshapeTo1DArray(date)
    test = reshapeTo1DArray(test)
    predict = reshapeTo1DArray(predict)
    mae = mean_absolute_error(test,predict)
    fileName = title.replace(" ","")
    
    resultMessage = f'{title} MAE: {mae}'
    plt.figure()
    plt.plot(date,test,label='actual')
    plt.plot(date,predict,label='predicted')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.title(resultMessage) 
    plt.legend()
    plt.show()
    plt.savefig(f'../Figures/{folder}/{fileName}')
    return resultMessage

def standardScale(data):
    data = reshapeTo1DArray(data)
    dataMean = np.mean(data)
    dataSTD = np.std(data)
    scaledData = (data - dataMean)/dataSTD
    return scaledData

def interpolateMissingValues(data):
    if np.any(np.isnan(data)):
        data = pd.Series(data).interpolate(method='linear').values
    return data

def extractSeasonality(data):
    data = reshapeTo1DArray(data)
    print(f'{np.isnan(data).sum()*100/len(data)}% of data is missing')
    if np.any(np.isnan(data)):
        print(data)
    result = seasonal_decompose(data, model='stl', period=20)
    return result.seasonal

def loadModel(data,modelName,params):
    model = None
    if modelName == 'Autoregression':
        model = tm.trainAutoRegression(data,params)
    elif modelName == 'ARIMA':
        model = tm.trainArima(data,params)
    elif modelName == 'Exponential Smoothing':
        model = tm.trainExponentialSmoothing(data,params)
    return model

def compare(date,test,predict,modelName):
    test = interpolateMissingValues(test)
    predict = interpolateMissingValues(predict)
    
    scaledTest = standardScale(test)
    scaledPredict = standardScale(predict)
    seasonalTest = extractSeasonality(test)
    seasonalPredict = extractSeasonality(predict)
    seasonalScaledTest = extractSeasonality(scaledTest)
    seasonalScaledPredict = extractSeasonality(scaledPredict)

    print(plotComparison(date,test,predict,'RawPrice',f'{modelName} Raw Price'))
    print(plotComparison(date,scaledTest,scaledPredict,'ScaledPrice',f'{modelName} Scaled Price'))
    print(plotComparison(date,seasonalTest,seasonalPredict,'RawSeasonality',f'{modelName} Raw Seasonality'))
    print(plotComparison(date,seasonalScaledTest,seasonalScaledPredict,'ScaledSeasonality',f'{modelName} Scaled Seasonality'))

def makePrediction(data,modelName,params,periods):
    if modelName == "SARIMA":
        model = pm.auto_arima(data,stepwise=True,seasonal=True)
        predict = model.predict(n_periods=periods)
    else:  
        model = loadModel(data,modelName,params)
        predict = model.forecast(periods)
    return list(np.array(predict).reshape(-1,))
        

def compareSimplePickleModel(data,modelName,paramsFile = ''):
    train,validation = loadData.splitData(data)
    fullPredict = []
    continueTraining = True
    params = {}
    if paramsFile != '':
        with open(f'../Models/{paramsFile}.json') as d:
            params = json.load(d)
    periods = 10
    while (continueTraining):
        predict = makePrediction(train['y'],modelName,params,periods)
        fullPredict += predict
        train = data.iloc[:(train.shape[0]+periods)]
        if ((train.shape[0] + periods ) >= data.shape[0]):
            periods = data.shape[0] - train.shape[0]
            predict = makePrediction(train['y'],modelName,params,periods)
            fullPredict += predict
            continueTraining = False
            
    print('len(fullPredict) ',len(fullPredict))
    print("len(validation['y']) ",len(validation['y']))
    compare(validation['ds'],validation['y'],fullPredict,modelName)

def makeProphetPrediction(data,periods):
    model = tm.trainProphet(data)
    future = model.make_future_dataframe(periods=periods,include_history=False)
    #capValue = test['y'].max() * 1.1
    #future['cap'] = capValue
    #future['floor'] = 0
    print('future.shape ',future.shape)
    return model.predict(future)

