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
from matplotlib.lines import Line2D

def reshapeTo1DArray(data):
    return np.array(data).reshape(-1,)

def plotComparison(date,test,predict,folder,title,buySellSuggestions = None):
    
    date = reshapeTo1DArray(date)
    test = reshapeTo1DArray(test)
    predict = reshapeTo1DArray(predict)
    mae = mean_absolute_error(test,predict)
    fileName = title.replace(" ","")
    
    resultMessage = f'{title} MAE: {mae}'
    plt.figure()
    plt.plot(date,test,linestyle =':',color='k',label='actual')
    plt.plot(date,predict,linestyle='--',color='orange',label='predicted')
    if buySellSuggestions is not None:
        plt.plot(date, buySellSuggestions['BU'], linestyle='--', color='b')
        plt.plot(date, buySellSuggestions['BL'], linestyle='--', color='b')
        bollinger_legend = Line2D([0], [0], color='b', linestyle='--', label='bollinger bands')

        buy = buySellSuggestions[buySellSuggestions['predict'] < buySellSuggestions['BL']]
        sell = buySellSuggestions[buySellSuggestions['predict'] > buySellSuggestions['BU']]

        plt.scatter(buy['ds'],buy['predict'],marker='^', color='r',label = 'buy',s=100)
        plt.scatter(sell['ds'],sell['predict'],marker='v', color='g',label = 'sell',s=100)

        bollinger_legend = Line2D([0], [0], color='b', linestyle='--', label='bollinger bands')
        plt.legend(handles=[bollinger_legend], loc='best')
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

def compare(validation, predict, modelName):
    validation = interpolateMissingValues(validation)
    predict = interpolateMissingValues(predict)

    min_length = min(validation.shape[0], len(predict))
    validation = validation.iloc[-min_length:]
    predict = predict[-min_length:]

    date = validation['ds']
    actual = validation['y']

    buySellSuggestions = pd.DataFrame({
        'ds': date,
        'predict': predict,
        'BU': validation['BU'],
        'BL': validation['BL']
    }, columns=['ds', 'predict', 'BU', 'BL'])

    scaledTest = standardScale(actual)
    scaledPredict = standardScale(predict)
    seasonalTest = extractSeasonality(actual)
    seasonalPredict = extractSeasonality(predict)
    seasonalScaledTest = extractSeasonality(scaledTest)
    seasonalScaledPredict = extractSeasonality(scaledPredict)

    plot_data = [
        (actual, predict, 'RawPrice', f'{modelName} Raw Price'),
        (scaledTest, scaledPredict, 'ScaledPrice', f'{modelName} Scaled Price'),
        (seasonalTest, seasonalPredict, 'RawSeasonality', f'{modelName} Raw Seasonality'),
        (seasonalScaledTest, seasonalScaledPredict, 'ScaledSeasonality', f'{modelName} Scaled Seasonality')
    ]

    for data, pred, folder, title in plot_data:
        if folder == 'RawPrice':
            plotComparison(date, data, pred, folder, title, buySellSuggestions)
        else:
            plotComparison(date, data, pred, folder, title)


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
    periods = 5
    while (continueTraining):
        predict = makePrediction(train['y'],modelName,params,periods)
        fullPredict += predict
        train = data.iloc[:(train.shape[0]+periods)]
        if ((train.shape[0] + periods ) >= data.shape[0]):
            periods = data.shape[0] - train.shape[0]
            predict = makePrediction(train['y'],modelName,params,periods)
            fullPredict += predict
            continueTraining = False
            
    compare(validation,fullPredict,modelName)

