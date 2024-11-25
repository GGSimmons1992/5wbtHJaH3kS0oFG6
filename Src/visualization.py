import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import pandas as pd

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

def loadModel(modelName):
    with open(f'../Models/{modelName}.pkl', 'rb') as file:
        model = pickle.load(file)
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

def compareSimplePickleModel(fileName,date,validation,modelName):
    model = loadModel(fileName)
    if fileName == "sarima":
        predict = model.predict(n_periods=len(validation))
    else:
        predict = model.forecast(len(validation))
    compare(date,validation,predict,modelName)

    

