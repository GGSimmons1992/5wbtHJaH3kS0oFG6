#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf

import sys
sys.path.insert(0, "../Src/")

get_ipython().run_line_magic('autosave', '5')


# In[2]:


def loadData(ticker):
    train = yf.download(ticker, start="2020-01-01", end="2024-03-01")[['Close']].reset_index()
    test = yf.download(ticker, start="2024-03-01", end="2024-10-13")[['Close']].reset_index()

    train = train.rename(columns = {'Date' : 'ds', 'Close': 'y'})
    test = test.rename(columns = {'Date' : 'ds', 'Close': 'y'})
    
    return train,test


# In[3]:


def main():
    train,test = loadData("MSFT")
    display(train)
    display(test)


# In[4]:

if __name__ == '__main__':
    main()

