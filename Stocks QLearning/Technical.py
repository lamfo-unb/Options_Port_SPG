# -*- coding: utf-8 -*-
"""
Created on Sat Oct 01 14:30:04 2016

@author: Stefano
"""

import pandas as pd
import numpy as np


def daily_return(df):
    daily = df['Close'].copy()
    daily[1:] =  (daily[1:]/daily[:-1].values)-1
    daily.ix[0:1] = 0
    return daily

def bollinger(data,n):
    data = data['Close']
    roll_mean = data.rolling(window = n, center = False).mean()
    roll_std = data.rolling(window = n, center = False).std()
    bollinger_value = (data - roll_mean)/roll_std
    upper = roll_mean + 2*roll_std
    lower = roll_mean - 2*roll_std
    
    return bollinger_value,lower,upper
    
def moving_average_convergence(data, nslow=26, nfast=12):
    data = data['Close']
    EMA_12 = data.ewm(ignore_na=False,span=nfast,min_periods=0,adjust=True).mean()    
    EMA_26 = data.ewm(ignore_na=False,span=nslow,min_periods=0,adjust=True).mean()
    macd = EMA_12 - EMA_26
    signal = macd.ewm(ignore_na=False,span=9,min_periods=0,adjust=True).mean()
    sig = np.sign(macd - signal)
    mov = 0.5*np.abs(sig[1:] - sig[:-1].values)*macd
    return macd,signal,mov
    
def moving_average(data, nslow = 100, nfast = 8):
    data = data['Close']
    MA_8 = data.rolling(window = nfast, center = False).mean()
    MA_8_percentage = (MA_8[2:] / MA_8[:-2].values) - 1
    MA_8_percentage[0:7] = 0
    
    MA_100 = data.rolling(window = nslow, center = False).mean()
    MA_100_percentage = (MA_100[10:] / MA_100[:-10].values) - 1
    MA_100_percentage[0:99] = 0
    
    return MA_8,MA_100,MA_8_percentage,MA_100_percentage
    
def stochastic_oscillator(df,n_days=14):
    low_min = df['Low'].rolling(min_periods=1, window=n_days, center=False).min()
    high_max = df['High'].rolling(min_periods=1, window=n_days, center=False).max()
    K = (df['Close'] - low_min) / (high_max - low_min)
    K = K.rolling(window = 3, center = False).mean()
    D = K.rolling(window = 3, center = False).mean()
    sig = np.sign(D-K)
    stoch = 0.5*np.abs(sig[1:] - sig[:-1].values)*D
    return K,D,stoch
    
def on_balance_volume(df):
    mult = df['Close'][1:]/df['Close'][:-1].values
    mult[mult > 1] = 1
    mult[mult < 1] = -1
    volume_reference = df['Volume'][1:]
    aux = mult * volume_reference
    aux = np.cumsum(aux)
    obv = aux[1:]/aux[:-1].values    
    
    return obv,volume_reference


def volume(df):
    volume_reference = df['Volume'][1:]
    volume_mean = volume_reference.rolling(window = 20, center = False).mean()
    volume = (volume_reference - volume_mean)/volume_mean
    return volume
