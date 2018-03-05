# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:22:44 2016

@author: Stefano
"""

import pandas as pd
import matplotlib.pyplot as plt
from Technical import bollinger, moving_average_convergence, moving_average, stochastic_oscillator, on_balance_volume


def plot_Bollinger(df):
    boll,lower,upper = bollinger(df,10)
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Bollinger Bands");plt.xlabel("Date")
    plt.plot(df,"b", label="BBDC");plt.legend()
    plt.plot(upper,"g", label="Upper");plt.legend()
    plt.plot(lower,"r", label="Lower");plt.legend()
    plt.subplot(212);plt.title("Bollinger Value")
    plt.plot(boll,"r", label="Bollinger Value");plt.legend()
    plt.tight_layout()
    plt.show()
           
    

def plot_MACD(df):
    macd,signal,mov = moving_average_convergence(df)
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Stock");plt.xlabel("Date")
    plt.plot(df,"b", label="BBDC");plt.legend()    
    plt.subplot(212);plt.title("Moving Average Divergence")
    plt.plot(macd,"g", label="MACD");plt.legend()
    plt.plot(signal,"r", label="Signal");plt.legend()
    plt.plot(mov,"b", label="BBDC");plt.legend()    
    plt.tight_layout()
    plt.show()
           
def plot_MA(df):
    MA_8,MA_100,MA_8_percentage,MA_100_percentage = moving_average(df)
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Stock");plt.xlabel("Date")
    plt.plot(df,"b", label="BBDC");plt.legend()
    plt.plot(MA_8,"g", label="8 periods moving average");plt.legend()
    plt.plot(MA_100,"r", label="100 periods moving average");plt.legend()
    plt.subplot(212);plt.title("Percentage")
    plt.plot(MA_8_percentage,"g", label="2 days variation");plt.legend()
    plt.plot(MA_100_percentage,"r", label="10 days variation");plt.legend()
    plt.tight_layout()
    plt.show()
           

def plot_stochastic(df):
    K,D,stoch = stochastic_oscillator(df)    
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Stock");plt.xlabel("Date")
    plt.plot(df['Close'],"b", label="BBDC");plt.legend()   
    plt.subplot(212);plt.title("Stochastic Oscillator")
    plt.plot(K,"g", label="%K");plt.legend()
    #plt.plot(stoch,"b", label="%D");plt.legend() 
    plt.plot(D,"r", label="%D");plt.legend()
    plt.tight_layout()
    plt.show()

def plot_volume(df):    
    obv,volume = on_balance_volume(df)
    plt.figure(figsize=(15,9))
    plt.subplot(211);plt.title("Stock");plt.xlabel("Date")
    #plt.plot(obv,"r", label="%D");plt.legend()
    plt.plot(df['Close'],"b", label="BBDC");plt.legend()
    plt.subplot(212);plt.title("Stochastic Oscillator")
    #plt.plot(volume,"g", label="%K");plt.legend()
    plt.plot(obv,"r", label="%D");plt.legend()
    plt.tight_layout()
    plt.show()
    
