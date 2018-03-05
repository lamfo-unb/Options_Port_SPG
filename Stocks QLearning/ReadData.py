# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:22:44 2016

@author: Stefano
"""

import pandas as pd
import pandas_datareader.data as pdr
import re
from datetime import datetime,timedelta


def getPrices(stockName,start_day,end_day):
    
    start,end = getTime(start_day,end_day)
    df = pdr.DataReader(stockName,'yahoo', start,end)
    #df_close = pd.DataFrame(data = df['Close'].values, index = df.index, columns = [stockName])

    return df
 
def getTime(start_day,end_day):
    start_date = re.findall(r'\d+',start_day)
    end_date = re.findall(r'\d+',end_day)
    start=datetime(int(start_date[0]), int(start_date[1]), int(start_date[2]))
    end=datetime(int(end_date[0]), int(end_date[1]), int(end_date[2])) + timedelta(days=1)
    return start,end

def mergeData(df_close,stockName,start,end):
    number_of_stocks = len(stockName)
    for i in range(1,number_of_stocks):
        df_temp = getPrices(stockName[i],start,end)
        df_close = pd.concat([df_temp,df_close],axis = 1)
        
    return df_close

 