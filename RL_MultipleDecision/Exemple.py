# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:53:54 2018

@author: pedro
"""
from os import listdir, stat
import getpass
import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import torch
import sys

sys.path.insert(0,'C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\OptionsSVM\\Ambiente Opçoes')
from Variables import*
#from Options_Class_Fast_SQL import*
from Options_Class_Fast import*

#sys.path.insert(0,'C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\OptionsSVM\\Ambiente Opçoes\\Legado')
#from OptinosClass_Fast import*

## ---- Interaction with the bases. 
##---- Evolution

def randomAction(k):
    '''
       Calcula Açao randomica, vetor de pesos para ativo.
           k: pandas dataframe do stado.
    '''
    w=np.random.normal(0.02, 0.04, size=len(k))
    w=w*(w).mean()
    W=w/sum(w)
    return(W)
    
## -------------   captura de arquivos - Identifique as pasta em que estao os arquivos.
lista="C:\\Users\\"+getpass.getuser()+"\\Documents\\optons full\\optons full"
numOpc=12 #     numero deativos por grupo
Refp=2 #        numero de meses a fente, maturidade de referencia.
numPeriodos=0 # numero de periodos, solo ou at'e expira'cao.
callput=0 #     nao implementado.
Capital0=50000# Captal teorico.
Ambi=Variaveis['Ambi_teste']
dmais=130
OP=OptionsBases(lista,numOpc,Refp,numPeriodos,callput,Ambi,Capital0,dmais)


## --- Start with a expecific asset
#OP.ListaAtivos
#state=OP.StartPred('BRPOB.csv','2018-01-08')
#
#for epoc in range(8):
#for i in range(30):
#    xx=time.time()
#    state=OP.Start()
#    state=OP.StartPred('BRIO3.csv','2017-10-09')
#    print((time.time()-xx))
#    if (OP.StateList[-2].EXPIR_DATE.values[np.nonzero(OP.StateList[-2].EXPIR_DATE.values)].astype('datetime64')[0]-OP.StateList[-1].Date.values[np.nonzero(OP.StateList[-1].Date.values)].astype('datetime64')[0]).astype(int)>5:
#        print(1)
#        break
#    if sum(OP.StateList[-1].EXPIR_DATE.values==0)>18:
#        print(2)
#        break
##
##OP.StartPred
##len(OP.StateList)
#OP.StateList[0][['Security','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
#OP.StateList[-1][['Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
#


results=[]
for epoc in range(1000):
    xx=time.time()
    try:
        state=OP.Start()
    except:
        state=OP.Start()
    print((time.time()-xx))
    ## --- randoum exemples actions
    OP.Reset()
    episode_durations=[]
    while True:
        action=randomAction(OP.StateList[0])
        new_state,Rew,done,_=OP.Action(action)
        ##-- graficos.
        episode_durations.append(OP.Capital1)
#        plot_durations()
#        time.sleep(0.2)
        if done:
            print("Relacáo de operaçaos dias:", len(OP.StateList),OP.StateCont,len(OP.StateList)-OP.StateCont)
            print("Ultimo retorno:", str(Rew))
            print("total de patrimonio:", str(OP.Capital1))
            results.append(OP.Capital1)
            break



len(results)
mean(results)

print("Ativo:", str(OP.ativo))
sum((np.array(results)/OP.Capital0)-1<-1)/len(results)
sum((np.array(results)/OP.Capital0)-1<-0.2)/len(results)
sum((np.array(results)/OP.Capital0)-1<0)/len(results)

sum((np.array(results)/OP.Capital0)-1>0)/len(results)
sum((np.array(results)/OP.Capital0)-1>0.2)/len(results)
sum((np.array(results)/OP.Capital0)-1>1)/len(results)
sum((np.array(results)/OP.Capital0)-1>2)/len(results)
sum((np.array(results)/OP.Capital0)-1>3)/len(results)


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

plt.hist(np.sort(results)[8:-20]/Capital0-1,100, facecolor='blue')
plt.show()

OP.StateList[OP.StateCont]
len(OP.StateList)
OP.ativo

#
#OP.StateList[1][['CLOSE_x','CLOSE_1','CLOSE_2','Date','EXPIR_DATE']]
OP.StateList[1][['PrecoFut1','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
OP.StateList[-1][['PrecoFut1','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
##OP.StateList[0][['PrecoFut1','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
#


