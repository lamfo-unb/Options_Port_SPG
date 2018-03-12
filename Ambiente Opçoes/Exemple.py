# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 22:53:54 2018

@author: pedro
"""
import dask.dataframe as dsk
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
from OptinosClass import*

## ---- Interaction with the bases. 
##---- Evolution

def randomAction(k):
    '''
       Calcula Açao randomica, vetor de pesos para ativo.
           k: pandas dataframe do stado.
    '''
    w=np.random.normal(0.05, 0.03, size=len(k))
    w=w*(w).mean()
    W=w/sum(w)
    return(W)

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Acompanhamento de PL')
    plt.xlabel('date')
    plt.ylabel('$')
    plt.plot(durations_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated



## -------------   captura de arquivos - Identifique as pasta em que estao os arquivos.
lista="C:\\Users\\"+getpass.getuser()+"\\Documents\\optons full\\optons full"
numOpc=12 #     numero deativos por grupo
Refp=1 #        numero de meses a fente, maturidade de referencia.
numPeriodos=0 # numero de periodos, solo ou at'e expira'cao.
callput=0 #     nao implementado.
Capital0=50000# Captal teorico.
Ambidente=Variaveis['Ambi_teste']
OP=OptinsBases(lista,numOpc,Refp,numPeriodos,callput,Ambidente,Capital0)


## --- Start with a expecific asset
#OP.ListaAtivos
#state=OP.StartPred('BRVL5.csv','2018-01-08')

#for epoc in range(8):
#xx=time.time()
#state=OP.Start()
#print((time.time()-xx))


results=[]
for epoc in range(1000):
    try:
        state=OP.Start()
    except:
        state=OP.Start()
       
    ## --- randoum exemples actions
    OP.Reset()
    episode_durations=[]
    while True:
        action=randomAction(OP.StateList[0])
        new_state,Rew,done=OP.Action(action)
        ##-- graficos.
        episode_durations.append(OP.Capital1)
#        plot_durations()
#        time.sleep(0.2)
        if done:
#            print("Retorno imediato:", str(Rew))
            print("total de patrimonio:", str(OP.Capital1))
            results.append(OP.Capital1)
            break

len(results)
mean(results)

print("Ativo:", str(OP.ativo))
sum((np.array(results)/OP.Capital0)-1<0)/len(results)
sum((np.array(results)/OP.Capital0)-1<-0.2)/len(results)

sum((np.array(results)/OP.Capital0)-1>0)/len(results)
sum((np.array(results)/OP.Capital0)-1>1)/len(results)
sum((np.array(results)/OP.Capital0)-1>2)/len(results)
sum((np.array(results)/OP.Capital0)-1>3)/len(results)


import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

plt.hist(np.sort(results)[:-50]/Capital0,500, facecolor='blue')
plt.show()

np.sort(results)



#
#
#OP.ativo
#jj=open(OP.lista+'\\0mapa\\mapa.json')
#jj=jj.read()
#jj=json.loads(jj)
#
#np.random.choice((sort(np.array(list(set(jj[OP.ativo]))).astype('datetime64'))[:-2]), 1)[0].astype('datetime64')
#
#(TratamentoStado(OP.Ambiente,OP.StateList[OP.StateCont]),_,OP.statos)
#
#
#
#(OP.Capital1<=OP.Capital0*0.2)
#(OP.StateCont==(len(OP.StateList)-2))
#(OP.StateList[OP.StateCont]['Date']==OP.StateList[OP.StateCont]['EXPIR_DATE']).sum()<0)
#
#OP.StateCont<len(OP.StateList)
#     
#OP.ListaAtivos
#state1=OP.StartPred(OP.ativo)
#state1.StateList[0]
#
#OP.StateList[1][['CLOSE_x','CLOSE_1','CLOSE_2','Date','EXPIR_DATE']]
##OP.StateList[-1][['PrecoFut1','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
##OP.StateList[-2][['PrecoFut1','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
##OP.StateList[0][['PrecoFut1','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
#


