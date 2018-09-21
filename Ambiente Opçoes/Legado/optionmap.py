# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:06:43 2018

@author: pedro
"""

import dask.dataframe as dsk
from os import listdir, stat
import getpass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
sys.path.insert(0,'C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\OptionsSVM\\Ambiente Opçoes')

from OptinosClass_Fast import*
#from OptinosClass import*

lista="C:\\Users\\"+getpass.getuser()+"\\Documents\\optons full\\optons full"
numOpc=12 #     numero deativos por grupo
Refp=1 #        numero de meses a fente, maturidade de referencia.
numPeriodos=0 # numero de periodos, solo ou at'e expira'cao.
callput=0 #     nao implementado.
Capital0=50000# Captal teorico.
Ambidente=Variaveis['Ambi_teste']


df = pd.read_csv(lista+'\\'+'@SIRI.csv', sep=',', header=0, nrows=10)
#df = pd.read_csv(lista+'\\'+'@SIRI.csv', sep=',', header=0,skiprows=range(1, 10), nrows=10000)

df = pd.read_csv(lista+'\\'+'@SIRI.csv', sep=',', header=0, nrows=10)
df['Date'] = df['Date'].astype('object')
df['EXPIR_DATE'] = df['EXPIR_DATE'].astype('object')
df['dia_1'] = df['dia_1'].astype('object')
df['dia_2'] = df['dia_2'].astype('object')
df['dia_3'] = df['dia_3'].astype('object')
df['dia_4'] = df['dia_4'].astype('object')
df['dia_5'] = df['dia_5'].astype('object')
df['dia_F1'] = df['dia_F1'].astype('object')
df['dia_F2'] = df['dia_F2'].astype('object')
df['dia_F3'] = df['dia_F3'].astype('object')
df['Security'] =df['Security'].astype('object')

dt = df.dtypes.to_dict()
### --- Random selection of date time.
#
#
#jj={}
#ListaAtivos=[f for f in listdir(lista) if '.csv' in f ]
##ativo=ListaAtivos[20]
#
#for ativo in ListaAtivos:
#    Mercado=dsk.read_csv(lista+'\\'+ativo,sep=',', dtype=dt)#,dtype=dtypes)
#    xx=Mercado.compute()
#    xx.index=range(len(xx))
#    xxd=xx[['Date']]
#    xxd['in']=list(xx.index)
#    g=xxd.groupby(['Date']).max()
#    g['Date']=g.index
#    jj.update({ativo:g.values.tolist()})
#
#
#dir(jj)
#len(jj.keys())
#
#with open(lista+'\\0mapa\\mapa.json', 'w') as fp:
#    json.dump(jj, fp)
#
#
#





def ExpirationFilter(numPeriodos,Refp,Mercado1c):
    '''
       Dentro de uma base de opções filtra conforme os vencimentos.
           numPeriodos: Numero de periodos, para caso de usar mais de um vencimento. Padrão 0
           Refp: Numero de vencimentos. Padrão 1 para proximo vencimento.
           Mercado1c: pandas - base com as opçoes de mesmo tipo
       Retrona:Base filtrada pelo vencimento
    '''
    temp=list(set(Mercado1c.EXPIR_DATE.values.astype('datetime64')))
    temp.sort()
    diaRef=temp[Refp-1]
    if numPeriodos==0:
        k=Mercado1c[Mercado1c.EXPIR_DATE.astype('datetime64')==diaRef]
    else :
        k=Mercado1c[Mercado1c.EXPIR_DATE.astype('datetime64')<=diaRef]
    return(k)

###--------------------------------------------------------


import time
## --- Random selection of date time.
jj=open(lista+'\\0mapa\\mapa.json')
jj=jj.read()
jj=json.loads(jj)
ativo=OP.ativo

AtivDate=(list(jj[ativo]))

#date='2011-06-10'

#date=None
#if date==None:
#    d=Refp*dmais
#    nu=np.random.choice(range(len(AtivDate[:-2])), 1)[0]
#    DataRef=np.array([AtivDate[nu],AtivDate[(d+nu if d+nu<len(AtivDate) else len(AtivDate)-1)]])
#    DataRef[:,1]=DataRef[:,1].astype('datetime64')
#else:
#    d=Refp*dmais
#    nu=pd.DataFrame(AtivDate)[pd.DataFrame(AtivDate)[1]==date].index[0]
#    DataRef=np.array([AtivDate[nu],AtivDate[(d+nu if d+nu<len(AtivDate) else len(AtivDate)-1)]])
#    DataRef[:,1]=DataRef[:,1].astype('datetime64')
#

DataRef=OP.DataRef

inic=int(DataRef[1][0])
fim=int(DataRef[0][0])


xx=time.time()
Mercado1= pd.read_csv(lista+'\\'+ativo,sep=',', dtype=dt,header=0,skiprows=range(1, inic-20),nrows=fim-inic)
Mercado1=Mercado1[(Mercado1.Date.astype('datetime64')>=DataRef[0,1])&(Mercado1[['CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3',]].isnull().any(1)==False)]
print((time.time()-xx))


#def Tranfs(Mercado1):
    Mercado1c=Mercado1[Mercado1.PUT_CALL=='Call']
    Mercado1p=Mercado1[Mercado1.PUT_CALL!='Call']            

    temp=[np.sort(list(set(Mercado1p.EXPIR_DATE.values.astype('datetime64'))))[np.isin(np.sort(list(set(Mercado1p.EXPIR_DATE.values.astype('datetime64')))),(np.sort(list(set(Mercado1c.EXPIR_DATE.values.astype('datetime64'))))))],
            np.sort(list(set(Mercado1c.EXPIR_DATE.values.astype('datetime64'))))[np.isin(np.sort(list(set(Mercado1c.EXPIR_DATE.values.astype('datetime64')))),(np.sort(list(set(Mercado1p.EXPIR_DATE.values.astype('datetime64'))))))]]
        
    temp=np.sort(temp[np.argmax([len(temp[0]),len(temp[1])])])

        
    Mercado2c=ExpirationFilter(0,Refp,Mercado1c,temp)
    Mercado2p=ExpirationFilter(0,Refp,Mercado1p,temp)

    listacall=DeltaStreikFilter(numOpc,Mercado2c,Mercado2p,DataRef,temp)
    listaput=DeltaStreikFilter(numOpc,Mercado2p,Mercado2c,DataRef,temp)

    Mercado2c=Mercado1c[Mercado1c.Security.isin(listacall)]
    Mercado2p=Mercado1p[Mercado1p.Security.isin(listaput)]

    ## -- geting the table with the observed assets
    kdate=list(set(list(set(Mercado2c.Date.astype('datetime64')))+list(set(Mercado2p.Date.astype('datetime64')))))
    kdate.sort()
    #        
    show=[]
    d=kdate[3]
    for d in kdate:
        Mercado3c=Mercado2c[Mercado2c.Date.astype('datetime64')==d].sort_values('Security')      
        Mercado3c=Mercado3c[Mercado3c.dia_1!=Mercado3c.Date]
        Mercado3c=Mercado3c.reindex(Mercado3c.STRIKE_PRC.sort_values().index)
        Mercado3c=Mercado3c.append(pd.DataFrame(np.zeros(((np.max([0,numOpc-len(Mercado3c)])),len(Mercado3c.columns))),columns=Mercado3c.columns))

        Mercado3p=Mercado2p[Mercado2p.Date.astype('datetime64')==d].sort_values('PriceToStrike0')
        Mercado3p=Mercado3p[Mercado3p.dia_1!=Mercado3p.Date]
        Mercado3p=Mercado3p.reindex(Mercado3p.STRIKE_PRC.sort_values().index)
        Mercado3p=Mercado3p.append(pd.DataFrame(np.zeros(((np.max([0,numOpc-len(Mercado3c)])),len(Mercado3p.columns))),columns=Mercado3p.columns))
        show=show+[Mercado3c.append(Mercado3p)]
    
    dataspace=kdate
    StateList=show
    StateCont=0
    last_Q=np.zeros(len(StateList[0]))
    return(StateList)


#Mercado1.tail(20)[['PrecoFut1','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
show[-2][['Security','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]
show[-5][['Security','Date','EXPIR_DATE','STRIKE_PRC','CLOSE_ATIVO']]

Mercado0=Tranfs(Mercado1)

#Mercado0=Tranfs(Mercado0)
diaRef=temp[Refp-1]

