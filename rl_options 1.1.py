"""
Created on Thu Feb 15 13:40:03 2018

@author: U6035631
"""
'''
                            Compilação de dados de opções
                            estruturando problema para RL.
'''
## ----- python SVM options
import pandas as pd
import random
import getpass
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import eikon as ek

USA=['AAP','BAC','IWM',
       'JPM','QQQ','SPY']
       
USA=['AAP','BAC','IWM',
     'JPM','QQQ','SPY',
     'USO','GE']

BR=['BBA','BRF','BVM','CIE','BBD',
            'CSN','EQT','FIB','GGB',
            'ITU','KRO','PET','USI','VAL']

#-Variaveis relevantes para decisao.
X=['HIGH', 'CLOSE_x', 'LOW', 'OPEN', 'VOLUME', 'HIGH_1', 'CLOSE_1', 'LOW_1', 'OPEN_1', 'VOLUME_1', 'HIGH_2', 'CLOSE_2', 'LOW_2', 'OPEN_2', 'VOLUME_2',
 'HIGH_3', 'CLOSE_3', 'LOW_3', 'OPEN_3', 'VOLUME_3', 'HIGH_4', 'CLOSE_4', 'LOW_4', 'OPEN_4', 'VOLUME_4', 'HIGH_5', 'CLOSE_5', 'LOW_5',
 'OPEN_5', 'VOLUME_5', 'CLOSE_ATIVO', 'Retorno_ATIVO', 'Rlog_ATIVO', 'VolatH_ATIVO', 'CLOSE1_ATIVO', 'Retorno1_ATIVO', 'Rlog1_ATIVO',
 'VolatH1_ATIVO', 'CLOSE2_ATIVO', 'Retorno2_ATIVO', 'Rlog2_ATIVO', 'VolatH2_ATIVO', 'CLOSE3_ATIVO', 'Retorno3_ATIVO', 'Rlog3_ATIVO',
 'VolatH3_ATIVO', 'CLOSE4_ATIVO', 'Retorno4_ATIVO', 'Rlog4_ATIVO', 'VolatH4_ATIVO', 'FreeRiskCLOSE', 'FreeRiskCLOSE_1', 'FreeRiskCLOSE_2', 'FreeRiskCLOSE_3',
 'FreeRiskCLOSE_4', 'Retonro0', 'Retonro1', 'Retonro2', 'Retonro3', 'Retonro4', 'PriceToStrike0', 'PriceToStrike1', 'PriceToStrike2',
 'PriceToStrike3', 'dayTOexp', 'DeltaC', 'DeltaP', 'EtasC', 'EtasP', 'GammaC', 'GammaP', 'RhoC', 'RhoP', 'ThetaC', 'ThetaP', 'ValorC', 'ValorP', 'VegaC', 'VegaP',
 'BLAKvolat1', 'BLAKvolat2', 'BLAKvolat3', 'BLAKvolat4', 'BLAKvolat5', 'ImplDeltaC1', 'ImplDeltaP1', 'ImplEtasC1',
 'ImplEtasP1', 'ImplGammaC1', 'ImplGammaP1', 'ImplRhoC1', 'ImplRhoP', 'ImplThetaC1', 'ImplThetaP1', 'ImplValorC1',
 'ImplValorP1', 'ImplVegaC1', 'ImplVegaP1', 'ImplDeltaC2', 'ImplDeltaP2', 'ImplEtasC2', 'ImplEtasP2', 'ImplGammaC2',
 'ImplGammaP2', 'ImplRhoC2', 'ImplThetaC2', 'ImplThetaP2', 'ImplValorC2', 'ImplValorP2', 'ImplVegaC2', 'ImplVegaP2',
 'ImplDeltaC3', 'ImplDeltaP3', 'ImplEtasC3', 'ImplEtasP3', 'ImplGammaC3', 'ImplGammaP3', 'ImplRhoC3', 'ImplThetaC3',
 'ImplThetaP3', 'ImplValorC3', 'ImplValorP3', 'ImplVegaC3', 'ImplVegaP3', 'ImplDeltaC4', 'ImplDeltaP4', 'ImplEtasC4',
 'ImplEtasP4', 'ImplGammaC4', 'ImplGammaP4', 'ImplRhoC4', 'ImplThetaC4', 'ImplThetaP4', 'ImplValorC4', 'ImplValorP4',
 'ImplVegaC4', 'ImplVegaP4', 'ImplDeltaC5', 'ImplDeltaP5', 'ImplEtasC5', 'ImplEtasP5', 'ImplGammaC5', 'ImplGammaP5',
 'ImplRhoC5', 'ImplThetaC5', 'ImplThetaP5', 'ImplValorC5', 'ImplValorP5', 'ImplVegaC5', 'ImplVegaP5']
##########################

#X=['dayTOexp',
#    'STRIKE_PRC',
#    'PriceToStrike0',
#    'Retonro0', 'Retonro1', 'Retonro2',
#    'VolatH_ATIVO', 'VolatH1_ATIVO', 'VolatH2_ATIVO', 'VolatH3_ATIVO','VolatH4_ATIVO',
#    'CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3','CLOSE_4',
#    'Rlog_ATIVO','Rlog1_ATIVO', 'Rlog2_ATIVO', 'Rlog3_ATIVO',
#    'BLAKvolat1', 'BLAKvolat2', 'BLAKvolat3','BLAKvolat4','BLAKvolat5']
#
## Definicao de opjetivo

TotalBC=pd.read_csv('C://Users//u6035631//Dropbox//Opcoes//csv//RL_optionsBasePrototipoCall.csv')



def Retorno(x,w,Capital):
    operado=Capital*0.95*w
    Q=(round(((operado/x['CLOSE_x'])/100),0)*100)
    operado=x['CLOSE_x']*Q
#    Q=operado/x['CLOSE_x']
    exe=[i if i>0 else 0 for i in (x['Value']-x['STRIKE_PRC']-x['CLOSE_x'])]*Q
    transaction=sum(abs(operado)*0.0025)
    R=(sum(exe)-sum(operado))-transaction
#    (sum(exe)-sum(operado))/sum(operado)
## Rt d+f
#    sum(x['PrecoFut3']*w)
    return(R)

Capital=25000
k=np.random.choice(len(list(set(TotalBC['Codiigo']))),1)[0]
x=TotalBC[TotalBC['Codiigo']==list(set(TotalBC['Codiigo']))[500]]
#w=np.random.lognormal(0.0, 0.015, size=len(x))
w=np.random.normal(0.0, 0.15, size=len(x))
w=w*min(w)
w=w/sum(w)
Retorno(x,w,Capital)


