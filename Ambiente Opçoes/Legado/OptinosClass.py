# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 15:03:35 2018
*** Classe de opçoes bases
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



def randomAction(k):
    '''
       Calcula Açao randomica, vetor de pesos para ativo.
           k: pandas dataframe do stado.
    '''
    w=np.random.normal(0.1, 0.06, size=len(k))
    w=w*(w).mean()
    W=w/sum(w)
    return(W)


def TratamentoStado(LISTA,BASE):
    '''
        Definir anbiente de variaveis a seremusadas.
        LISTA:Lista de variaveis relevantes
        BASE:Base de referncia (estado)
            Tratamento de NAN.
            Normalizaçao dosdados.
            Retorna tabela como vertor
    '''
    k=BASE[LISTA].fillna(0)
    scal=StandardScaler()
    k=scal.fit_transform(k)
    return(np.array(k).reshape(len(k)*len(LISTA),1))


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
    diaRef=temp[Refp]
    if numPeriodos==0:
        k=Mercado1c[Mercado1c.EXPIR_DATE.astype('datetime64')==diaRef]
    else :
        k=Mercado1c[Mercado1c.EXPIR_DATE.astype('datetime64')<=diaRef]
    return(k)


def DeltaStreikFilter(numOpc,Mercado2c,Mercado2p,DataRef):
    '''
       Dentro de uma base de opções filtra as que tem liquides, (Opões mais proximas o dinheiro) .
           numOpc: Numero de opções a ser coletado.
           Mercado2c: pandas - base com as opçoes de mesmo tipo
           DataRef: Referencia de dia em que o mercado foi observado para a filtragem.
        Retrona:lista de ativos que fazem parte do grupo observado.
    '''
    temp=list(set(Mercado2c.EXPIR_DATE.values.astype('datetime64')))
    temp.sort()
    datemin=np.array(Mercado2c.append(Mercado2p).Date.min())
    tempRes=[]
    for exp in temp:
        temps=Mercado2c[(Mercado2c.EXPIR_DATE.astype('datetime64')==exp)&(Mercado2c['Date'].astype('datetime64')==datemin.astype('datetime64'))]
        temps=temps.reindex(temps.PriceToStrike0.abs().sort_values().index)
        temps=temps[:numOpc]['Security'].values
        tempRes=tempRes+(list(temps))
    return(list(set(tempRes)))


def Retorno(StateList,Capital0,W,last_Q):
    '''
       Calcula retorno apra um vetro de pesos para um dia de mercado.
           Capital0: Numero de opções a ser coletado.
           StateList: pandas - base com as opçoes de mesmo tipo
           W: Referencia de dia em que o mercado foi observado para a filtragem.
        retorna: Retorno gerado$, Saldo, Retorno gerado%
    '''
    operado=Capital0*0.95*W
    Quantidade=(round(((operado/StateList['CLOSE_x'])/100),0)*100)
    Quantidade[np.isinf(Quantidade)] = 0
    exe=StateList['CLOSE_x']*StateList[['PrecoFut1','PrecoFut2','PrecoFut3']].mean(1)*Quantidade
    exe[np.isnan(exe)] = 0

    QuantidadeMov=Quantidade.values-last_Q
    Cost=sum(abs(QuantidadeMov*StateList['CLOSE_x'])*0.0025)
    R=sum(exe)-Cost
    return(R,Capital0+R,R/Capital0,Quantidade)


class OptinsBases:
    def __init__(self,lista,numOpc,Refp,numPeriodos,callput,Ambiente,Capital0):
        '''
            Classe que define ambiente de negoriaçao das opçoes.
            Variaveis: INICIAIS
                self.lista:         Path para arquivos das opçoes
                self.numOpc:        Numero de opções a serem trabalhadas por ciclo e por tipo(call e put).
                self.Refp:          Numero de referencia a ser trabalhado.
                self.numPeriodos: 0 -NÃO IMPLEMENTADO
                self.callput:0      -NÃO IMPLEMENTADO
                self.Capital0:      Valor utilizado como patrimonio para nerociação
                self.Ambiente:      Lista com nome de variaveis relevantes para decisão.

            Variaveis: INSPEÇAO.
                self.StateList: Lista em ordem crecente por dia de situaçoes de mercado para uma dada epoca.
                self.StateCont: Referencia de ponto na lista de estados de uma epoca
                self.statos:    Referencia de fim de epoca, ou interupçao por perda de dinheiro.
                self.Capital1:  Total da carteira no momento atual
                self.dataspace: Dias representados na base analisada
                self.HiscoricBalance: [None,None,Capital0,None]/ dias,retorno$,Total Carteira,retorno%
        '''
        self.lista,self.numOpc,self.Refp,self.numPeriodos,self.callput,self.Capital0=lista,numOpc,Refp,numPeriodos,callput,Capital0
        self.Ambiente=Ambiente
        self.StateList=None
        self.StateCont=0
        self.statos=False
        self.Capital1=self.Capital0
        self.dataspace=None
        self.HiscoricBalance=[None,None,Capital0,None]
        self.ListaAtivos=[f for f in listdir(self.lista) if '.csv' in f ]
    
    def StartPred(self,Ativ,date=None):

        df = pd.read_csv(self.lista+'\\'+'@SIRI.csv', sep=',', header=0, nrows=10)
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
        self.df=df
        #            dtc = list(df.columns)
        dt = df.dtypes.to_dict()
        ## -- Random selection of asset.
        self.ativo=Ativ
        self.statos=False
        self.StateCont=0
        
        ## --- Random selection of date time.
        jj=open(self.lista+'\\0mapa\\mapa.json')
        jj=jj.read()
        jj=json.loads(jj)
        self.AtivDate=list(set(jj[self.ativo]))
        print(self.ativo+" "+min(self.AtivDate)+" - "+max(self.AtivDate))

        if date==None:
            DataRef=np.random.choice((np.sort(np.array(list(set(jj[self.ativo]))).astype('datetime64'))[:-2]), 1)[0]
            DataRef=DataRef.astype('datetime64')           
        else:
            DataRef=np.array(date).astype('datetime64')           
        
        Mercado=dsk.read_csv(self.lista+'\\'+self.ativo,sep=',', dtype=dt)#,dtype=dtypes)
        Mercado1=Mercado[(Mercado.Date.astype('datetime64')>=DataRef)&(Mercado[['CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3',]].isnull().any(1)==False)].compute()

        Mercado1c=Mercado1[Mercado1.PUT_CALL=='Call']
        Mercado1p=Mercado1[Mercado1.PUT_CALL!='Call']
    
        Mercado2c=ExpirationFilter(0,self.Refp,Mercado1c)
        Mercado2p=ExpirationFilter(0,self.Refp,Mercado1p)
        
        listacall=DeltaStreikFilter(self.numOpc,Mercado2c,Mercado2p,DataRef)
        listaput=DeltaStreikFilter(self.numOpc,Mercado2p,Mercado2c,DataRef)
    ## -- geting the table with the observed assets
        Mercado2c=Mercado1c[Mercado1c.Security.isin(listacall)]
        Mercado2p=Mercado1p[Mercado1p.Security.isin(listaput)]

        kdate=list(set(list(set(Mercado2c.Date.astype('datetime64')))+list(set(Mercado2p.Date.astype('datetime64')))))
        kdate.sort()
#        
        show=[]
        for d in kdate:
            Mercado3c=Mercado2c[Mercado2c.Date.astype('datetime64')==d]#.sort_values('PriceToStrike0')
            Mercado3c=Mercado3c[Mercado3c.dia_1!=Mercado3c.Date]
            Mercado3c=Mercado3c.append(pd.DataFrame(np.zeros(((self.numOpc-len(Mercado3c)),len(Mercado3c.columns))),columns=Mercado3c.columns))
            Mercado3p=Mercado2p[Mercado2p.Date.astype('datetime64')==d]#.sort_values('PriceToStrike0')
            Mercado3p=Mercado3p[Mercado3p.dia_1!=Mercado3p.Date]
            Mercado3p=Mercado3p.append(pd.DataFrame(np.zeros(((self.numOpc-len(Mercado3p)),len(Mercado3p.columns))),columns=Mercado3p.columns))
            show=show+[Mercado3c.append(Mercado3p)]

        self.dataspace=kdate
        self.StateList=show
        self.StateCont=0
        self.last_Q=np.zeros(len(self.StateList[0]))

        if (sum(self.StateList[0].Security==0)<12 or len(self.StateList)>2) and (((self.StateList[0]['Date']==self.StateList[0]['EXPIR_DATE']).sum()>0) or len(self.StateList)>2):
            print('Change the days')
            
        return(TratamentoStado(self.Ambiente,self.StateList[self.StateCont]))

        
    def Start(self):
        '''
            Reinicia a base coletando novos dados de forma aleatoria e formadando de forma adequada.
                Inicia todas as variaveis e carrega todo o novo ambiente.
            Retorna: vetor com todas as variaveis importante para a Açao.
        '''
        df = pd.read_csv(self.lista+'\\'+'@SIRI.csv', sep=',', header=0, nrows=10)
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
        self.df=df
    #            dtc = list(df.columns)
        dt = df.dtypes.to_dict()
        ## -- Random selection of asset.
        tamanho=0
        while self.numOpc*2!=tamanho:
            self.ativo=np.random.choice(self.ListaAtivos, 1)[0]
            self.statos=False
            self.StateCont=0

            ## --- Random selection of date time.
            jj=open(self.lista+'\\0mapa\\mapa.json')
            jj=jj.read()
            jj=json.loads(jj)
            DataRef=np.random.choice((np.sort(np.array(list(set(jj[self.ativo]))).astype('datetime64'))[:-2]), 1)[0]
            DataRef=DataRef.astype('datetime64')           
            
            #--Filtro de liquides
            #Mercado[Mercado[['CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3',]].isnull().any(1)==False]

            Mercado=dsk.read_csv(self.lista+'\\'+self.ativo,sep=',', dtype=dt)#,dtype=dtypes)
            Mercado1=Mercado[(Mercado.Date.astype('datetime64')>=DataRef)&(Mercado[['CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3',]].isnull().any(1)==False)].compute()
    
            Mercado1c=Mercado1[Mercado1.PUT_CALL=='Call']
            Mercado1p=Mercado1[Mercado1.PUT_CALL!='Call']
        
            Mercado2c=ExpirationFilter(0,self.Refp,Mercado1c)
            Mercado2p=ExpirationFilter(0,self.Refp,Mercado1p)
            
            listacall=DeltaStreikFilter(self.numOpc,Mercado2c,Mercado2p,DataRef)
            listaput=DeltaStreikFilter(self.numOpc,Mercado2p,Mercado2c,DataRef)
        ## -- geting the table with the observed assets
            Mercado2c=Mercado1c[Mercado1c.Security.isin(listacall)]
            Mercado2p=Mercado1p[Mercado1p.Security.isin(listaput)]
    
            kdate=list(set(list(set(Mercado2c.Date.astype('datetime64')))+list(set(Mercado2p.Date.astype('datetime64')))))
            kdate.sort()
    #        
            show=[]
            for d in kdate:
                Mercado3c=Mercado2c[Mercado2c.Date.astype('datetime64')==d].sort_values('PriceToStrike0')
                Mercado3c=Mercado3c.append(pd.DataFrame(np.zeros(((self.numOpc-len(Mercado3c)),len(Mercado3c.columns))),columns=Mercado3c.columns))
                Mercado3p=Mercado2p[Mercado2p.Date.astype('datetime64')==d].sort_values('PriceToStrike0')
                Mercado3p=Mercado3p.append(pd.DataFrame(np.zeros(((self.numOpc-len(Mercado3p)),len(Mercado3p.columns))),columns=Mercado3p.columns))
                show=show+[Mercado3c.append(Mercado3p)]

            self.dataspace=kdate
            self.StateList=show
            self.StateCont=0
            self.last_Q=np.zeros(len(self.StateList[0]))

            if (sum(self.StateList[0].Security==0)<12 or len(self.StateList)>2) and (((self.StateList[0]['Date']==self.StateList[0]['EXPIR_DATE']).sum()>0) or len(self.StateList)>2):
                tamanho=len(self.StateList[0])
            
        return(TratamentoStado(self.Ambiente,self.StateList[self.StateCont]))

#
    def Action(self,W):
        '''
            Calcula o retorno de uma açao em um momento de mercado e passa toda a lista apra o proximo momento de mercado.
            Retorna:Estado seguinte, retrono obtigo e statos.
        '''
        StateRetorno,self.Capital1,_,self.last_Q=Retorno(self.StateList[self.StateCont],self.Capital1,W,self.last_Q)
        self.last_Q=self.last_Q.values
        self.HiscoricBalance=self.HiscoricBalance+[[self.dataspace[self.StateCont],StateRetorno,self.Capital1,_]]
        self.StateCont+=1

        if self.StateCont>=len(self.StateList):
                self.statos=True

        elif((self.Capital1<=self.Capital0*0.2) or (self.StateCont>=(len(self.StateList)-2)) or (self.StateList[self.StateCont]['Date']==self.StateList[self.StateCont]['EXPIR_DATE']).sum()<0):
                self.statos=True
    #            self.StateCont=0
        return(TratamentoStado(self.Ambiente,self.StateList[self.StateCont]),_,self.statos)

    def Reset(self):
        self.StateCont=0
        self.statos=False
        self.Capital1=self.Capital0
        self.last_Q=np.zeros(len(self.StateList[0]))
        self.HiscoricBalance=[None,None,self.Capital0,None]
        print('Done')
                
###---
###---
###---Fim da classe
###---
###---


###---------------------   Definindo lista de variaveis de decisão ------- ###
Variaveis={} # DICIONARIO QUE COMPILA DIFERENTES SELEÇOES DE VARIAVEIS.
## --- contem dados sobre o futuro para que ele aprenda rapido. Mas irrealista.
Variaveis.update({"Ambi_teste":['STRIKE_PRC', 'DeltaP', 'EtasC', 'CLOSE_F1', 'CLOSE_F2', 'ImplDeltaC2', 'CLOSE_F3',
                             'ImplDeltaP1', 'ImplDeltaP2', 'ImplDeltaP3', 'ImplEtasC2', 'ImplGammaC3', 'CLOSE_1F_ATIVO', 'ImplThetaC1', 'ImplThetaP3',
                             'ImplValorC1', 'ImplValorC2', 'ImplValorC3', 'BLAKvolat1', 'BLAKvolat2', 'BLAKvolat3', 'Retorno_2F_ATIVO', 'RhoC',
                             'Rlog1_ATIVO', 'Rlog2_ATIVO', 'Rlog3_ATIVO', 'ThetaC','ValorC', 'VegaC', 'VegaP', 'VolatH_1F_ATIVO',
                             'VolatH1_ATIVO', 'VolatH2_ATIVO','VolatH3_ATIVO', 'BLAKvolat_F3']})


## --- contem dados sobre o tiradas informaçossobre o futuro.
Variaveis.update({"Ambi_A0":['HIGH', 'CLOSE_x', 'LOW', 'OPEN', 'VOLUME', 'HIGH_1', 'CLOSE_1', 'LOW_1', 'OPEN_1', 'VOLUME_1', 'HIGH_2', 'CLOSE_2', 'LOW_2', 'OPEN_2', 'VOLUME_2',
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
                             'ImplRhoC5', 'ImplThetaC5', 'ImplThetaP5', 'ImplValorC5', 'ImplValorP5', 'ImplVegaC5', 'ImplVegaP5']})

## --- contem poiucos dados.
Variaveis.update({"Ambi_A1":['dayTOexp','STRIKE_PRC',
                            'PriceToStrike0',
                            'Retonro0', 'Retonro1', 'Retonro2',
                            'VolatH_ATIVO', 'VolatH1_ATIVO', 'VolatH2_ATIVO', 'VolatH3_ATIVO','VolatH4_ATIVO',
                            'CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3','CLOSE_4',
                            'Rlog_ATIVO','Rlog1_ATIVO', 'Rlog2_ATIVO', 'Rlog3_ATIVO',
                            'BLAKvolat1', 'BLAKvolat2', 'BLAKvolat3','BLAKvolat4','BLAKvolat5']})

    

###---
###---
###---
###---
###---





