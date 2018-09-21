# -*- coding: utf-8 -*-
"""
"""
#import dask.dataframe as dsk
from os import listdir
import numpy as np
import pandas as pd
import json
from AuxiliarFunctions import *
import MySQLdb as mariadb


class OptionsBases:
    def __init__(self,lista,numOpc,Refp,numPeriodos,callput,Ambiente,Capital0,dmais):
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
                self.dmais: numero de dias da janela de captura do arquivo
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
        self.dmais=dmais
        self.Transform=None

    def StartPred(self,Ativ,date=None):

        df = pd.read_csv(self.lista+'\\'+'@AAPL.csv', sep=',', header=0, nrows=10)
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
        self.AtivDate=(list(jj[self.ativo]))

        tamanho=0
        while self.numOpc*2!=tamanho:

            if date==None:
                d=self.Refp*self.dmais
                nu=np.random.choice(range(len(self.AtivDate[:-2])), 1)[0]
                DataRef=np.array([self.AtivDate[nu],self.AtivDate[(d+nu if d+nu<len(self.AtivDate) else len(self.AtivDate)-1)]])
                DataRef[:,1]=DataRef[:,1].astype('datetime64')
            else:
                d=self.Refp*self.dmais
                nu=pd.DataFrame(self.AtivDate)[pd.DataFrame(self.AtivDate)[1]==date].index[0]
                DataRef=np.array([self.AtivDate[nu],self.AtivDate[(d+nu if d+nu<len(self.AtivDate) else len(self.AtivDate)-1)]])
                DataRef[:,1]=DataRef[:,1].astype('datetime64')

            inic=int(DataRef[1][0])
            fim=int(DataRef[0][0])
            self.DataRef=DataRef

#            Mercado1= pd.read_csv(self.lista+'\\'+self.ativo,sep=',', dtype=dt,header=0,skiprows=range(1, inic-20),nrows=fim-inic)
#            Mercado1=Mercado1[(Mercado1.Date.astype('datetime64')>=self.DataRef[0,1])&(Mercado1[['CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3',]].isnull().any(1)==False)]

            Mercado1=SQLExtrat(self.ativo,self.DataRef,self.Ambiente,self.df)
           	
            self.dataspace,self.StateList,self.StateCont,self.last_Q=DataEditing(Mercado1,self.Refp,self.numOpc,self.DataRef)
            
            if len(self.StateList)>2 or not (sum(self.StateList[0].Security==0)<12 and not len(self.StateList)>2):
                tamanho=len(self.StateList[0])
                print('Change the days')
                
        return(TratamentoStado(self.Ambiente,self.StateList[self.StateCont]))

        
    def Start(self):
        '''
            Reinicia a base coletando novos dados de forma aleatoria e formadando de forma adequada.
                Inicia todas as variaveis e carrega todo o novo ambiente.
            Retorna: vetor com todas as variaveis importante para a Açao.
        '''
        df = pd.read_csv(self.lista+'\\'+'@AAPL.csv', sep=',', header=0, nrows=10)
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
        date=None
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
            self.AtivDate=(list(jj[self.ativo]))

            d=self.Refp*self.dmais
            nu=np.random.choice(range(len(self.AtivDate[:-2])), 1)[0]
            DataRef=np.array([self.AtivDate[nu],self.AtivDate[(d+nu if d+nu<len(self.AtivDate) else len(self.AtivDate)-1)]])
            DataRef[:,1]=DataRef[:,1].astype('datetime64')

            inic=int(DataRef[1][0])
            fim=int(DataRef[0][0])

            self.DataRef=DataRef

#            Mercado1= pd.read_csv(self.lista+'\\'+self.ativo,sep=',', dtype=dt,header=0,skiprows=range(1, inic-20),nrows=fim-inic)
#            Mercado1=Mercado1[(Mercado1.Date.astype('datetime64')>=self.DataRef[0,1])&(Mercado1[['CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3',]].isnull().any(1)==False)]

            Mercado1=SQLExtrat(self.ativo,self.DataRef,self.Ambiente,self.df)

            if len(Mercado1)>2 and sum(Mercado1['CLOSE_ATIVO'].isnull())!=len(Mercado1):
                self.dataspace,self.StateList,self.StateCont,self.last_Q=DataEditing(Mercado1,self.Refp,self.numOpc,self.DataRef)
               
                if len(self.StateList)>2:
                    tamanho=len(self.StateList[0])                   
                    
                    if (sum(self.StateList[0].Security==0)<12):
                        tamanho=len(self.StateList[0])
                    else:
                        tamanho=0
                else:
                    tamanho=0

        return(TratamentoStado(self.Ambiente,self.StateList[self.StateCont]))

#
    def Action(self,W):
        '''
            Calcula o retorno de uma açao em um momento de mercado e passa toda a lista apra o proximo momento de mercado.
            Retorna:Estado seguinte, retrono obtigo e statos.
        '''
        StateRetorno,self.Capital1,_,self.last_Q,self.Vector=Retorno(self.StateList[self.StateCont],self.Capital1,W,self.last_Q)
        self.last_Q=self.last_Q.values
        self.HiscoricBalance=self.HiscoricBalance+[[self.dataspace[self.StateCont],StateRetorno,self.Capital1,_]]
        self.StateCont+=1

        if self.StateCont>=len(self.StateList):
                self.statos=True

        elif((self.Capital1<=self.Capital0*0.25) or (self.StateCont>=(len(self.StateList)-2))):
                self.statos=True
#	            self.StateCont=0
        return(TratamentoStado(self.Ambiente,self.StateList[self.StateCont]),_,self.statos,self.Vector)

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