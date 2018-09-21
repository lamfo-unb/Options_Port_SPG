# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
import matplotlib.pyplot as plt
import MySQLdb as mariadb


def StrategiFlow(StateList,action):
    li= np.linspace((StateList['CLOSE_ATIVO'].values[0])*0.5, (StateList['CLOSE_ATIVO'].values[0])*1.5, 100)
    ret=[]
    for i in range(len(action)):
        k,j=StateList[['STRIKE_PRC','CLOSE_x','PUT_CALL']].values[i],action[i]
        if j>0:
            temp=np.array(li-k[0]-k[1] if k[2]=='Call' else -li+k[0]-k[1])
            temp[temp<=-k[1]]=-k[1]
            ret=ret+[temp*j]
        else:
            temp=np.array(-li+k[0]+k[1] if k[2]=='Call' else +li-k[0]+k[1])
            temp[temp>=k[1]]=k[1]
            ret=ret+[temp*-j]

    ret=pd.DataFrame(ret)
    ret.column=li
    plt.figure(5)
    plt.clf()
    plt.title('Acompanhamento de PL')
    plt.xlabel('stok')
    plt.ylabel('$')
    plt.plot(li,ret.sum(0).values)
    plt.pause(0.00001)  # pause a bit so that plots are u
    #ret
    return(plt)


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
    Quantidade[np.isnan(Quantidade)] = 0
#    exe=StateList['CLOSE_x']*StateList[['PrecoFut1','PrecoFut2','PrecoFut3']].mean(1)*Quantidade
    exe=StateList['CLOSE_x']*StateList[['PrecoFut1']].mean(1)*Quantidade
    exe[np.isnan(exe)] = 0

    QuantidadeMov=Quantidade.values-last_Q
    Cost=sum(abs(QuantidadeMov*StateList['CLOSE_x'])*0.0025)
    R=sum(exe)-Cost

    par=(StateList['CLOSE_x']*Quantidade)/(sum(StateList['CLOSE_x']*Quantidade))
    part=np.array(par*StateList[['PrecoFut1','PrecoFut2','PrecoFut3']].mean(1).values)
    zero=[np.isnan(part)]
    part[zero]=0

    return(R,Capital0+R,R/Capital0,Quantidade,part)


def RetornoT(StateList,StateList1,Capital0,W,last_Q):
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
    Quantidade[np.isnan(Quantidade)] = 0
#    exe=StateList['CLOSE_x']*StateList[['PrecoFut1']].mean(1)*Quantidade
#    exe[np.isnan(exe)] = 0
    exe=(StateList1['CLOSE_ATIVO']- StateList1['STRIKE_PRC']).values*Quantidade.values-StateList['CLOSE_x']*Quantidade
    exe[np.isnan(exe)] = 0

    QuantidadeMov=Quantidade.values-last_Q
    Cost=sum(abs(QuantidadeMov*StateList['CLOSE_x'])*0.0025)

    R=sum(exe)-Cost
    return(R,Capital0+R,R/Capital0,Quantidade,StateList['PrecoFut1'].values)

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
    return(np.array(k).reshape(len(k)*len(LISTA),))


def ExpirationFilter(numPeriodos,Refp,Mercado1c,temp):
    '''
       Dentro de uma base de opções filtra conforme os vencimentos.
           numPeriodos: Numero de periodos, para caso de usar mais de um vencimento. Padrão 0
           Refp: Numero de vencimentos. Padrão 1 para proximo vencimento.
           Mercado1c: pandas - base com as opçoes de mesmo tipo
       Retrona:Base filtrada pelo vencimento
    '''
    if len(temp)<Refp:
        diaRef=temp[0]
    else:
        diaRef=temp[Refp-1]
    if numPeriodos==0:
        k=Mercado1c[Mercado1c.EXPIR_DATE.astype('datetime64')==diaRef]
    else :
        k=Mercado1c[Mercado1c.EXPIR_DATE.astype('datetime64')<=diaRef]
    return(k)


def DeltaStreikFilter(numOpc,Mercado2c,Mercado2p,DataRef,temp):
    '''
       Dentro de uma base de opções filtra as que tem liquides, (Opões mais proximas o dinheiro) .
           numOpc: Numero de opções a ser coletado.
           Mercado2c: pandas - base com as opçoes de mesmo tipo
           DataRef: Referencia de dia em que o mercado foi observado para a filtragem.
        Retrona:lista de ativos que fazem parte do grupo observado.
    '''
    datemin=np.array([Mercado2c.append(Mercado2p).Date.min()]).astype('datetime64')
    tempRes=[]
    for exp in temp:
        temps=Mercado2c[(Mercado2c.EXPIR_DATE.astype('datetime64')==exp)&(Mercado2c['Date'].astype('datetime64')==datemin[0])]
        temps=temps.reindex(temps.PriceToStrike0.abs().sort_values().index)
        temps=temps[:numOpc]['Security'].values
        tempRes=tempRes+(list(temps))
    return(list(set(tempRes)))


def DataEditing(Mercado1,Refp,numOpc,DataRef):
    '''
       Aplica todas as funçoes em forma definida:
        Procedimento de tratamento de dados.
    '''
    if len(Mercado1)>10 and (len(set(Mercado1.PUT_CALL))>1):
        Mercado1c=Mercado1[Mercado1.PUT_CALL=='Call']
        Mercado1p=Mercado1[Mercado1.PUT_CALL!='Call']
        

        temp=[np.sort(list(set(Mercado1p.EXPIR_DATE.values.astype('datetime64'))))[np.isin(np.sort(list(set(Mercado1p.EXPIR_DATE.values.astype('datetime64')))),(np.sort(list(set(Mercado1c.EXPIR_DATE.values.astype('datetime64'))))))],
              np.sort(list(set(Mercado1c.EXPIR_DATE.values.astype('datetime64'))))[np.isin(np.sort(list(set(Mercado1c.EXPIR_DATE.values.astype('datetime64')))),(np.sort(list(set(Mercado1p.EXPIR_DATE.values.astype('datetime64'))))))]]

        if len(temp)>0:
            if (len(temp[0])+len(temp[1]))>1:

                temp=np.sort(temp[np.argmax([len(temp[0]),len(temp[1])])])
                if len(temp)>0:

                    Mercado2c=ExpirationFilter(0,Refp,Mercado1c,temp)
                    Mercado2p=ExpirationFilter(0,Refp,Mercado1p,temp)

                    listacall=DeltaStreikFilter(numOpc,Mercado2c,Mercado2p,DataRef,temp)
                    listaput=DeltaStreikFilter(numOpc,Mercado2p,Mercado2c,DataRef,temp)

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
                        Mercado3c=Mercado3c.reindex(Mercado3c.STRIKE_PRC.sort_values().index)
                        Mercado3c=Mercado3c.append(pd.DataFrame(np.zeros(((np.max([0,numOpc-len(Mercado3c)])),len(Mercado3c.columns))),columns=Mercado3c.columns))
                        Mercado3p=Mercado2p[Mercado2p.Date.astype('datetime64')==d]#.sort_values('PriceToStrike0')
                        Mercado3p=Mercado3p[Mercado3p.dia_1!=Mercado3p.Date]
                        Mercado3p=Mercado3p.reindex(Mercado3p.STRIKE_PRC.sort_values().index)
                        Mercado3p=Mercado3p.append(pd.DataFrame(np.zeros(((np.max([0,numOpc-len(Mercado3p)])),len(Mercado3p.columns))),columns=Mercado3p.columns))
                        show=show+[Mercado3c.append(Mercado3p)]
                
                    return (kdate,show,0,np.zeros(len(show[0])))
                else:
                    print("Try other")
                    return ([0],[0],0,np.zeros(len([0])))
            else:
                print("Try other")
                return ([0],[0],0,np.zeros(len([0])))
        else:
            print("Try other")
            return ([0],[0],0,np.zeros(len([0])))
    else:
        print("Try other")
        return ([0],[0],0,np.zeros(len([0])))

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Acompanhamento de PL')
    plt.xlabel('date')
    plt.ylabel('$')
    plt.plot(durations_t.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


def SQLExtrat(ativo,DataRef,Ambi,df):
    lista=list((pd.Series(list(df.columns)[:10]+Ambi+['dia_1','PriceToStrike0','CLOSE_ATIVO']+list(df.columns)[-3:])).drop_duplicates())
    k=str.replace(ativo,".csv","").replace('@',"0")
    mariadb_connection=mariadb.connect(user='root',passwd="1234",db="opcoes")
    cursor = mariadb_connection.cursor()
    query = ("""SELECT """+str(['opt.'+j for j in lista]).replace("'","").replace("[","").replace("]","")+"""
                    FROM opcoes."""+k+""" opt FORCE INDEX(IN_DATE)
                    INNER JOIN (SELECT a.Date, MAX(a.EXPIR_DATE) AS EXPIR_DATE
                            FROM (SELECT b.Date, b.EXPIR_DATE
                                  FROM opcoes."""+k+"""  b
                                  WHERE b.Date = '"""+str(DataRef[0,1])+"""'
                                  GROUP BY b.Date, b.EXPIR_DATE
                                  ORDER BY b.EXPIR_DATE ASC
                                  LIMIT 0,5) a) filtro
                            ON opt.Date BETWEEN filtro.Date AND filtro.EXPIR_DATE
                            WHERE CLOSE_x IS NOT NULL
                                    AND CLOSE_1  IS NOT NULL
                                    AND CLOSE_2 IS NOT NULL
                                    AND CLOSE_3 IS NOT NULL
                            ORDER BY opt.Date DESC""")

    df_verses = pd.read_sql(query, mariadb_connection)#,dtype=df.dtypes[lista].to_dict())
    df_verses['Date'] = df_verses['Date'].astype('object')
    df_verses['EXPIR_DATE'] = df_verses['EXPIR_DATE'].astype('object')
    df_verses['dia_1'] = df_verses['dia_1'].astype('object')
    df_verses['Security'] =df_verses['Security'].astype('object')

    cursor.close()
    mariadb_connection.close()
    return(df_verses)
