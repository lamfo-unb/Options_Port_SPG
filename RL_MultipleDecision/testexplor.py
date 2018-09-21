# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:23:41 2018

@author: pedro
"""
OP.Start()

OP.ativo
OP.DataRef[0][1]

OP.StartPred(OP.ativo,OP.DataRef[0][1])


OP.StateList[0][['Date','Security', 'PUT_CALL','EXPIR_DATE', 'STRIKE_PRC']]
OP.StateList[1][['Date','Security', 'PUT_CALL','EXPIR_DATE', 'STRIKE_PRC']]
OP.StateList[2][['Date','Security', 'PUT_CALL','EXPIR_DATE', 'STRIKE_PRC']]
OP.StateList[3][['Date','Security', 'PUT_CALL','EXPIR_DATE', 'STRIKE_PRC']]

len(OP.StateList)>2
    
Mercado1=SQLExtrat(OP.ativo,OP.DataRef,OP.Ambiente,OP.df)

DataEditing(Mercado1,OP.Refp,OP.numOpc,OP.DataRef)

OP.dataspace,OP.StateList,OP.StateCont,OP.last_Q

DataEditing(Mercado1,OP.Refp,OP.numOpc,OP.DataRef)

len(OP.StateList)>2 or (sum(OP.StateList[0].Security==0)<12)


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

        if len(temp)>0 :
            if (len(temp[0])+len(temp[1]))>1:

                temp=np.sort(temp[np.argmax([len(temp[0]),len(temp[1])])])
                if len(temp)>0:

                    Mercado2c=ExpirationFilter(0,Refp,Mercado1c,temp)
                    Mercado2p=ExpirationFilter(0,Refp,Mercado1p,temp)
                        
                    listacall=DeltaStreikFilter(numOpc,Mercado2c,Mercado2p,OP.DataRef,temp)
                    listaput=DeltaStreikFilter(numOpc,Mercado2p,Mercado2c,OP.DataRef,temp)


datemin=np.array([Mercado2c.append(Mercado2p).Date.min()]).astype('datetime64')
tempRes=[]

Mercado2c[['Date','Security', 'PUT_CALL','EXPIR_DATE', 'STRIKE_PRC','CLOSE_ATIVO']]

Mercado2c[['STRIKE_PRC','CLOSE_ATIVO']]
Mercado2c['STRIKE_PRC']


temps=Mercado2c[(Mercado2c['Date'].astype('datetime64')>=datemin[0])]
temps=temps.reindex(temps.PriceToStrike0.abs().sort_values().index)
temps=temps[:numOpc]['Security'].values
tempRes=tempRes+(list(temps))

return(list(set(tempRes)))





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
