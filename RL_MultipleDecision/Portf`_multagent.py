# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 22:38:26 2018

@author: pedro
"""


from __future__ import division
from copy import deepcopy
import sys
import csv
sys.path.insert(0,'C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\OptionsSVM\\Ambiente Opçoes')
from Variables import*
from Options_Class_Fast_SQL import*
from random import randint
import hiddenlayer as hl

def NormNeg(k):
    '''
       Calcula Açao randomica, vetor de pesos para ativo.
           k: pandas dataframe do stado.
    '''
    w=np.array(k)
    zero=[np.isnan(w)]
    w[zero]=0
    w=((w-w.min())/(w.max()-w.min()))*(1+0.5)-0.5
#    w=(w/(w.max()-w.min()))*(1+0.5)-0.5
#    w=w*(/(1+0.3))
    w=w/abs(sum(w))
    zero=[abs(w)<0.03]
    w[zero]=0
    W=w/(sum(w) if not sum(w)==0 else 1)
    return(W)

    
def randomAction(k,n):
    '''
       Calcula Açao randomica, vetor de pesos para ativo.
           k: pandas dataframe do stado.
    '''
    w=np.random.normal(mean(n), std(n)*2, size=len(k))
    w=w*(w).mean()
    W=w/sum(w)
    return(W)

#def randomAction(k,n):
#    '''
#       Calcula Açao randomica, vetor de pesos para ativo.
#           k: pandas dataframe do stado.
#    '''
#    w=np.random.normal(0.00, 0.10, size=len(k))
#    w=w*(w).mean()
#    W=w/sum(w)
#    return(W)


import numpy as np
import torch
from torch.autograd import Variable
import os
import psutil
import gc

sys.path.insert(0,'C:\\Users\\pedro\\OneDrive\\Documentos\\GitHub\\PyTorch-ActorCriticRL-master\\')

import train12
#import train
import buffer

from os import listdir, stat
import getpass
import os
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

    
## -------------   captura de arquivos - Identifique as pasta em que estao os arquivos.
lista="C:\\Users\\"+getpass.getuser()+"\\Documents\\optons full\\optons full"
numOpc=12 #     numero deativos por grupo
Refp=2 #        numero de meses a fente, maturidade de referencia.
numPeriodos=0 # numero de periodos, solo ou at'e expira'cao.
callput=0 #     nao implementado.
Capital0=50000# Captal teorico.
#Ambi=Variaveis['Ambi_teste']+['PrecoFut1','PriceToStrike0','PrecoFut1','PrecoFut2','PrecoFut3']
Ambi=Variaveis['Ambi_A0']
dmais=130
OP=OptionsBases(lista,numOpc,Refp,numPeriodos,callput,Ambi,Capital0,dmais)

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    durations_tR = torch.FloatTensor(episode_durationsR)
    plt.title('Acompanhamento de PL')
    plt.xlabel('date')
    plt.ylabel('$')
    plt.plot(durations_t.numpy())
    plt.plot(durations_tR.numpy())
    plt.pause(0.00001)  # pause a bit so that plots are u



MAX_EPISODES = 50000
MAX_STEPS = 1000
MAX_BUFFER = 1000000
MAX_TOTAL_REWARD = 1

S_DIM = (len(Ambi)*numOpc*2,1)[0]
A_DIM = numOpc*2
A_MAX = 1

print (' State Dimensions :- ', S_DIM)
print (' Action Dimensions :- ', A_DIM)
print (' Action Max :- ', A_MAX)

ram = buffer.MemoryBuffer(MAX_BUFFER)

trainer = train12.Trainer(S_DIM, A_DIM, A_MAX, ram)
#trainer = train.Trainer(S_DIM, A_DIM, A_MAX, ram)

resultsT=[]
resultsRT=[]

#file='./Results/R.csv'
#file='./Results/R0.csv'
#file='./Results/R2m.csv'
file='./Results/R5.csv'

with open(file,'w') as cr:
    w=csv.writer(cr)
    w.writerows([['EP','Asset','Start','End','ReturnDay','ReturnDayR','ReturnFin']+['Action'+str(i) for i in range(24)]])
cr.close()

#g=hl.build_graph(trainer.Jmodel['0'][0][0],torch.from_numpy(OP.Start()))
#g.save(file.replace('.csv','.pdf'))

for _ep in range(MAX_EPISODES):
    observation =OP.Start()
#    observation =OP.StartPred('BRPOB.csv')

    OPr=deepcopy(OP)
    ## --- randoum exemples actions
    OP.Reset()
    observation =TratamentoStado(OP.Ambiente,OP.StateList[0],)
#    state = torch.from_numpy(state).float()
    episode_durations = []
    episode_durationsR = []
    results=[]
    resultsR=[]
    Aca=np.array([])
    Capital1=OP.Capital1
    print ('EPISODE :- ', _ep)
    for r in range(MAX_STEPS):
        state = np.float32(observation)
    #    action1 = trainer.get_exploitation_action(state)
        if _ep%5 == 0:
    		# 	# validate every 5th episode
            action1 = trainer.get_exploitation_action(state)
        else:
    		# 	# get action based on observation, use exploration policy here
            action1 = trainer.get_exploration_action(state)

        actionR1=randomAction(OPr.StateList[0],action1)
        
        action=NormNeg(action1)       
        actionR=NormNeg(actionR1)
        
        # if _ep%5 == 0:
        # 	# validate every 5th episode
        # 	action = trainer.get_exploitation_action(state)
        # else:
        # 	# get action based on observation, use exploration policy here
        # 	action = trainer.get_exploration_action(state)
        rewardR1=Retorno(OP.StateList[OP.StateCont],OP.Capital1,actionR,OP.last_Q)[2]
        rewardR=Retorno(OP.StateList[OP.StateCont],OP.Capital1,actionR,OP.last_Q)[4]
        Capital1=Capital1*(1+rewardR1)

#        if r==1:
        StrategiFlow(OP.StateList[OP.StateCont],action)

        new_observation, reward1, done, info =OP.Action(action)
        reward=info
#        new_observationR, rewardR, doneR, infoR =OPr.Action(actionR)
#        if sum(np.isnan(reward))>0:
#            print(OP.ativo,OP.DataRef)
#            break

#        if reward<0:
#           reward=reward*2
#
#        if np.isnan(reward):
#           reward=0
#        if np.isnan(rewardR):
#           rewardR=0

#        reward=reward+((new_observation[0]+1.21)/1.81)/100
        # # dont update if this is validation
        # if _ep%50 == 0 or _ep>450:
        # 	continue
        episode_durations.append(OP.Capital1)
        episode_durationsR.append(Capital1)
        plot_durations()

        results.append(reward1)
        resultsR.append(rewardR1)
        Aca=np.append(Aca,[np.array(action1)])
        
        if done or sum(np.isnan(reward))>1:
            new_state = None
        else:
            new_state = np.float32(new_observation)
            # push this exp in ram
#            ram.add(state, action1, reward1, new_state)
            ram.add(state, action1, reward, new_state)
#            ram.add(state, actionR1, rewardR, new_state)

        observation = new_observation

		# perform optimization
        if _ep>100:
#            for e in range(1):
#            xx=time.time()
            trainer.optimize()
#            print((time.time()-xx))
#            durations_t = torch.FloatTensor([np.array(results[i:5000+i]).mean() for i in range(5000,len(results))])
#            plt.subplot(2, 1, 2)
#            plt.plot(durations_t.numpy())

        if done or sum(np.isnan(reward))>1:
            resultsT.append(OP.Capital1/OP.Capital0-1)
            resultsRT.append(Capital1/OP.Capital0-1)
           
#            pd.DataFrame(np.append((np.array([[_ep]*len(results),[OP.ativo]*len(results),
#                            [OP.DataRef[0,1]]*len(results),
#                            [OP.DataRef[1,1]]*len(results),
#                            results,[OP.Capital1/OP.Capital0-1]*len(results)])),
#                            Aca.reshape(24,len(results))).reshape(30,len(results))).T


            with open(file,'a',newline='\n') as cr:
                w=csv.writer(cr)
                w.writerows((np.append((np.array([[_ep]*len(results),[OP.ativo]*len(results),
                    [OP.DataRef[0,1]]*len(results),
                    [OP.DataRef[1,1]]*len(results),
                    results,resultsR,[OP.Capital1/OP.Capital0-1]*len(results)])),
                    Aca.reshape(len(results),24)).reshape(31,len(results))).T)
            cr.close()

            print(OP.Capital1/OP.Capital0-1,Capital1/OP.Capital0-1)
            print(action)
#            print(actionR)
            break

#    if sum(np.isnan(ram.sample(200)[2]))>0:
#        break

	# check memory consumption and clear memory
    gc.collect()
	# process = psutil.Process(os.getpid())
	# print(process.memory_info().rss)

    if _ep%100 == 0:
        trainer.save_models(_ep)

print ('Completed episodes')



StrategiFlow(OP.StateList[0],action)
StrategiFlow(OP.StateList[0],action1)
StrategiFlow(OP.StateList[0],randomAction(OPr.StateList[0],action1))

###------ real test
observation =OP.StartPred('BRPOB.csv')

observation =OP.Start()
print(OP.StateList[0])

OP.Reset()
observation =TratamentoStado(OP.Ambiente,OP.StateList[0],)
OPr=deepcopy(OP)

episode_durations=[]
episode_durationsR=[]
Capital1=OP.Capital1
for r in range(MAX_STEPS):
    state = np.float32(observation)
    action1 = trainer.get_exploitation_action(state)
#    action1 = trainer.get_exploitation_action(state)

    actionR1=randomAction(OPr.StateList[0],action1)
    
    action=NormNeg(action1)       
    actionR=NormNeg(actionR1)
    
    StrategiFlow(OP.StateList[OP.StateCont],action)

    rewardR1=Retorno(OP.StateList[OP.StateCont],OP.Capital1,actionR,OP.last_Q)[2]
    rewardR=Retorno(OP.StateList[OP.StateCont],OP.Capital1,actionR,OP.last_Q)[4]
    Capital1=Capital1*(1+rewardR1)

    new_observation, reward, done, info =OP.Action(action)
    
    episode_durations.append(OP.Capital1)
    episode_durationsR.append(OPr.Capital1)
    plot_durations()
    
    results.append(reward)
    resultsR.append(rewardR1)
    print(action)

    if done or np.isnan(reward):
        new_state = None

    observation = new_observation
    
    if done or np.isnan(reward):
        break

state = np.float32(observation)
action1 = trainer.get_exploration_action(state)
action=NormNeg(action1)       
print(action)
new_observation, reward, done, info =OP.Action(action)
print(OP.StateCont,OP.Capital1)
observation = new_observation

state = Variable(torch.from_numpy(state))
action = trainer.target_actor.forward(state.float()).detach()

action = trainer.actor.forward(state.float()).detach()
new_action = action.data.numpy() + (trainer.noise.sample() * trainer.action_lim)



#print("Ativo:", str(OP.ativo))
#sum(np.array(results)<-1)/len(results)
#sum(np.array(results)<-0.2)/len(results)
#sum(np.array(results)<0)/len(results)
#
#sum(np.array(results)>0)/len(results)
#sum(np.array(results)>0.2)/len(results)
#sum(np.array(results)>1)/len(results)
#sum(np.array(results)>2)/len(results)
#sum(np.array(results)>3)/len(results)
#
#print("Ativo:", str(OP.ativo))
#sum(np.array(resultsR)<-1)/len(resultsR)
#sum(np.array(resultsR)<-0.2)/len(resultsR)
#sum(np.array(resultsR)<0)/len(resultsR)
#
#sum(np.array(resultsR)>0)/len(resultsR)
#sum(np.array(resultsR)>0.2)/len(resultsR)
#sum(np.array(resultsR)>1)/len(resultsR)
#sum(np.array(resultsR)>2)/len(resultsR)
#sum(np.array(resultsR)>3)/len(resultsR)

def porcent(x,y,i,s):
    xi=sum(((x>=i)) & ((x<=s)))/len(x)
    yi=sum(((y>=i)) & ((y<=s)))/len(y)
    return(xi,yi)

k=-2000
x = np.array(results[k:])
y = np.array(resultsR[k:])

y = np.array(results[:-k])

x = np.array(resultsT[k:])
y = np.array(resultsRT[k:])

y = np.array(resultsT[:-k])

mean(x-y)
std(x-y)

i,s=-0.1,0
porcent(x,y,i,s)

i,s=0,0.1
porcent(x,y,i,s)

i,s=0,0.5
porcent(x,y,i,s)

i,s=-0.5,0
porcent(x,y,i,s)


mean(x)
mean(y)

bins = numpy.linspace(-1, 1, 300)

pyplot.hist(x,bins, alpha=0.5, label='x')
pyplot.hist(y,bins, alpha=0.5, label='y')
pyplot.legend(loc='upper right')
pyplot.show()


x.min()
y.min()

x[x<0].mean()
y[y<0].mean()

x[x>0].mean()
y[y>0].mean()

sum(x<-.050)/len(x)
sum(y<-.050)/len(y)

sum(x>.2)/len(x)
sum(y>.2)/len(y)


np.array(results[:5000]).max()
np.array(results[-5000:]).max()



x = np.array(results[k:])
y = np.array(resultsR[k:])

episode_results=[]
episode_resultsS=[]
episode_resultsR=[]
for k in range(0,len(results),100):
#    episode_results.append(np.array(results[k:k+1000]).mean())
#    episode_resultsS.append(np.array(results[k:k+1000])[np.array(results[k:k+1000])>0].mean())
    episode_resultsR.append(np.array(results[k:k+1000])[np.array(results[k:k+1000])<0].mean())
#    episode_resultsR.append(np.array(results[k:k+100]).min())

    durations_t = torch.FloatTensor(episode_results)
    durations_tR = torch.FloatTensor(episode_resultsR)
    durations_tMax = torch.FloatTensor(episode_resultsS)
    plt.figure(8)
    plt.clf()
    plt.title('Acompanhamento de PL')
    plt.xlabel('date')
    plt.ylabel('$')
#    plt.plot(durations_t.numpy())
#    plt.fill_between(durations_t.numpy(),durations_t.numpy()-durations_tMax.numpy(),durations_t.numpy()+durations_tMax.numpy(),color='#888888', alpha=0.4)
#    plt.fill_between(durations_t.numpy(),durations_tMax.numpy(),durations_tR.numpy(),color='#888888', alpha=0.4)
#    plt.plot(durations_tMax.numpy())
    plt.plot(durations_tR.numpy())
    plt.pause(0.0001)  # pause a bit so that plots are u




episode_results=[]
episode_resultsS=[]
episode_resultsR=[]

for k in range(0,len(resultsT),100):
    episode_results.append(np.array(resultsT[k:k+100]).mean())
#    episode_resultsS.append(np.array(resultsT[k:k+100])[np.array(resultsT[k:k+100])>0].mean())
#    episode_resultsR.append(np.array(resultsT[k:k+100])[np.array(resultsT[k:k+100])<0].mean())
#    episode_resultsR.append(np.array(resultsT[k:k+100]).min())

    durations_t = torch.FloatTensor(episode_results)
    durations_tR = torch.FloatTensor(episode_resultsR)
    durations_tMax = torch.FloatTensor(episode_resultsS)
    plt.figure(8)
    plt.clf()
    plt.title('Acompanhamento de PL')
    plt.xlabel('date')
    plt.ylabel('$')
    plt.plot(durations_t.numpy())
#    plt.fill_between(durations_t.numpy(),durations_t.numpy()-durations_tMax.numpy(),durations_t.numpy()+durations_tMax.numpy(),color='#888888', alpha=0.4)
#    plt.fill_between(durations_t.numpy(),durations_tMax.numpy(),durations_tR.numpy(),color='#888888', alpha=0.4)
#    plt.plot(durations_tMax.numpy())
#    plt.plot(durations_tR.numpy())
    plt.pause(1)  # pause a bit so that plots are u