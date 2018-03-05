# -*- coding: utf-8 -*-
"""
Created on Sun May 07 11:26:04 2017

@author: Stefano
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from ReadData import getPrices #load data
from Technical import bollinger, moving_average_convergence, moving_average, stochastic_oscillator,daily_return,volume #evaluate indicators
from LMS_Algorithm import LMS #defines the approximation function
from IPython.display import clear_output

#initializing the variables

def run_algo(symbol,epochs,initial_train,final_train,initial_test,final_test):
    
    

#    Trains a Q-Learning agent for stock trading
#    Parameters
#    ----------
#    symbol : str
#        Name of the stock on the exchange, ex: APPL, MSFT, PETR4.SA
#    epochs : int
#        Number of epochs for the training
#    initial_train,final_train,initial_test,final_test : str/datetime
#        Training and test period in the format 'AAAA/MM/DD'

#   example of parameters: run_algo('BBAS3.SA',2000,'2013-01-10','2017-01-10','2017-01-10','2017-12-10')
#    Returns
#    ----------
#    new_df : pandas.DataFrame
#        A df like DataFrame with the price column replaced by the log difference in time.
#        The first row will contain NaNs due to first diferentiation.
    
    
    
    initial_money = 50000
    columns = ['Position','NumShares','Money','PortValue','Reward']
    
    #reading training data
    df_stock = getPrices(symbol,initial_train,final_train)
    
    df_stock.dropna(inplace = True)

    df_system = pd.DataFrame(index = df_stock.index, columns = columns)
    
    
    
    #reading test data

    df_stock_test = getPrices(symbol,initial_test,final_test)
    
    df_stock_test.dropna(inplace = True)
    
    df_system_test = pd.DataFrame(index = df_stock_test.index, columns = columns)
    

    
    
    total_periods = np.size(df_stock,axis = 0) - 1
    
    
    df_system.loc[0:1,'Money'] = initial_money
    df_system = df_system.fillna(0)
    df_system['Close'] = df_stock['Close']
    
    
    df_system_test.loc[0:1,'Money'] = initial_money
    df_system_test = df_system_test.fillna(0)
    df_system_test['Close'] = df_stock_test['Close']
    
    
    
    
    #states vector
    
    num_states = 4
    def getStates(df,num_states):
        states = np.zeros([np.size(df,axis = 0),num_states])
        states[:,1] = daily_return(df)    
        states[:,0] = bollinger(df,20)[0]
        states[:,2] = moving_average_convergence(df)[2]
        states[:,3] = (stochastic_oscillator(df)[2] -0.5)
        states[np.isnan(states)] = 0

        return states
    
    
    
    states = getStates(df_stock,num_states)
    states_test = getStates(df_stock_test,num_states)
    
    
    def update(df):
        df['Total'] = df['Money'] + df['PortValue']
        
    update(df_system)
    
    update(df_system_test)
    
    n = num_states #number of parameters (input dimension)
    m = 3 #number of possible actions (output dimensions)
    w_initial = np.squeeze(np.random.rand(m,n)/100)
    
    #initialize LMS algorithm with random weights with the correct dimensions
    model = LMS(0.05,w_initial)
    

    def getReward(i,df,state):
        reward = 100*(1.5*((df.iloc[i]['Total'] - df.iloc[i-1]['Total'])/df.iloc[i-1]['Total']) - (df.iloc[i]['Close'] - df.iloc[i-1]['Close'])/df.iloc[i-1]['Close'])
        return reward
    

    
    df_system.loc[0:1,'Money'] = initial_money
    df_system_test.loc[20:21,'Money'] = initial_money
    
    
    
    import time
    start_time = time.time()
    
    
    
    epsilon = 1
    gamma = 0.95
    # Y is used to assess whether the convergence of Q was achieved or not 
    Y = np.zeros((epochs,3))
    cout = 0
    reco = np.zeros(epochs)
    rewa = np.zeros(epochs)
    
    for k in range(epochs):
        i = 1
        state = states[i]
        position = 0
        while (total_periods > i):
            qval = model.predict(state.reshape(1,num_states)) #approximating Q values
            if (random.random() < epsilon): #explore x exploit
                action = np.random.randint(-1,2)
            else:
                action = np.argmax(qval)-1
            df_system.ix[i,'Position'] = action #tracking actions taken
            if action == 1 and df_system.ix[i-1,'NumShares'] > 0 :
                numshare = 0 #numshare is a variable that tells us how many shares are bought/sold in this state
            elif action == -1 and df_system.ix[i-1,'NumShares'] < 0:
                numshare = 0
            elif df_system.iloc[i-1]['NumShares'] !=0 :
                numshare = np.abs(df_system.iloc[i-1]['NumShares']) * df_system.iloc[i]['Position']
            else:
                numshare = np.int(df_system.iloc[i-1]['Total']/df_system.iloc[i]['Close']) * df_system.iloc[i]['Position']
            
            #updating the dataframe
            df_system.ix[i,'Money'] = df_system.iloc[i-1]['Money'] - numshare*df_system.iloc[i]['Close']
            df_system.ix[i,'NumShares'] = df_system.iloc[i-1]['NumShares'] + numshare         
            df_system.ix[i,'PortValue'] = df_system.iloc[i]['NumShares']*df_system.iloc[i]['Close']     
            df_system['Total'] = df_system['Money'] + df_system['PortValue']
            
            new_state = states[i+1] #goes to the next state
            reward = getReward(i,df_system,position) #evaluate the reward
            df_system.ix[i,'Reward'] = reward
            newQ = model.predict(new_state.reshape(1,num_states)) #approximates Q
            maxQ = np.max(newQ)
            y = np.zeros((1,3))
            y[:] = qval[:]
            update = (reward + (gamma * maxQ))
            y[0][action+1] = update      #updates only the Q of the action taken
            model.fit(state.reshape(1,num_states), y) #weight updates
            state = new_state
            i+=1
            clear_output(wait=True)
            if epsilon > 0.1:  #reduces epsilon
                epsilon -= (0.5/epochs)
            if (epochs - k) < 10: # epsilon goes to zero in the last 10 epochs
                epsilon = 0
            
        reco[cout] = (df_system.iloc[-2]['Total'] - df_system.iloc[2]['Total'])/df_system.iloc[2]['Total']
        Y[cout] = y  # Y is a array with Q values of the last time period, we can evaluate the convergence of Q using this
        rewa[cout] = np.sum(df_system['Reward'])
        cout+=1
        
    summary = df_system[['Position','NumShares','Reward','Close','Total']] 
    
    bench = (df_system.iloc[-2]['Close'] - df_system.iloc[2]['Close'])/df_system.iloc[2]['Close']
    
    print("--- %s seconds ---" % (time.time() - start_time))


    
    
    def test_algo(XD,states,df):
        df_test_algo = df.copy()           
        total_periods = np.size(states,0)
        i = 1
        while(total_periods > i):
            Q = np.dot(states[i],XD.T)
            action = np.argmax(Q)-1
            df_test_algo.ix[i,'Position'] = action
            if action == 1 and df_test_algo.ix[i-1,'NumShares'] > 0 :
                numshare = 0 #numshare is a variable that tells us how many shares are bought/sold in this state
            elif action == -1 and df_test_algo.ix[i-1,'NumShares'] < 0:
                numshare = 0
            elif df_test_algo.iloc[i-1]['NumShares'] !=0 :
                numshare = np.abs(df_test_algo.iloc[i-1]['NumShares']) * df_test_algo.iloc[i]['Position']
            else:
                numshare = np.int(df_test_algo.iloc[i-1]['Total']/df_test_algo.iloc[i]['Close']) * df_test_algo.iloc[i]['Position']
            df_test_algo.ix[i,'Money'] = df_test_algo.iloc[i-1]['Money'] - numshare*df_test_algo.iloc[i]['Close']
            df_test_algo.ix[i,'NumShares'] = df_test_algo.iloc[i-1]['NumShares'] + numshare         
            df_test_algo.ix[i,'PortValue'] = df_test_algo.iloc[i]['NumShares']*df_test_algo.iloc[i]['Close']     
            df_test_algo['Total'] = df_test_algo['Money'] + df_test_algo['PortValue']
            i+=1
        return df_test_algo
        
    X = test_algo(model.w,states_test[20:],df_system_test[20:])
    print model.w
    result = (X.iloc[-2]['Total'] - X.iloc[2]['Total'])/X.iloc[2]['Total']
    bench_test = (X.iloc[-2]['Close'] - X.iloc[2]['Close'])/X.iloc[2]['Close']
    print '\n'
    print 'Inicio do teste %s, fim do teste %s'  %(initial_test,final_test)
    print 'Resultado teste %f' %result
    print 'Benchmark teste %f' %bench_test  
    print 'Inicio do treino %s, fim do treino %s'  %(initial_train,final_train)
    print 'Resultado treino %f' %reco[-1]
    print 'Benchmark treino %f' %bench 
    

    df_system = 0
    df_system_test = 0
    
    

