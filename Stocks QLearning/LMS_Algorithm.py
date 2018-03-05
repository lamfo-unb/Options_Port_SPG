# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 19:18:54 2016

@author: Stefano
"""

import numpy as np

#This algortihm implements the Linear Mean Square algorithm, it receives the learning rate and the initial weights
#as input, the function fit executes the adaptation loop, the function predict receives an input and returns the
#predicted output  

class LMS():
    def __init__(self, mu, w):
        self.kind = "LMS filter"
        self.n = np.shape(w)
        self.mu = mu
        self.w = w

    def fit(self, x,d):

        N = np.shape(d)
        #output dimension
        o_d = np.shape(d)
        #input dimension
        i_d = np.shape(x)
        # create empty arrays
        y = np.zeros(N)
        e = np.zeros(N)
        self.w_history = np.zeros(self.n)
        # adaptation loop
        for k in range(N[0]):
            X = np.reshape(x[k,:],(1,i_d[1]))
            y[k,:] = np.dot(X,self.w.T)
            e[k,:] = d[k,:] - y[k,:]
            E = np.reshape(e[k,:],(1,o_d[1]))
            dw = self.mu * np.dot(X.T,E)
            self.w += dw.T
            
        return y, e, self.w

    def predict(self,x_pred):
        y_predicted = np.dot(x_pred,self.w.T) 
        return y_predicted

    def show_weights(self):
        print self.w
