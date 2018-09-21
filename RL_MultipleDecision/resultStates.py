# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 08:52:28 2018

@author: pedro
"""
import matplotlib.pyplot as plt

import pandas as pd
file='./Results/R4.csv'

b=pd.read_csv(file)
b.head()
b.columns


x = np.array(b['ReturnDay'][10000:])
y = np.array(b['ReturnDay'][:10000])

 
mean(x)
mean(y)

std(x)
std(y)

bins = numpy.linspace(-5, 5, 300)

pyplot.hist(x,bins, alpha=0.5, label='x')
pyplot.hist(y,bins, alpha=0.5, label='y')
pyplot.legend(loc='upper right')
pyplot.show()

k=0
sum(x>k)/len(x)
sum(y>k)/len(y)

k=0.1
sum(x>k)/len(x)
sum(y>k)/len(y)

k=0.2
sum(x>k)/len(x)
sum(y>k)/len(y)

k=1
sum(x>k)/len(x)
sum(y>k)/len(y)

k=-.1
sum(x<k)/len(x)
sum(y<k)/len(y)

k=-0.2
sum(x<k)/len(x)
sum(y<k)/len(y)

k=-1
sum(x<k)/len(x)
sum(y<k)/len(y)

k=-2
sum(x<k)/len(x)
sum(y<k)/len(y)





