#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:20:27 2018

@author: dmitriy
"""

import numpy as np
pos = np.load('pos.npy')
neg = np.load('neg.npy')


import pandas as pd

pos = pd.DataFrame(pos)
neg = pd.DataFrame(neg)

pos.plot.hist()
neg.plot.hist()


pos = np.load('pos.npy')
neg = np.load('neg.npy')



class wrapper():
    def __init__(self, value, class_):
        self.value = value
        self.class_ = class_


arr = []
for i in pos:
    arr.append(wrapper(i[0], 1))
    
for i in neg:
    arr.append(wrapper(i[0], 0))
    
arr.sort(key = lambda w: w.value, reverse = True)

total_0 = 0
total_1 = 0
for i in range(0, len(arr)):
    if(arr[i].class_==0):
        total_0 += 1
    else:
        total_1 += 1
        

fpr = []
tpr = []

tpr.append(total_1)
fpr.append(total_0)

for i in range(1, len(arr)):
    if(arr[i].class_ == 1):
        tpr.append(tpr[i-1] - 1)
        fpr.append(fpr[i-1])
    else:
        tpr.append(tpr[i-1])
        fpr.append(fpr[i-1] - 1)
        

for i in range(0, len(fpr)):
    fpr[i] = float(fpr[i])/total_0
    tpr[i] = float(tpr[i])/total_1
    
    
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)

s = 0
g = 0
for i in tpr:
    s+=i
    g += 1
    

s = s / len(tpr)

print(s)


