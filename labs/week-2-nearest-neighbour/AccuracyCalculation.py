# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 20:11:52 2021

@author: Felipe
"""
import ImplementingANearestNeighborFunction

def accuracy(Ytest,Ypred):
    # to do: calculate accuracy
    n = len(Ytest)
    fa = 0   #frequency right prediction
    for i in range(0,n):
        if Ytest[i] == Ypred[i]:
            fa += 1              #frequency right prediction
    a = fa/n
    return a
