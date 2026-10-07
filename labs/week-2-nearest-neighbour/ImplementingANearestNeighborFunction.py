# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:39:14 2021

@author: Felipe
"""
from  GVData import getExamples
import numpy as np
from ImplementingTheEuclideanDistanceFunction import distance

Xtest,Ytest = getExamples(n=100,d=2)

def NN(Xtr,Ytr,Xtt):
    """
    Input:
        Xtr: (n x d) numpy matrix of n training examples each with d features
        Ytr: (n) dimensional numpy vector matrix of training labels
        Xtt: (m x d) numpy matrix of m testing examples each with d features
    Return:
        Ypred: (m) dimensional numpy vector matrix of testing labels
    TO IMPLEMENT:
        A nearest neighbor classifier
        For each example in Xtt, find its closest example in Xtr
        You can find the euclidean distance between two examples
        Associate the label of the nearest training example to the testing one
        Construct a vector Y with the label of each testing example 
    
    """
    n = len(Xtr)
    m = len(Xtt)
    a = np.zeros((n,1))
    Ypred = np.zeros(m) #entry min value
    for i in range (0,m): 
        for j in range (0,n):                   #searching for the closest row in Xtr
            a[j] = distance(Xtr[j],Xtt[i])  
        Ypred[i] = Ytr[np.argmin(a)]                   #saving entry of the closest row in Xtr
        
    # Below is a dummy return for testing only. It generates random labels.
    "evm[i] = np.argmin(a)"
    # Remove it when you implement your classifier.
    #Ypred = 2*np.randint(0,2,Xtt.shape[0])-1 
    return Ypred

if __name__=='__main__':
    X, Y = getExamples(n=100,d=2)
    Y = np.random.permutation(Y)
    Ypred = NN(X,Y,Xtest)