# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:03:01 2021

@author: Felipe
"""

import numpy as np

def distance(p, q):
    """
    Input:
        p: (1xd) numpy vector of 1 training examples each with d features
        q: (1xd) numpy vector of 1 training examples each with d features        
    Return:
        d: Euclidean Distance
    TO IMPLEMENT:
        Euclidean distance formula.    
    """
    n = len(p)
    d = 0
    if len(p) != len(q):
        raise ValueError("Vecors must have the same size")
    for i in range(0,n):
        d += (p[i]-q[i])**2
    d = np.sqrt(d)
    return d

if __name__=='__main__':
    print(distance(np.array([0,0]),np.array([1,1])))
    print(distance(np.array([1,1]),np.array([0,0])))
    print(distance(np.array([-1,-5]),np.array([1,0])))
