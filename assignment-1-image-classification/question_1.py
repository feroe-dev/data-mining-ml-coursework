# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:01:24 2021

@author: Felipe Rötger
"""
import numpy as np
import matplotlib.pyplot as plt

Xtest = np.loadtxt("Xtest.txt") #Xtrain: Training Data (each row is a single image)
Xtrain = np.loadtxt("Xtrain.txt") #Ytrain: Training labels
Ytrain = np.loadtxt("Ytrain.txt") #Xtest: Test labels (each row is a single image)


#%% i
#dont forget to load Xtest, Xtrain and Ytrain!!

print("Amount of training examples:",len(Xtest))
print("Amount of test examples:",len(Xtrain))

for i in range (0,10):  
    r = np.random.randint(len(Xtest)) #choosing one of one of 10 examples randomly
    A = np.reshape(Xtest[r],(28,28))    #rearrange the example in a 28x28 matrix, in order to create the image
    cl = (r,Ytrain[r])
    plt.matshow(A)  #creating image based on the example
    plt.title('Image No/Label: %d/%d' %cl)
    plt.show()
  
#%% ii
#self-explanatory, counting all negative & positive numbers and all zeros entries
def counter(A): 
    pos_count, neg_count, zero_count = 0, 0, 0
    m = len(A[0])
    n = len(A)
    for i in range(n): 
        for j in range(m):  
            if A[i][j] > 0: 
                pos_count += 1
            elif A[i][j] < 0: 
                 neg_count += 1
            else: 
                zero_count += 1                
    return pos_count, neg_count, zero_count,
x = counter(Xtest)          
print("Amount of pos. numbers: ",x[0],"Amount of neg. numbers: ",x[1],"Amount of zeros: ",x[2])
#%% iii
Ytrain = Ytrain[:,np.newaxis] #adding additional dimension, needed otherwise the function counter would cannot handle Ytrain
x = counter(Ytrain)
print("Amount of pos. numbers: ",x[0],"Amount of neg. numbers: ",x[1],"Amount of zeros: ",x[2])
# =============================================================================
# I would not use accuracy as a performance metric and the AUC-ROC neither beacuse one key assumption of both metrics are that the data set is supposed to be balanced.
# This is obviusly not the case here beacusethere are arround 61% more positive labels than negative labels. Finally, only the AUC-RC is left.
# Nevertheless I would suggest to use the AUC-RC, besides the logic above, because we also want to know the precision of our classifier, independant of the treshold and it
# is useful for imbalance-classes which is the case here.
# =============================================================================
#%% iv

from sklearn.metrics import confusion_matrix ,balanced_accuracy_score, accuracy_score
x = np.random.randint(1, 1000)*0.001                       #random classifier with random weights
p = int(round(x*3000))                                     #weight times 3000
q = int(round((1-x)*3000))                                 #p and q are linear combination such that q+p=3000 always!
Ypred = np.random.permutation(np.array([+1]*p+[-1]*q))     #random allocating prediction labels with permutation to guarantee a random order
#YPred = np.random.permutation(np.array([-1]*3000))
print("confusion_matrix= \n",confusion_matrix(Ytrain, Ypred))
print("weight on Y=1: ",round(x,5),"weight on Y=-1: ",round((1-x),5))
print("accuracy by CPU =",accuracy_score(Ytrain,Ypred))
#print("balanced accuracy by CPU =",balanced_accuracy_score(Ytrain,YPred))
print( "E[Accuracy] = (x*1824+(1+x)*1179)/3000 =",(x*1824+(1-x)*1179)/3000)
# =============================================================================
# Let the classifier be distrubted the following: P(Y_C=1) = x and P(Y_C=-1) = (1-x), where x element [1,0]. E[Y] = 1*x + (-1)*(1-x) = 2x-1.
# We have n = 3000, Frequency(Y=1) = 1824 & Frequency(Y=-1) = 1179, or Frequenzy(Y=1) = n*fra(Y=1) & Frequenzy(Y=-1) = n*fra(Y=-1)
# Fraction of examples labelled as -1 = 0.608 (=fra(Y=1)) & Fraction of examples labelled as +1 = 0.392 (=fra(Y=-1)),
# E[Y_C=1] = x*3000 & E[Y_C=1] = (1-x)*3000, E[tp]=x*Frequency(Y=1) & E[tn]=(1-x)*Frequency(Y=-1),
# where tp is the amount of true postitive allocations and tn is the amount of true neagtive allocations.
# E[Accuracy] = E[(tp+tn)/n] = (E[tp]+E[tn])/n = (x*n*fra(Y=1)+(1-x)*n*fra(Y=-1))/n = (x*fra(Y=1)+(1-x)*fra(Y=-1)).
# As conclusion: E[Accuracy] = (x*1824+(1+x)*1179)/3000, where  x element [1,0] determines the distrubution of weights on the binary labels.
# Comparing my formula for E[Accuracy] to the calculated accuracy by the CPU you see that they are nearly equal. The deviation results from a numerical rounding error.
# =============================================================================
#%% v
from sklearn.metrics import average_precision_score, roc_curve, roc_auc_score, auc, precision_recall_curve
#similar to evaluation_example.ipynb I am using plotRoc & plotPRC to calculate AUC-Roc and AUC-PR here.
def plotROC(y,z,pstr = ''):
    fpr,tpr,tt = roc_curve(y, z)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr,tpr,'o-');plt.xlabel('FPR');plt.ylabel('TPR');plt.grid();plt.title('ROC '+pstr+' AUC: '+str(roc_auc))
    return roc_auc

def plotPRC(y,z,pstr = ''):
    P,R,tt = precision_recall_curve(y, z)
    pr_auc = average_precision_score(y, z)
    plt.figure()
    plt.plot(R,P,'o-');plt.xlabel('Recall/TPR');plt.ylabel('Precision');plt.grid();plt.title('PRC '+pstr+' AUC: '+str(pr_auc))
    return pr_auc

roc = plotROC(Ytrain,Ypred,'')
pr = plotPRC(Ytrain,Ypred,'')

print("By the CPU we get: AUC-ROC= ",roc,"and AUC-PR= ",pr)
print("AUC-PR calculated by my self = ",0.5*(0.608/x))
# =============================================================================
# 
# We also use the same designations from the previous task iv) here!
# a)
# The random classifier allocats each label with some propbality: P(Y_C=1) = x and P(Y_C=-1) = (1-x),
# the ROC would look like the function f(x) = x in the intervall [0,1].
# Thats because the TPR and FPR are equal at evry point. But why is TPR(FPR) = FPR?
# Mathematical approach:
# TPR(x) = TP/F(Y=1) = x*F(Y=1)/F(Y=1) = x
# FPR(x) = FP/F(Y=-1) = x*F(Y=-1)/F(Y=-1) = x
# => TPR(x) = FPR (x) for all x element [0,1]
# Integrating f(x) = x with the integral boundaries 0 and 1 we would get 0.5. Therefore AUCROC = 0.5 for all x element[0,1]
# The image of the ROC in the output underlines this and the calculated AUC-ROC is almost 0.5!
# b)
# Claim: Precision(TPR) = fra(Y=1) = 0.6081
# Proof: Precision(TPR) = TP/E[Y=1] = x*F(Y=1)/(x*n) = F(Y=1)/n = (fra(Y=1)*n)/n = fra(Y=1) = 0.608 (from previous task we know fra(Y=1) = 0.608)
# Intergrating Precision(TPR) = fra(Y=1) in respect to TPR gives us [fra(Y=1)*TPR] with the boundaries 0 and 1. The result is fra(Y=1). The computed AUC-PR supports the claim.
# =============================================================================
