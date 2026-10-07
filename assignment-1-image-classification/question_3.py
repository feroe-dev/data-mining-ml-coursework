# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 23:38:21 2021

@author: Felipe
"""
import numpy as np
import matplotlib.pyplot as plt

Xtest = np.loadtxt("Xtest.txt") #Xtrain: Training Data (each row is a single image)
Xtrain = np.loadtxt("Xtrain.txt") #Ytrain: Training labels
Ytrain = np.loadtxt("Ytrain.txt") #Xtest: Test labels (each row is a single image)

#%%framework
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
skf = StratifiedKFold(shuffle = True)
from sklearn import preprocessing

normalizer = preprocessing.Normalizer().fit(Xtrain) #default=’l2’ l2 refers to the  Euclidean norm or l^2 norm - here we normalized Xtrain
Xtrain_n = normalizer.transform(Xtrain) #Normalization of Xtrain
#only used when cross-validation performance is better than without pre-processing

#%% k-nearest neighbour
from sklearn.neighbors import KNeighborsClassifier

n=2
accuracy = np.zeros((n,2))
auc_roc = np.zeros((n,2))
auc_pr = np.zeros((n,2))

for k in range (1,n+1): #searchin for the best k in the intervall [1,31]
    sknn = KNeighborsClassifier(n_neighbors=k)
    acc = cross_val_score(sknn, Xtrain, Ytrain, scoring = 'accuracy',cv=skf) #reporting accuracy for each fold 
    ar = cross_val_score(sknn, Xtrain, Ytrain, scoring = 'roc_auc',cv=skf)  #reporting AUC-ROC for each fold 
    ap = cross_val_score(sknn, Xtrain, Ytrain, scoring = 'average_precision',cv=skf)    #reporting AUC-PR for each fold 
    accuracy[k-1,0],  accuracy[k-1,1] = np.mean(acc), np.std(acc) #saving means and the std over the fold for each performance metric
    auc_roc[k-1,0], auc_roc[k-1,1] = np.mean(ar), np.std(ar)
    auc_pr[k-1,0], auc_pr[k-1,1] = np.mean(ap), np.std(ap)
    
optik = np.argmax(auc_pr[:,0])    
print("\nK-Nearest Neighbor with k = %i such that AUC-PR is optimal" % (np.argmax(auc_pr[:,0])+1))
print("| cross validation results\t\t| predicted accuracy\t\t| AUC-ROC \t\t\t| Best AUC-PR \t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (accuracy[optik,0], auc_roc[optik,0], np.max(auc_pr[:,0])))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (accuracy[optik,1], auc_roc[optik,1], auc_pr[optik,1]))

#%% Perceptron
from sklearn.linear_model import Perceptron
perc = Perceptron(tol=1e-3, random_state=True)
mem = np.zeros((3,5)) #saving the results of PA, AUC-PR, AUC-PR for the 5-fold strat. cross-validation. Is overwritten again and again in the course of the code for different classifiers

for i, (train_index, test_index) in enumerate(skf.split(Xtrain_n, Ytrain)):   #Perform 5-fold stratified cross-validation
         Xtr, Xv = Xtrain_n[train_index], Xtrain_n[test_index]
         Ytr, Yv = Ytrain[train_index], Ytrain[test_index]
         perc.fit(Xtr,Ytr)
         pred_acc = accuracy_score(Yv,perc.predict(Xv)) #calculating prediction accuracy for each fold
         auc_roc = roc_auc_score(Yv,perc.predict(Xv))   #calculating ROC-AUC for each fold
         auc_pr = average_precision_score(Yv,perc.predict(Xv)) #calculating AUC-PR for each fold
         mem[0,i], mem[1,i], mem[2,i] = pred_acc, auc_roc, auc_pr
         #print('| %i. Fold\t| Predicted accuracy = %1.4f\t| AUC-ROC = %1.4f\t| AUC-PR = %1.4f\t|' % (i+1, pred_acc, auc_roc, auc_pr))
print("\nPerceptron")
print("| cross validation results\t\t| Predicted accuracy \t\t| AUC-ROC t\t\t\t| AUC_PR \t\t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (np.mean(mem[0]), np.mean(mem[1]), np.mean(mem[2])))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (np.std(mem[0]), np.std(mem[1]), np.std(mem[2])))

#nothing to optimize here. In the broadest sense, one could determine the maximum iteration steps of the perceptron algorithm.
#However, this parameter is set by default in such a way that the algorithm stops after 5 iteration steps that do not show any significant improvement in accuracy.

#%% Naive Bayes Classifier (NBC)
# So now wir have the choice between diffrent kinds of NBCs. (Gaussian, Multinomial, Complement, Bernoulli & Categorical)
# As we so in Q2 the training sets are not really Normal distributed, therefore we can exclude the Guassian.
# Obviously our features are not Bernoulli destributed, as you can see in the training data set. Complement and Multinomial we can exclude as well
# because they are mostly used for "test" recognition. I would stick to the categorial distribution. Becuase we know the maximum of each feature space.
# Due each example maps a image we know that each featurevalue refers to a pixel which take binary values from 0 to 255. This refers to the 24-bit RGB color presentation.
# This assumption gets confirmed by having a look on the max and min values of Xtrain. min(Xtrain) = 0 while max(Xtrain) = 255

from  sklearn.naive_bayes import CategoricalNB
# there is a problem with the preinstalled package naiv_bayes. It does not contain the optain to choose min_categories!!!!
# therefore go in the explorer and replace the code of the file naive_bays with the following code: https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/naive_bayes.py#L1054
# or update sklearn to 0.24.1 !! 
r =np.random.randint(0,785) 
pixelcoulour, frequencies = np.unique(Xtrain[:,5], return_counts=True) #counting the frequencie of each pixel in a random feature 
fig, ax = plt.subplots()
plt.title("The %ith feature and its frequency/categorical distribution" % (r))
plt.ylabel('Frequencie of colour')
plt.xlabel("Colour as integer")
plt.bar(pixelcoulour, frequencies,width= 2,align='center') #visulization of the result as bar graph
ax.set_yscale('log')
plt.show()

mem = np.zeros((3,5))
cate = CategoricalNB(fit_prior=True, min_categories = 256) #fir_pror = True prevents uniform distribution

for i, (train_index, test_index) in enumerate(skf.split(Xtrain, Ytrain)):   #Perform 5-fold stratified cross-validation
         Xtr, Xv = Xtrain[train_index], Xtrain[test_index]
         Ytr, Yv = Ytrain[train_index], Ytrain[test_index]
         cate.fit(Xtr,Ytr)
         pred_acc = accuracy_score(Yv,cate.predict(Xv)) #calculating prediction accuracy for each fold         
         auc_roc = roc_auc_score(Yv,cate.predict(Xv))   #calculating ROC-AUC for each fold
         auc_pr = average_precision_score(Yv,cate.predict(Xv)) #calculating AUC-PR for each fold
         mem[0,i], mem[1,i], mem[2,i] = pred_acc, auc_roc, auc_pr
         #print('| %i. Fold\t| Predicted accuracy = %1.4f\t| AUC-ROC = %1.4f\t| AUC-PR = %1.4f\t|' % (i+1, pred_acc, auc_roc, auc_pr))
print("\nCategorical Naive Bayes")
print("| cross validation results\t\t| Predicted accuracy \t\t| AUC-ROC t\t\t\t| AUC_PR \t\t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (np.mean(mem[0]), np.mean(mem[1]), np.mean(mem[2])))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (np.std(mem[0]), np.std(mem[1]), np.std(mem[2])))

#no parameter to obtimize here
#%% Logisic regression
from sklearn.linear_model import LogisticRegression

mem = np.zeros((3,5))
LR = LogisticRegression(penalty='l2', dual=False, tol=0.0001, fit_intercept=True, solver='lbfgs', random_state=True, max_iter=100)
# default amount of iterations is not sufficent, an other solver (optimizer) than the lbfgs (refers to Broyden–Fletcher–Goldfarb–Shanno algorithm) would go beyond the time frame 
# Prefer dual=False when n_samples > n_features.
# usage of pre-processed data (normalized) beaucse cross-validation performance is better and significantly fewer iteration steps. From over 5000 to less than 100 now! 
for i, (train_index, test_index) in enumerate(skf.split(Xtrain_n, Ytrain)):   #Perform 5-fold stratified cross-validation
         Xtr, Xv = Xtrain_n[train_index], Xtrain_n[test_index]
         Ytr, Yv = Ytrain[train_index], Ytrain[test_index]
         LR.fit(Xtr, Ytr)
         A = LR.predict(Xv)
         pred_acc = accuracy_score(Yv,LR.predict(Xv)) #calculating prediction accuracy for each fold        
         auc_roc = roc_auc_score(Yv,LR.predict(Xv))   #calculating ROC-AUC for each fold
         auc_pr = average_precision_score(Yv,LR.predict(Xv)) #calculating AUC-PR for each fold
         mem[0,i], mem[1,i], mem[2,i] = pred_acc, auc_roc, auc_pr
         #print('| %i. Fold\t| Predicted accuracy = %1.4f\t| AUC-ROC = %1.4f\t| AUC-PR = %1.4f\t|' % (i+1, pred_acc, auc_roc, auc_pr))
print("\nLogisticRegression with C=1 (default)")
print("| cross validation results\t\t| Predicted accuracy \t\t| AUC-ROC t\t\t\t| AUC_PR \t\t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (np.mean(mem[0]), np.mean(mem[1]), np.mean(mem[2])))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (np.std(mem[0]), np.std(mem[1]), np.std(mem[2])))

#optimize C now with grid search
#GridSearchCV is inefficent and more like a brute force, trying a lot of parameters
searchpara = [0.1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144] #the parameters according to fibunacci sequence, values lower than one are do not lead an encrease of precesion
n= len(searchpara)
accuracy = np.zeros((n,2))
auc_roc = np.zeros((n,2))
auc_pr = np.zeros((n,2))

for c in enumerate(searchpara):
    sp = searchpara[c[0]]
    c = c[0]
    LR = LogisticRegression(C=sp, max_iter=1000) #default iteration steps are not enough even though I am using the normalized data
    acc = cross_val_score(LR, Xtrain_n, Ytrain, scoring = 'accuracy', cv=skf)
    ar = cross_val_score(LR, Xtrain_n, Ytrain, scoring = 'roc_auc', cv=skf)
    ap = cross_val_score(LR, Xtrain_n, Ytrain, scoring = 'average_precision', cv=skf)
    accuracy[c,0],  accuracy[c,1] = np.mean(acc), np.std(acc)
    auc_roc[c,0], auc_roc[c,1] = np.mean(ar), np.std(ar)
    auc_pr[c,0], auc_pr[c,1] = np.mean(ap), np.std(ap)

print("\nLogisticRegression with C = %i such that AUC-PR is optimal" % (searchpara[np.argmax(auc_pr[:,0])]))
print("| cross validation results\t\t Best predicted accuracy\t| Best AUC-ROC \t\t| Best AUC-PR \t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % ( np.max(accuracy[:,0]), np.max(auc_roc[:,0]), np.max(auc_pr[:,0])))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (accuracy[np.argmax(accuracy[:,0]),1], auc_roc[np.argmax(auc_roc[:,0]),1], auc_pr[np.argmax(auc_pr[:,0]),1]))
print('| corresponding best parameter\t| C = %i \t\t\t\t\t| C = %i\t\t\t\t| C = %i\t\t\t|' % (searchpara[np.argmax(accuracy[:,0])], searchpara[np.argmax(auc_roc[:,0])], searchpara[np.argmax(auc_pr[:,0])]))

print("\nLogisticRegression with C = %i such that AUC-PR is optimal" % (searchpara[np.argmax(auc_pr[:,0])]))
print("| cross validation results\t\t Best predicted accuracy\t| Best AUC-ROC \t\t| Best AUC-PR \t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % ( accuracy[np.argmax(auc_pr[:,0]),0], auc_roc[np.argmax(auc_pr[:,0]),0], np.max(auc_pr[:,0])))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (accuracy[np.argmax(auc_pr[:,0]),1], auc_roc[np.argmax(auc_pr[:,0]),1], auc_pr[np.argmax(auc_pr[:,0]),1]))
print('| corresponding best parameter\t| C = %i \t\t\t\t\t| C = %i\t\t\t\t| C = %i\t\t\t|' % (searchpara[np.argmax(accuracy[:,0])], searchpara[np.argmax(auc_roc[:,0])], searchpara[np.argmax(auc_pr[:,0])]))

#%% Linear SVM & KernelizedSVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

searchpara = [{'kernel': ['linear'], 'C': [0.1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]}] #first eleven numbers of fibunacci sequence and one lower than 1
gs = GridSearchCV(SVC(), searchpara, scoring='average_precision', cv= skf) #using gridsearch to find best parameter for C, such that average precision is optimized
gs.fit(Xtrain_n, Ytrain)
lsvm = SVC(C = gs.best_params_['C'], kernel = 'linear', class_weight = 'balanced')
m_acc, std_acc = np.mean(cross_val_score(lsvm, Xtrain_n, Ytrain, scoring = 'accuracy', cv=skf)),  np.std(cross_val_score(lsvm, Xtrain_n, Ytrain, scoring = 'accuracy', cv=skf))
m_ar, std_ar = np.mean(cross_val_score(lsvm, Xtrain_n, Ytrain, scoring = 'roc_auc', cv=skf)), np.std(cross_val_score(lsvm, Xtrain_n, Ytrain, scoring = 'roc_auc', cv=skf))
m_ap, std_ap = np.mean(cross_val_score(lsvm, Xtrain_n, Ytrain, scoring = 'average_precision', cv=skf)), np.std(cross_val_score(lsvm, Xtrain_n, Ytrain, scoring = 'average_precision', cv=skf))

print("\nLinear Support Vector Machine with C = %i such that AUC-PR is optimal" % (gs.best_params_['C']))
print("| cross validation results\t\t| predicted accuracy\t\t| AUC-ROC \t\t\t| Best AUC-PR \t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (m_acc, m_ar, m_ap))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (std_acc, std_ar, std_ap))

#%% Kernelized SVM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

searchpara = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4],'C': [0.001, 0.1, 1, 2, 3, 5, 8, 13, 21, 34, 55],'class_weight': ['balanced']},
              {'kernel': ['poly'], 'coef0':[0,1] ,'gamma': [1e-2, 1e-3, 1e-4], 'degree': [0.5, 2, 3],'C': [0.001, 0.1, 1, 2, 3, 5, 8, 13, 21, 34, 55],'class_weight': ['balanced']},
              {'kernel': ['sigmoid'], 'coef0':[0,1] ,'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.001, 0.1, 1, 2, 3, 5, 8, 13, 21, 34, 55],'class_weight': ['balanced']}]
#analyzed kernels: RBF, Polynomial and Sigmoid
gs = GridSearchCV(SVC(), searchpara, scoring='average_precision', cv= skf) #overwrite gs from linear svm with gridsearch
# to find best parameter for C and the best Kernel, such that average precision is optimized
gs.fit(Xtrain_n, Ytrain)

print(gs.best_params_)
ksvm = SVC(C = gs.best_params_['C'], degree = gs.best_params_['degree'], gamma =  gs.best_params_['gamma'], kernel = gs.best_params_['kernel'], class_weight = 'balanced')
m_acc, std_acc = np.mean(cross_val_score(ksvm, Xtrain_n, Ytrain, scoring = 'accuracy', cv=skf)),  np.std(cross_val_score(ksvm, Xtrain_n, Ytrain, scoring = 'accuracy', cv=skf))
m_ar, std_ar = np.mean(cross_val_score(ksvm, Xtrain_n, Ytrain, scoring = 'roc_auc', cv=skf)), np.std(cross_val_score(ksvm, Xtrain_n, Ytrain, scoring = 'roc_auc', cv=skf))
m_ap, std_ap = np.mean(cross_val_score(ksvm, Xtrain_n, Ytrain, scoring = 'average_precision', cv=skf)), np.std(cross_val_score(ksvm, Xtrain_n, Ytrain, scoring = 'average_precision', cv=skf))

print("\nKernelized Support Vector Machine with ",gs.best_params_," such that AUC-PR is optimal")
print("| cross validation results\t\t| predicted accuracy\t\t| AUC-ROC \t\t\t| Best AUC-PR \t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (m_acc, m_ar, m_ap))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (std_acc, std_ar, std_ap))


