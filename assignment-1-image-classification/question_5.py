# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:48:56 2021

@author: Felipe
"""
#import packages
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

#import data
Xtest = np.loadtxt("Xtest.txt") #Xtrain: Training Data (each row is a single image)
Xtrain = np.loadtxt("Xtrain.txt") #Ytrain: Training labels
Ytrain = np.loadtxt("Ytrain.txt") #Xtest: Test labels (each row is a single image)


#%%

skf = StratifiedKFold(shuffle = True, n_splits = 8) #cross-validation over 8 Folds to get a more robust result
pca = PCA()
svc = SVC()

pipe = Pipeline(steps=[('scale', StandardScaler()), #our pipeline with the following steps: Stanadtize, find sufficient dimension using pca and kernelized SVM as classification tool
                       ('pca', pca), ('svc', svc)])


searchpara = [{'pca__n_components': [96, 144, 256, 314],'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-1, 1],
               'svc__C': [0.1, 1, 2, 3, 5, 8, 13, 34],'svc__class_weight': ['balanced']}, 
             {'pca__n_components': [96, 144, 256, 314],'svc__kernel': ['poly'], 'svc__coef0':[0,1],
               'svc__gamma': [1e-3, 1e-1, 1], 'svc__degree': [2, 3],'svc__C': [0.1, 1, 2, 3, 5, 8, 13, 34],
               'svc__class_weight': ['balanced']}] #determine optimal hyperparameters

estimator = GridSearchCV(pipe, searchpara, scoring='f1', cv= skf,  n_jobs=-1, verbose= 2) #here we use the F as performence metric
estimator.fit(Xtrain, Ytrain)
print("The best parameters: {0}".format(estimator.best_params_))
pipe.set_params(**estimator.best_params_)
pipe.fit(Xtrain, Ytrain)
#same procedure as always 
m_acc, std_acc = np.mean(cross_val_score(pipe, Xtrain, Ytrain, scoring = 'accuracy', cv=skf)),  np.std(cross_val_score(pipe, Xtrain, Ytrain, scoring = 'accuracy', cv=skf))
m_ar, std_ar = np.mean(cross_val_score(pipe, Xtrain, Ytrain, scoring = 'roc_auc', cv=skf)), np.std(cross_val_score(pipe, Xtrain, Ytrain, scoring = 'roc_auc', cv=skf))
m_ap, std_ap = np.mean(cross_val_score(pipe, Xtrain, Ytrain, scoring = 'average_precision', cv=skf)), np.std(cross_val_score(pipe, Xtrain, Ytrain, scoring = 'average_precision', cv=skf))


print("\nI used a Kernelized Support Vector Machine with with the properties: ",estimator.best_params_)
print("| cross validation results\t\t| predicted accuracy\t\t| AUC-ROC \t\t\t| Best AUC-PR \t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (m_acc, m_ar, m_ap))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (std_acc, std_ar, std_ap))

Ypred = pipe.predict(Xtest) #saving the predicted Labels in the file Ypred.txt
np.savetxt("Ypred.txt", Ypred, fmt= '%i')