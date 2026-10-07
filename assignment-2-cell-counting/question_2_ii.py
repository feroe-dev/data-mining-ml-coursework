# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 00:55:58 2021

@author: Felipe
"""
import numpy as np
data = np.load("cell-data.npz")
images = data["images"]
counts = data["counts"]
folds = data["folds"]

X0 = images[folds == 0]
Y0 = counts[folds == 0]
X1 = images[folds == 1] #Using the 2nd Fold for validation (Fold = 1)
Y1 = counts[folds == 1]

from skimage.color import rgb2hed
import skimage.measure 
#%%
#load saved variables
import dill #relaoding the old variables of the kernel
dill.load_session('PickleAssignment2.db')

#%% ii

def featuremem(imgs):
    memory = np.zeros((len(imgs),4,3)) #the results are getting saved here
    for j,im in enumerate(imgs):    
        memory[j,0,0] = np.mean(rgb2hed(im)[:,:,0],axis= (0,1)) #mean of H-channel of image j
        memory[j,0,1] = np.var(rgb2hed(im)[:,:,0],axis= (0,1)) #mean of H-channel of image j
        memory[j,0,2] = skimage.measure.shannon_entropy(rgb2hed(im)[:,:,0]) #entropy of H-Channel of image j
        for i in range (0,3):
            # cf. 
            # runs through the Red, Blue and Green Channel
            memory[j,i+1,0] = np.mean(im[:,:,i],axis= (0,1)) #mean of channel of image j
            memory[j,i+1,1] = np.var(im[:,:,i],axis= (0,1)) #mean of channel of image j
            memory[j,i+1,2] = skimage.measure.shannon_entropy(im[:,:,i]) #entropy of H-Channel of image j
    return memory

memory0 = featuremem(X0)
memory1 = featuremem(X1)
#%% OLS

import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from scipy import stats

ols = linear_model.LinearRegression(fit_intercept=True) #only FOLD = 0 (First Fold, Training sample)

features = ['Average','Variance','Entropy']
print('| Model: OLS\t| RMSE\t\t| Pearsons Corr. Coeff.\t| Spearmans Corr. Coeff.\t| R2 score\t|')

fig=plt.figure(figsize=(9,6))

for i in range (0,3):
    ols.fit(memory0[:,:,i], Y0[:,0])
    pred = ols.predict(memory1[:,:,i])
    pred[pred<0] = 0 #values lower than zero would make no sense
    ax = fig.add_subplot(1,3,i+1)
    ax.plot(Y1[:,0], pred,'o');ax.grid();ax.set_xlabel('True value');ax.set_ylabel('Predicted value');ax.set_title('Regression Scatter Plot\nfor %s' % features[i])
    print('-------------------------------------------------------------------------------------------')
    print('| %s\t\t| %1.4f\t| %1.4f\t\t\t\t| %1.4f\t\t\t\t\t| %1.4f\t|' % (features[i],np.sqrt(metrics.mean_squared_error(Y1[:,0], pred)),
                                                            stats.pearsonr(Y1[:,0], pred)[0],
                                                            stats.spearmanr(Y1[:,0], pred)[0],
                                                            ols.score(memory1[:,:,i], Y1[:,0])))
fig.tight_layout()

#%% Multilayer Perceptron

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers.experimental import preprocessing

print('TF-Version:', tf.__version__)
A = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs available:', len(A))

#preprocessing data
"""
norm0 = preprocessing.Normalization()
norm0.adapt(np.array(memory0))
"""
fig=plt.figure(figsize=(9,6))
print('| Model: OLS\t| RMSE\t\t| Pearsons Corr. Coeff.\t| Spearmans Corr. Coeff.\t| R2 score\t|')
for i in range (0,3):
    n_features = memory0.shape[1]
    model = Sequential()
    #model.add(preprocessing.Normalization()) #uncomment for preprocessing - seems to have no impact
    model.add(tf.keras.Input(shape=(n_features,)))
    model.add(Dense(units = 4,activation='linear', kernel_initializer='identity'))
    model.add(Dense(units = 827,activation='linear', kernel_initializer='RandomUniform'))
    model.add(Dense(units = 1,activation='linear', kernel_initializer='RandomUniform')) #excludes values lower than zero!
    
    model.compile(loss = 'mse',optimizer=Adam(learning_rate = 0.0001),metrics=['accuracy'])
    history = model.fit(memory0[:,:,i], Y0[:,0], epochs = 1000, shuffle = True,
                        batch_size = 46, verbose = 0)    
    pred = model.predict(memory1[:,:,i])
    pred[pred<0] = 0
    ax = fig.add_subplot(1,3,i+1)
    ax.plot(Y1[:,0], pred,'o');ax.grid();ax.set_xlabel('True value');ax.set_ylabel('Predicted value');ax.set_title('Regression Scatter Plot\nfor %s' % features[i])
    pred = np.reshape(pred,(len(pred),))
    print('-------------------------------------------------------------------------------------------')
    print('| %s\t\t| %1.4f\t| %1.4f\t\t\t\t| %1.4f\t\t\t\t\t| %1.4f\t|' % (features[i],np.sqrt(metrics.mean_squared_error(Y1[:,0], pred)),
                                                            stats.pearsonr(Y1[:,0], pred)[0],
                                                            stats.spearmanr(Y1[:,0], pred)[0],
                                                            metrics.r2_score(Y1[:,0], pred)))
fig.tight_layout()

#%% MLP sklearn

from sklearn.neural_network import MLPRegressor

fig=plt.figure(figsize=(9,6))


for i in range (0,3):
    regr = MLPRegressor(batch_size=1, max_iter = 1000, verbose=0) #default: activation=’relu’, solver=’adam’, shuffle = True
    regr.fit(memory0[:,:,i], Y0[:,0])
    pred = regr.predict(memory1[:,:,i])
    pred[pred<0] = 0
    ax = fig.add_subplot(1,3,i+1)
    ax.plot(Y1[:,0], pred,'o');ax.grid();ax.set_xlabel('True value');ax.set_ylabel('Predicted value');ax.set_title('Regression Scatter Plot\nfor %s' % features[i])
    print('-------------------------------------------------------------------------------------------')
    print('| %s\t\t| %1.4f\t| %1.4f\t\t\t\t| %1.4f\t\t\t\t\t| %1.4f\t|' % (features[i],np.sqrt(metrics.mean_squared_error(Y1[:,0], pred)),
                                                            stats.pearsonr(Y1[:,0], pred)[0],
                                                            stats.spearmanr(Y1[:,0], pred)[0],
                                                            regr.score(memory1[:,:,i], Y1[:,0])))
fig.tight_layout()
#%% Support Vector Regression

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold

fig=plt.figure(figsize=(9,6))
skf = StratifiedKFold(shuffle = True, n_splits = 6)

for i in range (0,3):
    svr = SVR()
    pipe = Pipeline(steps=[('scale', StandardScaler()),('svr', svr)])
    searchpara = [{'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                   'svr__gamma': [1e-3, 1e-1, 1], 'svr__C': [0.001, 0.1, 1, 3, 5],
                   'svr__coef0':[0,1], 'svr__epsilon':[0.1, 0.01, 0.001, 0.0001],
                   'svr__degree':[2,3,0.5]}]
    
    estimator = GridSearchCV(pipe, searchpara, scoring='neg_mean_squared_error', cv= skf, n_jobs=-1, verbose= 0)
    estimator.fit(memory0[:,:,i], Y0[:,0])
    print("The best parameters: {0}".format(estimator.best_params_))
    pipe.set_params(**estimator.best_params_)
    pipe.fit(memory0[:,:,i], Y0[:,0])
    pred = pipe.predict(memory1[:,:,i])
    pred[pred<0] = 0
    ax = fig.add_subplot(1,3,i+1)
    ax.plot(Y1[:,0], pred,'o');ax.grid();ax.set_xlabel('True value');ax.set_ylabel('Predicted value');ax.set_title('Regression Scatter Plot\nfor %s' % features[i])
    print('-------------------------------------------------------------------------------------------')
    print('| %s\t\t| %1.4f\t| %1.4f\t\t\t\t| %1.4f\t\t\t\t\t| %1.4f\t|' % (features[i],np.sqrt(metrics.mean_squared_error(Y1[:,0], pred)),
                                                            stats.pearsonr(Y1[:,0], pred)[0],
                                                            stats.spearmanr(Y1[:,0], pred)[0],
                                                            pipe.score(memory1[:,:,i], Y1[:,0])))
fig.tight_layout()
plt.show