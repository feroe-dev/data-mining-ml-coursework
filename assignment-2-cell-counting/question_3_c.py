# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:24:01 2021

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
X2 = images[folds == 2] #Using the 3rd Fold for validation (Fold = 2)
Y2 = counts[folds == 2]

#%% Import Moduls
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #disable GPU - I do not have one on my device!! 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D, AveragePooling2D, experimental, LocallyConnected2D
from keras.callbacks import History 
import tensorflow_addons as tfa
from tensorflow_addons.layers import Maxout
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Model
from tensorflow.nn import pool
import glob
import itertools
import shutil
import random
import warnings
import matplotlib.pyplot as plt
from skimage.color import rgb2hed, rgb2hsv
from sklearn import metrics
from scipy import stats
#%%
for i in range (0,1):
    print('FOLD No.%i' %i)
    batch_size = 32
    fld = np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow(
         images[folds == fld[0,i]], counts[folds == fld[0,i]], batch_size = batch_size)
    valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow(
         images[folds == fld[1,i]], counts[folds == fld[1,i]], batch_size = batch_size)
    test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow(
         images[folds == fld[2,i]], counts[folds == fld[2,i]], batch_size = batch_size, shuffle=False)
    
    mobile = tf.keras.applications.mobilenet.MobileNet()
 
    x = mobile.layers[-6].output
    output = Dense(units=6, activation='relu')(x)
    model = Model(inputs=mobile.input, outputs=output)

    for layer in model.layers[:-23]:
        layer.trainable = False

    epochs = 1
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, min_delta = 1, mode='min')# baseline = 130)
    model.compile(optimizer=Adam(lr=0.0001), loss='mse', metrics=['accuracy'])
    history = model.fit(x=train_batches,
                steps_per_epoch=len(train_batches),
                validation_data=valid_batches,
                validation_steps=len(valid_batches),
                epochs=epochs, callbacks = [callback], verbose=2)
    
    #Evaluation
    fig= plt.figure(figsize=(16,9))
    fig.suptitle('Fold No.%i' %i , fontsize=20)
    
    #Convergence plot
    ax = fig.add_subplot(3,1,1)
    ax.plot(history.history['loss'], label='loss'); ax.plot(history.history['val_loss'], label='val_loss')
    ax.set_title('Convergence plot'); ax.set_xlim([0, epochs]); ax.set_ylim([0, 175]); ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean Squared Error'); ax.legend(); ax.grid(True)

    #predict
    pred = model.predict(x= test_batches, verbose=2)
    pred[pred<0] = 0 #prevend values lower than zero and set those to zero
    
    #Scatter plot    
    for j in range (0,6):
        ax = fig.add_subplot(3,3,4+j)
        ax.plot(counts[folds == fld[2,i]][:,j], pred[:,j],'o');ax.grid();ax.set_xlabel('True value');ax.set_ylabel('Predicted value');
        ax.set_title('Regression Scatter Plot\n true vs predicted cell counts for T%i-Cells' %(j+1))
    
        #Metrics
        print('| Cell type: T%i\t| RMSE\t\t| Pearsons Corr. Coeff.\t| Spearmans Corr. Coeff.| R2 score\t|' % (j+1))
        print('-------------------------------------------------------------------------------------------------')
        print('| \t\t| %1.3f\t\t| %1.3f\t| %1.3f\t\t\t| %1.3f\t\t|' % (np.sqrt(metrics.mean_squared_error(counts[folds == fld[2,i]][:,j], pred[:,j])),
                                                                    stats.pearsonr(counts[folds == fld[2,i]][:,j], pred[:,j])[0],
                                                                    stats.spearmanr(counts[folds == fld[2,i]][:,j], pred[:,j])[0],
                                                                    metrics.r2_score(counts[folds == fld[2,i]][:,j], pred[:,j])))
    fig.tight_layout()
    plt.show 



