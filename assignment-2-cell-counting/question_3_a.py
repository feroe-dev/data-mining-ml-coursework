# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 18:11:10 2021

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
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D, experimental
from keras.callbacks import History 
import tensorflow_addons as tfa
from tensorflow_addons.layers import Maxout
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental import preprocessing
import glob
import itertools
import pickle
import shutil
import random
import warnings
import matplotlib.pyplot as plt

#%% Data Preprocessing

#warnings.filterwarnings("ignore", category=FutureWarning) #ignore warnings for the first to keep everythin clear and to
                                  #avoid unnecessary "spam".

print('TensorFlow Version:', tf.__version__) #Checking TF-Version
A = tf.config.experimental.list_physical_devices('GPU')
print('Num GPUs available:', len(A)) #asking for available GPUs

"""
preprocess the data with vgg16! "preprocessing_function= tf.keras.applications.vgg16.preprocess_input"
Details: (https://arxiv.org/pdf/1409.1556.pdf & https://neurohive.io/en/popular-networks/vgg16/)
& batching each fold with into size 12
- might use  preprocessing.Normalization() for preprocessing via Normalization or
ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
- might rescale=(1./255) use to stanardizise
"""
batch_size = 12 #default batch size for a reason: A good tradeoff between convergence speed and noise
epochs = 100


train_batches = ImageDataGenerator(preprocessing_function= tf.keras.applications.vgg16.preprocess_input)\
    .flow(X0, Y0[:,0], batch_size = batch_size)
val_batches = ImageDataGenerator(preprocessing_function= tf.keras.applications.vgg16.preprocess_input)\
    .flow(X1,  Y1[:,0], batch_size = batch_size)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow(X2,  Y2[:,0], batch_size = batch_size, shuffle=False)
    

#Show Images in batch
imgs, labels = next(train_batches)

def plotImages(images_arr):
     fig, axes = plt.subplots(1, 10, figsize=(20,20))
     axes = axes.flatten()
     for img, ax in zip( images_arr, axes):
         ax.imshow(img)
         ax.axis('off')
     plt.tight_layout()
     plt.show()
 
plotImages(imgs)
print(labels)


#%% Representation
"""
Sources for choosing filters and layers: [1]: https://datascience.stackexchange.com/questions/51470/what-are-the-differences-between-convolutional1d-convolutional2d-and-convoluti#:~:text=Conv2D%20is%20used%20for%20images,frame%20for%20each%20time%20span.

Using hidden layer Conv2D with 32x3 two-demensional filter                       
"""
inputshape = X0[0,:,:,:].shape
model = Sequential()

#Representation
model.add(Conv2D(filters = 32, kernel_size=(3, 3), input_shape = inputshape, activation='relu', #1st hidden layer
                 padding = 'same', kernel_initializer='RandomUniform'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2)) #2n hidden layer
model.add(Conv2D(filters = 64, kernel_size=(3, 3), activation='relu', #3rd hidden layer
                 padding = 'same', kernel_initializer='RandomUniform'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2)) #4th hidden layer
model.add(Conv2D(filters = 128, kernel_size=(3, 3), activation='relu', #5th hidden layer
                 padding = 'same', kernel_initializer='RandomUniform'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))
model.add(Maxout(num_units = 1))
model.add(Flatten())
model.add(Dense(1))
model.summary()


#%%Optimitazion

model.compile(loss = 'mse', optimizer=Adam(learning_rate = 0.0001), metrics=['accuracy'])
history = model.fit(x = train_batches, validation_data = val_batches, epochs = epochs, shuffle = True,
                    verbose = 2, steps_per_epoch = len(train_batches),
                    validation_steps = len(val_batches))    

#%% vgg16
"""
vgg16model = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(256,256,3))
#vgg16model.summary()

adjvgg16model = Sequential()
for layer in vgg16model.layers[:-1]:
    adjvgg16model.add(layer)
#adjvgg16model.summary() #same as vgg16model without last layer, we want only a single output there
adjvgg16model.add(Flatten())
adjvgg16model.add(Dense(4096, activation=('relu')))
adjvgg16model.add(Dense(4096, activation=('relu')))

for layer in adjvgg16model.layers:
    layer.trainable = False

epochs = 5

adjvgg16model.add(Dense(1, activation=('relu')))
adjvgg16model.summary()

adjvgg16model.compile(loss = 'mse', optimizer=Adam(learning_rate = 0.0001), metrics=['accuracy'])
history = adjvgg16model.fit(x = train_batches, validation_data = val_batches, epochs = epochs, shuffle = True,
                    verbose = 2, steps_per_epoch = len(train_batches),
                    validation_steps = len(val_batches))
"""
#%%Evaluation

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

history = load_obj('History1')

fig= plt.figure(figsize=(16,9))
ax = fig.add_subplot(1,2,1)


#Convergence plot
ax.plot(history['loss'], label='loss'); ax.plot(history['val_loss'], label='val_loss')
ax.set_title('Convergence plot'); ax.set_xlim([0, epochs]); ax.set_ylim([0, 175]); ax.set_xlabel('Epoch')
ax.set_ylabel('Mean Squared Error'); ax.legend(); ax.grid(True)

#Scatter plot
ax = fig.add_subplot(1,2,2)
pred = model.predict(x= test_batches, verbose=0)
pred[pred<0] = 0 #prevend values lower than zero and set those to zero
ax.plot(Y2[:,0], pred,'o');ax.grid();ax.set_xlabel('True value');ax.set_ylabel('Predicted value');
ax.set_title('Regression Scatter Plot\n true vs predicted cell counts')

#Metrics
from sklearn import metrics
from scipy import stats
pred = np.reshape(pred,(len(pred),))
print('| Model: CNN\t| RMSE\t\t| Pearsons CC\t| Spearmans CC\t| R2 score\t|')
print('------------------------------------------------------------------------')
print('| \t\t\t\t| %1.3f\t| %1.4f\t\t| %1.3f\t\t| %1.3f\t|' % (np.sqrt(metrics.mean_squared_error(Y2[:,0], pred)),
                                                            stats.pearsonr(Y2[:,0], pred)[0],
                                                            stats.spearmanr(Y2[:,0], pred)[0],
                                                            metrics.r2_score(Y2[:,0], pred)))

#%% pickle

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

save_obj(history.history, 'History2')