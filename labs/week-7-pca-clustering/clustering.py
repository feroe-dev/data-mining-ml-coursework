# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 17:28:33 2021

@author: Felipe
"""
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris = datasets.load_iris()
X = iris.data
Y = iris.target

df = pd.DataFrame(X, columns=['A','B','C','D'])
pd.plotting.scatter_matrix(df, alpha=0.8) 

#%%Part 1
from sklearn.decomposition import PCA
import matplotlib.colors

pca = PCA(n_components=2) #fitting pca to training data
pca.fit(X) #training PCA - here we do not standardtize, regarding a more vivid scatter plot in ii)
projected = pca.transform(X)

plt.figure() #vizualitation with scatter plot
plt.scatter(projected[:, 0], projected[:, 1], #r1th and second r2th component vizulalited (r1 and r2 choosen randomly)and reducing dimension from 784 to 2
            c = Y, edgecolor='face', alpha=0.2, cmap= matplotlib.colors.ListedColormap(['red', 'blue'])
            );
plt.grid()
plt.xlabel('PC%i'%(1))
plt.ylabel('PC%i'%(2))
plt.colorbar();
plt.show()
#%% Part 2
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3).fit(X)
C = kmeans.predict(X)
print('Cluster Assignment',C)
print('True Type',Y)

trans = np.c_[X,Y]

print(np.mean(X[0:49,:],0))
print(np.mean(X[50:99,:],0))
print(np.mean(X[100:149,:],0))
        