# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:36:44 2021

@author: Felipe
"""
import numpy as np
import matplotlib.pyplot as plt


Xtest = np.loadtxt("Xtest.txt") #Xtrain: Training Data (each row is a single image)
Xtrain = np.loadtxt("Xtrain.txt") #Ytrain: Training labels
Ytrain = np.loadtxt("Ytrain.txt") #Xtest: Test labels (each row is a single image)

#%% framework
from sklearn import preprocessing
import matplotlib.colors
stand = preprocessing.StandardScaler().fit(Xtrain) #preporcessing via Standardtization
Xtrain_st = stand.transform(Xtrain)
normalizer = preprocessing.Normalizer().fit(Xtrain) #default=’l2’ l2 refers to the  Euclidean norm or l^2 norm - here we normalized Xtrain
Xtrain_n = normalizer.transform(Xtrain)

"""
for i in range (1,10):#test for myself 
    plt.figure()
    plt.scatter(Xtrain[:, 0], Xtrain[:, i], #r1th and second r2th component vizulalited (r1 and r2 choosen randomly)and reducing dimension from 784 to 2
                c = Ytrain, edgecolor='face', alpha=0.2, cmap= matplotlib.colors.ListedColormap(['red', 'blue'])
                );
"""
#%% i)
from sklearn.decomposition import PCA #import PCA

"""
procedure without sklearn.decomposition leads to same outcome!
ev, pc = np.linalg.eig(cm)
ev = np.abs(ev)
z = np.dot(Xtrain_st,pc) #projection
"""
pca784 = PCA(n_components=784) #fitting pca to training data
pca784.fit(Xtrain) #training PCA - here we do not standardtize, regarding a more vivid scatter plot in ii)
projected = pca784.transform(Xtrain) #projection
a = pca784.get_covariance()
print("The dimensions of X are",Xtrain.shape)
print("Covariance matrix Dimensionality is: ",a.shape)

r1 = 0 #np.random.randint(0,784)
r2 = 1 #np.random.randint(0,784)
if r1 == r2:
    r2 = r1+1
plt.figure() #vizualitation with scatter plot
plt.scatter(projected[:, r1], projected[:, r2], #r1th and second r2th component vizulalited (r1 and r2 choosen randomly)and reducing dimension from 784 to 2
            c = Ytrain, edgecolor='face', alpha=0.2, cmap= matplotlib.colors.ListedColormap(['red', 'blue'])
            );
plt.grid()
plt.xlabel('PC%i'%(1))
plt.ylabel('PC%i'%(2))
plt.colorbar();
plt.show()
# =============================================================================
# Unfortunately, the scatter plot hardly gives us any information
# The points points are clustered around the origin. The density of points increases the closer you get to the origin.
# What we can say, however, is that these feature vectors are quite similar.
# If you look closely, you can see that the density of the red labels (i.e. Y=-1) does not decrease as quickly as the blue labels (i.e. Y=1) when you move away vertically from the origin. 
# =============================================================================
#%% ii)
"""
plt.figure()
plt.plot(pca784.explained_variance_); plt.grid()
plt.xlabel('Explained Variance')
plt.ylabel('Eigenvalues')
plt.yscale('log') #log yscale is reasonable in order to capture the whole graph
"""

plt.figure()
plt.scatter(314.4, 0.95, s=500, marker='|', color ='red',alpha = 1) #min level
plt.scatter(315, 0.950155, s=500, marker='x', color ='red',alpha = 1) #nearest greater point
plt.plot(np.arange(len(pca784.explained_variance_ratio_))+1,np.cumsum(pca784.explained_variance_ratio_),'o-') #plot the scree graph
plt.axis([1,len(pca784.explained_variance_ratio_),0,1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.title('Scree Graph')
plt.grid()
plt.show()

print("\nA dimension of 314 explain",round(np.cumsum(pca784.explained_variance_ratio_)[314],4),"of the varinace in the training set,")
print("whereas a dimension of 313 explain",round(np.cumsum(pca784.explained_variance_ratio_)[313],4),"of the varinace in the training set")
# By reading off and zooming in many times, one obtains that the dimension should be at least 314 (i.e. 315 or more)
# in order to be able to explain at least 95% of the variance.
#%% iii)
d = 314 
# =============================================================================
# from ii) I would also take a similiar number.
# A good insight would give us the derivative of the scree graph. Well we cannot calculate it conretly
# but how it would look like, if you imagine that the first derivative represents the slope of a function.
# For the sake of simplicity, I will leave out the aspect of continuity here. First of all,
# we can see that the slope is positive everywhere and due to the asymptotic behaviour in the direction of y = 1,
# we can say that the slope decreases monotonically and even exponentially. It would make little sense to choose too few dimensions
# because the loss of information would be enormous compared to taking only one dimension more. However,
# if you look at values above 200, you will see that the difference between the dimensions becomes smaller and smaller.
# I wanted to express this effect with my first two sentences
# =============================================================================


pca784s = PCA(n_components=d) #fitting pca to training data
pca784s.fit(Xtrain_st)
Z = pca784s.transform(Xtrain_st) #projection
pc = pca784s.components_[0:d]  #principal component matrix
cm = pca784s.get_covariance() #covariance-matrix
ev = pca784s.explained_variance_ #eigen vector of covariance matrix of Xtrain

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

skf = StratifiedKFold(shuffle = True)

searchpara = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [0.001, 0.1, 1, 2, 3, 5, 8, 13, 21, 34],'class_weight': ['balanced']},
              {'kernel': ['poly'], 'coef0':[0,1] ,'gamma': [1e-3, 1e-4], 'degree': [0.5, 2, 3],'C': [0.001, 0.1, 1, 2, 3, 5, 8, 13, 21, 34],'class_weight': ['balanced']},
              {'kernel': ['sigmoid'], 'coef0':[0,1] ,'gamma': [1e-3, 1e-4], 'C': [0.001, 0.1, 1, 2, 3, 5, 8, 13, 21, 34],'class_weight': ['balanced']}]
#analyzed kernels: RBF, Polynomial and Sigmoid
gs = GridSearchCV(SVC(), searchpara, scoring='average_precision', cv= skf) #overwrite gs from linear svm with gridsearch
# to find best parameter for C and the best Kernel, such that average precision is optimized

gs.fit(Z, Ytrain)
try:
    gs.best_params_['degree']
    ksvm = SVC(C = gs.best_params_['C'], degree = gs.best_params_['degree'], gamma =  gs.best_params_['gamma'], kernel = gs.best_params_['kernel'], class_weight = 'balanced')
except KeyError:    
    ksvm = SVC(C = gs.best_params_['C'], gamma =  gs.best_params_['gamma'], kernel = gs.best_params_['kernel'], class_weight = 'balanced')
m_acc, std_acc = np.mean(cross_val_score(ksvm, Z, Ytrain, scoring = 'accuracy', cv=skf)),  np.std(cross_val_score(ksvm, Z, Ytrain, scoring = 'accuracy', cv=skf))
m_ar, std_ar = np.mean(cross_val_score(ksvm, Z, Ytrain, scoring = 'roc_auc', cv=skf)), np.std(cross_val_score(ksvm, Z, Ytrain, scoring = 'roc_auc', cv=skf))
m_ap, std_ap = np.mean(cross_val_score(ksvm, Z, Ytrain, scoring = 'average_precision', cv=skf)), np.std(cross_val_score(ksvm, Z, Ytrain, scoring = 'average_precision', cv=skf))

print("\nKernelized Support Vector Machine with ",gs.best_params_," such that AUC-PR is optimal")
print("| cross validation results\t\t| predicted accuracy\t\t| AUC-ROC \t\t\t| Best AUC-PR \t\t|")
print('---------------------------------------------------------------------------------------------------')
print('| Total\t\t\t\t\t\t\t| mean = %1.4f \t\t\t| mean = %1.4f\t\t| mean = %1.4f\t\t| ' % (m_acc, m_ar, m_ap))
print('| Total\t\t\t\t\t\t\t| std = %1.4f \t\t\t\t| std = %1.4f\t\t| std = %1.4f\t\t| ' % (std_acc, std_ar, std_ap))

