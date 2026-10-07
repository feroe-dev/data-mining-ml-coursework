# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:04:33 2021

@author: Felipe
"""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

Xtest = np.loadtxt("Xtest.txt") #Xtrain: Training Data (each row is a single image)
Xtrain = np.loadtxt("Xtrain.txt") #Ytrain: Training labels
Ytrain = np.loadtxt("Ytrain.txt") #Xtest: Test labels (each row is a single image)

#%% framework
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

skf = StratifiedKFold(shuffle = True) #default fold=5 & shuffle = true to get an unbiased result
#sknn = KNeighborsClassifier(n_neighbors=1)  #initialise earest Neighbor Classifier with k=1

def k5foldcrossvalper(A,B,k, Print = True):   #A = Training feature space, B = Training labels, k parameter for KNN, P query whether to print [P=1] or not [P=0]
    sknn = KNeighborsClassifier(n_neighbors=k) #initialise nearest Neighbor Classifier dependant on k
    if Print == True:
        print("Folds =",skf.get_n_splits(A, B))  #checking the fold
    mem = np.zeros((3,5))
    for i, (train_index, test_index) in enumerate(skf.split(A, B)):   #Perform 5-fold stratified cross-validation
         Xtr, Xv = A[train_index], A[test_index]
         Ytr, Yv = B[train_index], B[test_index]
         sknn.fit(Xtr,Ytr)
         pred_acc = accuracy_score(Yv,sknn.predict(Xv)) #calculating prediction accuracy for each fold
         auc_roc = roc_auc_score(Yv,sknn.predict(Xv))   #calculating ROC-AUC for each fold
         auc_pr = average_precision_score(Yv,sknn.predict(Xv)) #calculating AUC-PR for each fold
         mem[0,i], mem[1,i], mem[2,i] = pred_acc, auc_roc, auc_pr
         if Print == True:
             print('| %i. Fold | Predicted accuracy = %1.4f | AUC-ROC = %1.4f | AUC-PR = %1.4f |' % (i+1, pred_acc, auc_roc, auc_pr))
    return mem

if __name__ == "__main__":
#%% i)
    mem = k5foldcrossvalper(Xtrain,Ytrain,1,True)
#%% ii)
    
    mean_pa, sd_pa = np.mean(mem[0]), np.std(mem[0])
    print('| Prediction accuracy: \t Mean = %1.4f | standard deviation = %1.4f |' % (mean_pa, sd_pa))
    mean_ar, sd_ar = np.mean(mem[1]), np.std(mem[1])
    print('| AUC-ROC: \t\t\t\t Mean = %1.4f | standard deviation = %1.4f |' % (mean_ar, sd_ar))
    mean_ap, sd_ap = np.mean(mem[2]), np.std(mem[2])
    print('| AUC-PR: \t\t\t\t Mean = %1.4f | standard deviation = %1.4f |' % (mean_ap, sd_ap))
    
    #%% iii)
    
    from sklearn import preprocessing
    import time as time
    import signal
    
    # =============================================================================
    # According to the provided source (https://scikitlearn.org/stable/modules/preprocessing.html), there are various of froms of pre-processing. In principle, under pre-processing
    # is understood the transformation of a raw feature space in to a more manageable or suitable one. In the following I will introduce you to three of those pre-processing transforamtions. 
    # =============================================================================
    
    #Standardization via standardscalar
    print("1.) Standardization")
    # =============================================================================
    #  "they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance."
    # =============================================================================
    stand = preprocessing.StandardScaler().fit(Xtrain)
    r = np.random.randint(0,778)
    print("Mean of 5 random features: ", stand.mean_[r:r+5])
    Xtr_scaled = stand.transform(Xtrain) #sclaing each example in Xtrain , scaledfeature = (feature - mean(features)) / standarddeviation(features)
    print("Mean of the 5 the scaled random features: ", Xtr_scaled.mean(axis=0)[r:r+5])
    print("Std of the 5 the scaled random features: ", Xtr_scaled.std(axis=0)[r:r+5])
    #Here you can already see that the mean of those scaled features are not equal to really zero and that the std is not really one, that is a hint that the features are not normal distributed around its mean.
    #Checking cross-validation performance
    StPerform = k5foldcrossvalper(Xtr_scaled, Ytrain,1,True)
    # =============================================================================
    # Just by viewing on the result I conclude that the Predicted accuracy increased marginally and AUC-ROC & AUC-PR stayed nearly the same. An accurate insight would result in a hypothesis test 
    # =============================================================================  
    #Generating polynomial features
    print("2.) Generating polynomial features")
    # =============================================================================
    # "Often it’s useful to add complexity to the model by considering nonlinear features of the input data. A simple and common method to use is polynomial features,
    # which can get features’ high-order and interaction terms"
    # =============================================================================
    
    poly = preprocessing.PolynomialFeatures(3) #Are our features maybe complexer and follow polynomialring of the grade 3?
    try:    #testing for MemoryError
        poly.fit_transform(Xtrain) 
    except MemoryError:
       print("Degree = 3 delivers MemoryError")
    # This results in an error because the the transformed feature space would be too big to handle for a commercially available CPU. To be precise an array of the form (3000, 80931145).
    # okay so lets use "only" a degree of 2
    
    poly = preprocessing.PolynomialFeatures(2) #Are our features maybe complexer and follow polynomialring of the grade 2?
    try:    #testing for MemoryError
        poly.fit_transform(Xtrain)
        print("Well, we are able to tansform the data, but are we going to be able to classify it?")
        start_time = time.time()
        #PolyPred = k5foldcrossvalper(poly.fit_transform(Xtrain) , Ytrain, 1, True)
        end_time = time.time()
        time_lapsed = end_time - start_time
        if time_lapsed > 1:
            print("Well we terminated it, but at what price? I took us",time_lapsed,"hours")
    except MemoryError:
        print("Degree = 2 delivers MemoryError")
       
    #build timer so this does not run till day of judgement
    """
    try:
       loop(10) 
       PolyPred = k5foldcrossvalper(poly.fit_transform(Xtrain) , Ytrain, 1, True)
    except TimeOutException and MemoryError:
       print("Cannot manage to classify")
    """
       
    #same here
    #here dont even get the chance to calculate the cross-validation performance, therefore unusable in our case
    
    #Normalization
    print("3.) Normalization")
    # =============================================================================
    # "Normalization is the process of scaling individual samples to have unit norm.
    # This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.
    # =============================================================================
    
    normalizer = preprocessing.Normalizer().fit(Xtrain) #default=’l2’ l2 refers to the  Euclidean norm or l^2 norm - here we normalized Xtrain
    Xtr_normalized = normalizer.transform(Xtrain) #Normalization of Xtrain on Xtrain
    NormPerform = k5foldcrossvalper(Xtr_normalized, Ytrain,1,True)
    
    
    #%% iv)
    import matplotlib.pyplot as plt
    
    means = np.zeros((34,4))
    stds = np.zeros((34,3))
    
    for k in range(1,35): #k element [1,30], I choose this interval because sqrt(784) = 28 .
        #print("k= ",k)
        mem2 = k5foldcrossvalper(Xtr_normalized,Ytrain,k, False) #excecuting knn  for each k and saving their perfomance in mem2 for each k
        means[k-1,0] = int(k)
        means[k-1,1], stds[k-1,0] = np.mean(mem2[0]), np.std(mem2[0]) #calculating for each k the mean and the standard deviation of the prediction accuracy, AUC-ROC and AUC-PR over the 5 folds.
        means[k-1,2], stds[k-1,1] = np.mean(mem2[1]), np.std(mem2[1])
        means[k-1,3], stds[k-1,2] = np.mean(mem2[2]), np.std(mem2[2])
    
    #plotting a graph of prediction accuracy, AUC-ROC and AUC-PR dependant on k
    fig, ((m1, m2, m3), (st1, st2, st3)) = plt.subplots(2, 3,figsize=(16,9))
    #plt.subplot(131)
    
    m1.plot(means[:,0], means[:,1])
    m1.scatter(np.argmax(means[:,1])+1, np.amax(means[:,1]), s=10, marker='o', color ='red')
    m1.grid(True)
    m1.legend(('mean of prediction accuracy of k over Folds','max point'))
    m1.set(xlabel='k')
    
    st1.plot(means[:,0], stds[:,0])
    st1.scatter(np.argmax(means[:,1])+1, stds[np.argmax(means[:,1]),0], s=10, marker='o', color ='red')
    st1.grid(True)
    st1.legend(('Std of prediction accuracy of k over Folds','max point'),loc="upper right")
    st1.set(xlabel='k')
    #plt.subplot(132)
    
    m2.plot(means[:,0], means[:,2])
    m2.scatter(np.argmax(means[:,2])+1, np.amax(means[:,2]), s=10, marker='o', color ='red')
    m2.grid(True)
    m2.legend(('mean of AUC-ROC of k over Folds','max point'))
    m2.set(xlabel='k')
    
    st2.plot(means[:,0], stds[:,1])
    st2.scatter(np.argmax(means[:,2])+1, stds[np.argmax(means[:,2]),0], s=10, marker='o', color ='red')
    st2.grid(True)
    st2.legend(('Std of AUC-ROC of k over Folds','max point'),loc="upper right")
    st2.set(xlabel='k')
    #plt.subplot(133)                   
    
    m3.plot(means[:,0], means[:,3])
    m3.scatter(np.argmax(means[:,3])+1, np.amax(means[:,3]), s=10, marker='o', color ='red')
    m3.grid(True)
    m3.legend(('mean of AUC-PR of k over Folds','max point'),loc="lower left")
    m3.set(xlabel='k')
    
    st3.plot(means[:,0], stds[:,2])
    st3.scatter(np.argmax(means[:,3])+1, stds[np.argmax(means[:,3]),0], s=10, marker='o', color ='red')
    st3.grid(True)
    st3.legend(('Std of AUC-PR of k over Folds','max point'),loc="lower left")
    st3.set(xlabel='k')
        
    print("max point of prediction accuracy: k= %i Mean(PA)= %1.4f" % (np.argmax(means[:,1])+1, np.amax(means[:,1])))
    print("max point of AUC-ROC: k= %i Mean(AUC-ROC)= %1.4f" % (np.argmax(means[:,2])+1, np.amax(means[:,2])))
    print("max point of AUC-PR: k= %i Mean(AUC-PR)= %1.4f" % (np.argmax(means[:,3])+1, np.amax(means[:,3])))
    # =============================================================================
    # As you see in the plot I would recommend a k element [6,9]. Here the mean of prediction accuracy, AUC-ROC and AUC-PR are at their highest.
    # Finally I would choose k=6 because here we have the maximum of AUC-PR, and only marginal loses in prediction accuracacy (crossvalidation accuracy)
    # and AUC-ROC. The arguments from Q1 v) support my argumentation. The standard deviation at k=6 is also very low. This lets our result be more accurate
    # =============================================================================
    k5foldcrossvalper(Xtr_normalized,Ytrain,np.argmax(means[:,1])+1, True)