# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 16:51:03 2021

@author: Felipe
"""
import numpy as np
data = np.load("cell-data.npz")
images = data["images"]
counts = data["counts"]
folds = data["folds"]
X0 = images[folds == 0]
Y0 = counts[folds == 0]
#%% i
for i in range (0,3):
    print('In the %i. Fold we have %i Examples' % (i+1, len(images[folds == i])))
# =============================================================================
# outputs the length of each example, which is equal to the number of examples in each fold,
# since each entry represents one example.
# =============================================================================
#%% ii
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(16,9))
plt.suptitle('Images with the most cells of a certain type', fontsize=24)
for i in range (0,6):
    ax = fig.add_subplot(2, 3, i+1)
    a = np.argmax(Y0[:,i])
    ax.imshow(images[a])
    ax.set_title('Image No. %i with bigest value for cell type T%i' % (a,i+1))    
    print('| Image No. %i\t|#T1-cells = %i\t|#T2-cells = %i\t|#T3-cells = %i\t|#T4-cells = %i\t|#T5-cells = %i\t|#T6-cells = %i\t|' % (a,Y0[a,0],Y0[a,1],Y0[a,2],Y0[a,3],Y0[a,4],Y0[a,5]))
# =============================================================================
# outputs the Image with the bigest cell counter of each Cell Type T1 - T5
# =============================================================================
#%% iii
bins = [0, 1, 6, 11, 21, 31, 41, 51, 99]
for i in range (0,3):
    fig=plt.figure(figsize=(16,9))
    plt.suptitle('Fold No.%i' %(i+1), fontsize=24)
    for j in range(0,6):
        ax = fig.add_subplot(2, 3, j+1)
        x = counts[folds == i,][:,j]
        #ax.ticklabel_format(style='plain', axis='y')
        h,e = np.histogram(x, bins=bins)
        ax.bar(range(len(bins)-1), h, width=0.8, edgecolor='k')
        labels = ['0', '1-5', '6-10','11-20', '21-30', '31-40', '41-50', '50-99']
        plt.xticks(range(len(bins)-1), labels)
        plt.yscale('log')
        ax.set_title('Histogram of counts of cell type: T%i' % (j+1))
#%% iv
from skimage.color import rgb2hed, hed2rgb

fig=plt.figure(figsize=(12,12))
plt.suptitle('Same images as in ii) but showing the\nHED-pace and H-channel of each Iamge', fontsize='24')

for j in range(0,2):
    for i in range (j*2,3+j*2):
        m = np.argmax(Y0[:,i]) #comapre ii)
        ax = fig.add_subplot(4, 3, i+1+j*4)
        I_hed = rgb2hed(images[m]); im = ax.imshow(images[m]); fig.colorbar(im)  #converting from RGB space to HED-space
        ax.set_title('Original image No. %i' % (m))
        ax = fig.add_subplot(4, 3, i+4+j*4)
        null = np.zeros_like(I_hed[:, :, 0])
        toll = np.stack((I_hed[:, :, 0], null, null), axis=-1)
        #toll = np.stack((null, I_hed[:, :, 1], null), axis=-1) #uncomment for E-Channel
        #toll = np.stack((null, null, I_hed[:, :, 2]), axis=-1)   #uncomment for D-Channel      
        I_h = I_hed[:,:,0]; im = ax.imshow(hed2rgb(toll)); fig.colorbar(im)   #extracting H-Channel of HED-Space
        ax.set_title('H-channel of Image No. %i' % (m))
fig.tight_layout() #fixing layout
#%% v
I_hed = rgb2hed(X0) #converting every image of Fold=0 (referng to Fold-1)in to the HED-Space 
I_h = I_hed[:,:,:,0] #extracting the H-Channel. In this case the last entry determines a certain Channel 
immeans = np.mean(I_h,axis= (1,2)) #calculating the mean of each Image. In the end we have 827 means.

fig=plt.figure(figsize=(16,9)) #same procedure as usual
plt.suptitle('H-channel for each image vs. its cell count of a certain type for images in Fold-1', fontsize='24')
for i in range(0,6):
    ax = fig.add_subplot(2, 3, i+1)
    C = Y0[:,i] #Cell Count of each typ
    plt.scatter(C, immeans,linewidths=0.01, alpha=0.34) #scatter plot, cell count against mean of each picture. In the end we have 827 dots, refering each to one image
    ax.set_title('Comparison with T%i-Cells' % (i+1))
    ax.set_ylabel('average of the H-channel for each image')
    ax.set_xlabel('Cell count of T%i-Cells' %(i+1))
fig.tight_layout() 
#%% vi
# =============================================================================
# For this problem, you can use either error measures, such as R^2-Score and the MSE,
# or the correlation coefficient between the true and predicted outputs for another example in a test set.
# However, I would exclude the error measures as a performance metric and trust on the correlation coefficient.
# Because of the reason that a Regression would be extremely imprecise based on our observations in v) and
# therefore we can expect a very high MSE and a very low R^2-Score. Whereas using the correlation coefficient
# only compares our regression with another set.
# =============================================================================


