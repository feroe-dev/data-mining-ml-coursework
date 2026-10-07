# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:51:03 2021

@author: Felipe
"""
import numpy as np
data = np.load("cell-data.npz")
images = data["images"]
counts = data["counts"]
folds = data["folds"]
X0 = images[folds == 0]
Y0 = counts[folds == 0]
#%%

#load saved variables
import dill #relaoding the old variables of the kernel
dill.load_session('PickleAssignment2.db')

#%% i a-c
from skimage.color import rgb2hed
import skimage.measure 

no = 809 #number of image
im_rgb = images[no] #choosing image no.809 as our given image
   

#H
im_hed = rgb2hed(im_rgb) #converting image 809 in to the HED-Space
im_h = im_hed[:,:,0] #"h"-channel of image 809
im_h_mean = np.mean(im_h,axis= (0,1)) #mean of H-channel of image 809
im_h_var = np.var(im_h,axis= (0,1)) #mean of H-channel of image 809
im_h_en = skimage.measure.shannon_entropy(im_h) #entropy of H-Channel of iamge 809

print('| Channel\t\t| a. Average\t| b. Variance\t| c. Entropy\t|')
print('----------------------------------------------------------------')
print('|\t%s\t\t\t| %1.3f\t\t| %1.3f\t\t\t| %1.3f\t\t|' % ('H', im_h_mean, im_h_var, im_h_en))
"""
# There are 2 ways to calculate the entropy.
# 1. Way, directly through the module scipy.stats:
from scipy import stats
im_h_en = stats.entropy(im_h,axis= (0,1), base=2) #entropy of H-Channel of image 809
# 2. Way, through a count functions - skimage.measure.shannon_entropy uses this method
# - somehow they deliver different results. Entropy(1.Way) = 15.99 whereas Entropy(2.Ways) = 15.164
# through this task I use the second way to obtain comparable results
A, counts = np.unique(im_h, return_counts=True)
ent = stats.entropy(counts, base=2)
"""
channels_rgb = ['Blue','Green','Red\t']
im_rgb_res = np.zeros((3,3)) #saving the average, variance and the entropy of the red, green and blue channels

for i in range (0,3):
    # for loop runs through each Channel: Where i=0 refers to the blue, i=1 refers to the green and
    # i=2 refers to the red channel - saving the results in im_rgb
    sh = im_rgb[:,:,i]
    im_rgb_res[i,0] = np.mean(sh,axis= (0,1)) #mean of H-channel of image 809
    im_rgb_res[i,1] = np.var(sh,axis= (0,1)) #mean of H-channel of image 809
    im_rgb_res[i,2] = skimage.measure.shannon_entropy(sh) #entropy of H-Channel of image 809
    print('----------------------------------------------------------------')
    print('|\t%s\t\t| %1.3f\t\t| %1.3f\t\t| %1.3f\t\t\t|' % (channels_rgb[i], im_rgb_res[i,0], im_rgb_res[i,1], im_rgb_res[i,2]))


#Visulitation of each channel 
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(16,9))
ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)
imb, img, imr = images[809].copy(), images[809].copy(), images[809].copy()
imb[:, :, 0] = 0
imb[:, :, 1] = 0
ax1.imshow(imb)

img[:, :, 0] = 0
img[:, :, 2] = 0
ax2.imshow(img)

imr[:, :, 1] = 0
imr[:, :, 2] = 0
ax3.imshow(imr)

ax1.set_title('Blue-Channel')
ax2.set_title('Green-Channel')
ax3.set_title('Red-Channel')
fig.tight_layout()

#%% i d1
# =============================================================================
# Now we want to convert our images from RGB-Space to HSV-Space (Hue, Saturation and Value)
# (https://www.tech-faq.com/hsv.html & https://en.wikipedia.org/wiki/HSL_and_HSV for further explanations).
# Hue Channel: representing the colors
# Saturation Channel:  indicating the range of grey in the color space
# Value Channel: parameter for brightness of the color and varies with color saturation
#
# But why the HSV-Space? The HSV colour space is easier to use to filter for specific colours
# compared to the RBG colour space. (This link gives an easy understanding of the topic:https://handmap.github.io/hsv-vs-rgb/).
# For example in our case I want to search for specific violet tones in the images which would be a lot
# in the RGB-space space than in HSV-space.
# The Hue-Channel and Saturation channel has less meaning in our case because our
# images have a very similar Saturation and Hue values . The juxtaposition of those channels
# of the HSV-Space underlines my statement due the Saturation- and Hue Channel does not provide great contrasts
# but the Value-Channel does. Here the cells stand out in blue. The low entropy of the value-channel,
# which is at the same level as the Blue-Channel, clarifies this aswell.
# =============================================================================
print('----------------------------------------------------------------')
from skimage.color import rgb2hsv, hsv2rgb

im_hsv =  rgb2hsv(im_rgb)
channels_hsv = ['Hue\t\t','Saturation','Value\t']
im_hsv_res = np.zeros((3,3)) #saving the average, variance and the entropy of the red, green and blue channels

for i in range (0,3):
    # for loop runs through each Channel: Where i=0 refers to the blue, i=1 refers to the green and
    # i=2 refers to the red channel - saving the results in im_rgb
    sh = im_hsv[:,:,i]
    im_hsv_res[i,0] = np.mean(sh,axis= (0,1)) #mean of H-channel of image 809
    im_hsv_res[i,1] = np.var(sh,axis= (0,1)) #mean of H-channel of image 809
    im_hsv_res[i,2] = skimage.measure.shannon_entropy(sh) #entropy of H-Channel of image 809
    print('----------------------------------------------------------------')
    print('|\t%s\t| %1.3f\t\t\t| %1.3f\t\t\t| %1.3f\t\t\t|' % (channels_hsv[i], im_hsv_res[i,0], im_hsv_res[i,1], im_hsv_res[i,2]))

#Visulitation of each channel 
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16,9))
ax1, ax2, ax3 = fig.add_subplot(1, 3, 1), fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)

null = np.zeros_like(im_hsv[:, :, 1])
mx = null + 1

ax1.imshow(hsv2rgb(np.stack((im_hsv[:, :, 0], mx, mx), axis=-1)))
ax2.imshow(hsv2rgb(np.stack((null, im_hsv[:, :, 1], mx), axis=-1)))
ax3.imshow(hsv2rgb(np.stack((null, null, im_hsv[:, :, 2]), axis=-1)))

ax1.set_title("Hue channel")
ax3.set_title("Value channel")
ax2.set_title("Saturation channel")

fig.tight_layout()

from sklearn.decomposition import IncrementalPCA
from skimage.color import rgb2gray

im_grey = rgb2gray(images)
print('----------------------------------------------------------------')
print('|\t%s\t\t| %1.3f\t\t\t| %1.3f\t\t\t| %1.3f\t\t|' % ('Grey', np.mean(im_grey[no],axis= (0,1)), np.var(im_grey[no],axis= (0,1)), skimage.measure.shannon_entropy(im_grey[no])))

ipca = IncrementalPCA()
im_greyfl = np.zeros((len(im_grey),256**2))
for i in range (0,len(im_grey)):
    im_greyfl[i] = im_grey[i,:,:].flatten()
ipca.partial_fit(im_greyfl[1:500,:])



from skimage.feature import greycomatrix, greycoprops
from skimage import data
from skimage import util


# =============================================================================
# Sources:[1] https://www.researchgate.net/publication/312107023_Extraction_of_Texture_Features_using_GLCM_and_Shape_Features_using_Connected_Regions
#         [2] https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_glcm.html
#         [3] https://support.echoview.com/WebHelp/Windows_and_Dialog_Boxes/Dialog_Boxes/Variable_properties_dialog_box/Operator_pages/GLCM_Texture_Features.htm
#         [4] https://prism.ucalgary.ca/bitstream/handle/1880/51900/texture%20tutorial%20v%203_0%20180206.pdf?sequence=11&isAllowed=y
#
# "[...]GLCM (Gray-Level-Co-Occurance Matrix) represents the relation between refernce pixel (i) 
# and neighbour pixel (j) in various orientation. Initially, the value of each elements in GLCM (i,j) is zero.
# The value of each element is updated as per occurance of pixels together. Texture features calculated using 
# GLCM are Contrast, Correlation, Dissimilarity, Energy, Entropy, Homogenity, Mean, Variance, Standard Deviation.[...]" [3]
# Many of the feautes mentioned above say the same thing at their core. It is mainly about the distance,
# difference or deviation between the grey levels. Therefore, as in source [2],
# I see correlation and dissimilarity as interesting features.
#
#
# =============================================================================
im_grey = rgb2gray(images[809])
im_grey = util.img_as_ubyte(im_grey)

PATCH_SIZE = 19

# select some patches from sky areas of the image
cell_locations = [(17, 108), (169, 14), (240, 113), (218, 186), (36, 175), (218,236), (143,123), (28,70), (146,67)]
cell_patches = []
for loc in cell_locations:
    print(loc)
    cell_patches.append(im_grey[loc[1]:loc[1] + PATCH_SIZE,
                             loc[0]:loc[0] + PATCH_SIZE])


#compute some GLCM properties each patch
xs = []
ys = []
for patch in cell_patches:
    glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
    ys.append(greycoprops(glcm, 'correlation')[0, 0])
   
# create the figure
fig = plt.figure()

# display original image with locations of patches
ax = fig.add_subplot(1, 2, 1)
ax.imshow(im_grey, cmap=plt.cm.gray,
          vmin=0, vmax=255)

for i,(x, y) in enumerate(cell_locations):
    ax.plot(x, y, 'x', color='blue')
    ax.annotate('Cell %d' % (i + 1), xy=(x, y), fontsize=8, color='red')
    
ax.set_title('Original Image')
ax.set_xticks([])
ax.set_yticks([])
ax.axis('image')
"""
# create the figure
plt.figure()

# display original image with locations of patches
plt.imshow(im_grey, cmap=plt.cm.gray,
          vmin=0, vmax=255)

for (y, x) in cell_locations:
    plt.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs')
"""    
# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(1, 2, 2)
ax.plot(xs[:len(cell_patches)],  ys[:len(cell_patches)], 'go',
        label='Cell')
ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

for i in range (0,len(cell_patches)):
    ax.scatter(xs[i], ys[i], 50, color ='blue')
    ax.annotate('Cell %d' % (i + 1), xy=(xs[i], ys[i]), fontsize=8)    
fig.tight_layout()
fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.show()

fig = plt.figure(figsize=(9, 9))
cal = 1+round(len(cell_patches)/3+0.5)

# display the image patches
for i, patch in enumerate(cell_patches):
    ax = fig.add_subplot(cal, 3, 4 + i)
    ax.imshow(patch, cmap=plt.cm.gray,
              vmin=0, vmax=255)
    ax.set_title('Cell %d' % (i + 1))


# display the patches and plot
fig.tight_layout()
plt.show()

#%% i d2

memory = np.zeros((2351,4,3)) #the results are getting saved here
for j,im in enumerate(images):    
    memory[j,0,0] = np.mean(rgb2hed(im)[:,:,0],axis= (0,1)) #mean of H-channel of image j
    memory[j,0,1] = np.var(rgb2hed(im)[:,:,0],axis= (0,1)) #mean of H-channel of image j
    memory[j,0,2] = skimage.measure.shannon_entropy(rgb2hed(im)[:,:,0]) #entropy of H-Channel of image j
    for i in range (0,3):
        # cf. 
        # runs through the Red, Blue and Green Channel
        memory[j,i+1,0] = np.mean(im[:,:,i],axis= (0,1)) #mean of channel of image j
        memory[j,i+1,1] = np.var(im[:,:,i],axis= (0,1)) #mean of channel of image j
        memory[j,i+1,2] = skimage.measure.shannon_entropy(im[:,:,i]) #entropy of H-Channel of image j

memT = memory.T
fig=plt.figure(figsize=(16,9))
features = ['Average','Variance','Entropy']
channels = ['H','Red','Green','Blue']
print('|Corr. coefficient\t| H-channel\t| Red-channel\t| Green-channel\t| Blue-channel\t|')
for i,mem in enumerate(memT):
    coef = np.zeros((1,4))
    for j in range (0,4):
        ax = fig.add_subplot(3, 4, j+i*4+1)
        coef[:,j] = np.corrcoef(mem[j],counts[:,0])[0,1]
        #print('The correlation coefficient of the %s of %s-Channel = %1.4f' % (features[i],
                                                                      # channels[j]
                                                                      #,np.corrcoef(mem[j],counts[:,0])[0,1]))
        ax.scatter(counts[:,0], mem[j],linewidths=0.01, alpha=0.34)
        #ax.set_title('%s of %s-Channel vs cell count of T1-cells' % (features[i], channels[j]))
        ax.set_ylabel('%s of %s-Channel' % (features[i], channels[j]))
        ax.set_xlabel('Cell-count  of T1-cells')
    print('-------------------------------------------------------------------------------')
    print('| %s\t\t\t| %1.4f\t| %1.4f\t\t| %1.4f\t\t| %1.4f\t\t|' % (features[i], coef[:,0], coef[:,1],
                                                           coef[:,2], coef[:,3]))
fig.tight_layout()

# =============================================================================
# ### Which features are important?
# You should focus on features with high absolute correlation coefficients like the average here.
# A high absolute correlation coefficient gives us a certain confidence that the two variables are
# linearly dependant. This knowledge could help us to predict the cell count of unseen images.
# A positive/negative correlation coefficient states out that if there is a positive increase in one variable,
# there is also a positive/negative increase in the second variable. In our case, the H-Channel or
# the Red-Channel have a big enough value to claim that there exists a linear dependency between the cell-count
# and the averages of the respective channels. Namely around |0.45|. If I had to choose between those
# two channels I would choose the red-channel due to the halftimes lower entropy.
# The lower the entropy the more information we can expect in a set of values.
# =============================================================================
#%% ii a
memT = memory.T
a
from sklearn import linear_model

ols = linear_model.LinearRegression(fit_intercept=True, copy_X = True) #only FOLD = 0 (First Fold, Training sample)
ols.fit(memT[folds==0], Y0[0])

