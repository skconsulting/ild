# coding: utf-8
import os
from scipy import misc
import numpy as np
#from matplotlib import pyplot as plt
#from keras.utils.data_utils import get_file
#from six.moves import cPickle
#import sys
#from keras.utils import np_utils 
import cPickle as pickle
from sklearn.cross_validation import train_test_split
import random

HUG='HUG'
#define the working directory
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)
patch_dir=os.path.join(dirHUG,'patches_norm_32_trial')
pickle_dir=os.path.join(cwdtop,'pickle_ds32_essai')
if not os.path.exists(pickle_dir):
   os.mkdir(pickle_dir)  


#define a list of used labels
usedclassif = [
    'back_ground',
#    'consolidation',
    'fibrosis',
    'ground_glass'
#    'healthy',
#    'micronodules',
#    'reticulation',
#    'air_trapping',
#    'cysts',
#    'bronchiectasis'
    ]
#print (usedclassif)
classif ={
'back_ground':0,
#'consolidation':1,
'fibrosis':1,
'ground_glass':2,
'healthy':4,
'micronodules':5,
'reticulation':6,
'air_trapping':7,
'cysts':8,
'bronchiectasis':9,

 'bronchial_wall_thickening':10,
 'early_fibrosis':11,
 'emphysema':12,
 'increased_attenuation':13,
 'macronodules':14,
 'pcp':15,
 'peripheral_micronodules':16,
 'tuberculosis':17
}
#augmentation factor
augf=1
#define a dictionary with labels
#classif={}
#i=0
for f in usedclassif:
    print (f, classif[f])
    
#define another dictionaries to calculate the number of label
classNumberInit={}
for f in usedclassif:
    classNumberInit[f]=0

classNumberNew={}
for f in usedclassif:
    classNumberNew[f]=0

#another to define the coeff to apply
classConso={}

print ('classification used:',classif)

# list all directories under patch directory. They are representing the categories

category_list=os.walk( patch_dir).next()[1]
# print what we have as categories
print ('all classes:',category_list)


# go through all categories to calculate the number of patches per class
# 
for category in usedclassif:
    category_dir = os.path.join(patch_dir, category)
    print  'the path into the categories is: ', category_dir
    sub_categories_dir_list = (os.listdir(category_dir))
    print 'the sub categories are : ', sub_categories_dir_list
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
        #print  'the path into the sub categories is: '
        #print subCategory_dir
        image_files = (os.listdir(subCategory_dir))
        for file in image_files:
            if file.find('.bmp') > 0:
                classNumberInit[category]=classNumberInit[category]+augf


total=0
print('number of patches init')
for f in usedclassif:
    print (f,classNumberInit[f])
    total=total+classNumberInit[f]
print('total:',total)


#define coeff
maxl=0
for f in usedclassif:
   if classNumberInit[f]>maxl and f !='back_ground':
      maxl=classNumberInit[f]
print ('max number of patches in : ',maxl)

for f in usedclassif:
  classConso[f]=float(maxl)/classNumberInit[f]
for f in usedclassif:
    print (f,classConso[f])



def listcl(lc):
    dlist = []
    print('list patches from class : ',lc)
#    while classNumberNew[lc]<m:        
    category_dir = os.path.join(patch_dir, lc)
#            print  ('the path into the categories is: ', lc)
    sub_categories_dir_list = (os.listdir(category_dir))
    print ('the sub categories are : ', sub_categories_dir_list)
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
        #print  'the path into the sub categories is: '
        #print subCategory_dir
        image_files = (os.listdir(subCategory_dir))

        for filei in image_files:

            if filei.find('.bmp') > 0:
            # load the .bmp file into array
                classNumberNew[lc]=classNumberNew[lc]+augf
                image = misc.imread(os.path.join(subCategory_dir,filei), flatten= 0)
                #print image                  
                # 1 append the array to the dataset list                        
                dlist.append(image)
##                        #2 created rotated copies of images
#                image90 = np.rot90(image)                        
#                dlist.append(image90)
##                        #3 created rotated copies of images                        
#                image180 = np.rot90(image90)
#                dlist.append(image180)
##                        #4 created rotated copies of images                                          
#                image270 = np.rot90(image180)                      
#                dlist.append(image270)   
                
                #flip fimage left-right
#                        imagefliplr=np.fliplr(image)
                #2
#                        dlist.append(imagefliplr)
#                        imageflipud=np.flipud(image)
                #3                        
#                        dlist.append(imageflipud)
    return dlist


# list for the merged pixel data

# list of the label data
label_list = []
dataset_list =[]
for f in usedclassif:
     print('work on :',f)
    #fill list with patches
     resul = listcl(f)
#     resul=equal(maxl,dlf,f)
     i=0
     while i <  classNumberNew[f]:
        dataset_list.append(resul[i])
        label_list.append(classif[f])
        i+=1
 


for f in usedclassif:
    print ('init',f,classNumberInit[f])
    print ('after',f,classNumberNew[f])



print (len(dataset_list),len(label_list))


# transform dataset list into numpy array                   
X = np.array(dataset_list)
#this is already in greyscale
# use only one of the 3 color channels as greyscale info
#X = dataset[:,:, :,1]

print 'dataset shape is now: ', X.shape
#print('X22 as example:', X[22])
# 
y = np.array(label_list)
# sampling item 22
#print ('y22 as example:',y[22])

print ('Xshape : ',X.shape)
print ('yshape : ',y.shape)


X_train, X_intermediate, y_train, y_intermediate = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_intermediate, y_intermediate, test_size=0.5, random_state=42)
print ('Xtrain :',X_train.shape)
print ('Xval : ',X_val.shape)
print ('Xtest : ',X_test.shape)
print ('ytrain : ',y_train.shape)
print ('ytest : ',y_test.shape)
print ('yval : ',y_val.shape)
    

# save the dataset and label set into serial formatted pkl 

pickle.dump(X_train, open( os.path.join(pickle_dir,"X_train.pkl"), "wb" ))
pickle.dump(X_test, open( os.path.join(pickle_dir,"X_test.pkl"), "wb" ))
pickle.dump(X_val, open(os.path.join(pickle_dir,"X_val.pkl"), "wb" ))
pickle.dump(y_train, open( os.path.join(pickle_dir,"y_train.pkl"), "wb" ))
pickle.dump(y_test, open( os.path.join(pickle_dir,"y_test.pkl"), "wb" ))
pickle.dump(y_val, open( os.path.join(pickle_dir,"y_val.pkl"), "wb" ))


# testing if pickls was working fine
recuperated_X_train = pickle.load( open( os.path.join(pickle_dir,"X_train.pkl"), "rb" ) )


#print ('recuparated 22 as example:',recuperated_X_train[22])


