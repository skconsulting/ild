# coding: utf-8
'''create dataset from patches using split into 3 bmp databases and
 augmentation only on training, using bg, bg number limited to max and with equalization
 for all in training, validation and test
 defined used pattern set
 '''
import os
import dircache
import sys
import shutil
from scipy import misc
import numpy as np

import cPickle as pickle
#from sklearn.cross_validation import train_test_split
import random

#####################################################################
#define the working directory
HUG='HUG'   

#output pickle dir with dataset   
pickel_dirsource='pickle_ds23'

#input patch directory
patch_dirsource='patches_norm_20_set0'

#output database for database generation
patch_dirSplitsource=   'patches_norm_ref_20_set0'


#define the pattern set
pset=0

###############################################################
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)


pickle_dir=os.path.join(cwdtop,pickel_dirsource)  


patch_dir=os.path.join(dirHUG,patch_dirsource)



#define a list of used labels
if pset ==0:
    usedclassif = [
        'back_ground',
        'consolidation',
        'fibrosis',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping'
    #    'cysts',
    #    'bronchiectasis'
        ]
    classif ={
        'back_ground':0,
        'consolidation':1,
        'fibrosis':2,
        'ground_glass':3,
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
elif pset==1:
    usedclassif = [
        'back_ground',
        'consolidation',
        'ground_glass',
        'healthy'
    #    ,'cysts'
        ]
        
    classif ={
    'back_ground':0,
    'consolidation':1,
    'ground_glass':2,
    'healthy':3
    #,'cysts':4
    }
elif pset==2:
        usedclassif = [
        'back_ground',
        'fibrosis',
        'healthy',
        'micronodules'
        ,'reticulation'
        ]
        
        classif ={
    'back_ground':0,
    'fibrosis':1,
    'healthy':2,
    'micronodules':3,
    'reticulation':4,
    }
elif pset==3:
    usedclassif = [
        'back_ground',
        'healthy',
        'air_trapping',
        ]
    classif ={
        'back_ground':0,
        'healthy':1,
        'air_trapping':2,
        }
else:
        print 'eRROR :', pset, 'not allowed'


#classif ={
#'back_ground':0,
#'consolidation':1,
#'fibrosis':2,
#'ground_glass':3,
#'healthy':4,
#'micronodules':5,
#'reticulation':6,
#'air_trapping':7,
#'cysts':8,
#'bronchiectasis':9,
#
# 'bronchial_wall_thickening':10,
# 'early_fibrosis':11,
# 'emphysema':12,
# 'increased_attenuation':13,
# 'macronodules':14,
# 'pcp':15,
# 'peripheral_micronodules':16,
# 'tuberculosis':17
#}
        
#print (usedclassif)

#augmentation factor
augf=6
#define a dictionary with labels
#classif={}
#i=0
print ('classification used:')
for f in usedclassif:
    print (f, classif[f])

print '----------'
#define another dictionaries to calculate the number of label
classNumberInit={}    
classNumberNewTr={}
classNumberNewV={}
classNumberNewTe={}

for f in usedclassif:
    classNumberInit[f]=0
    classNumberNewTr[f]=0
    classNumberNewV[f]=0
    classNumberNewTe[f]=0

# list all directories under patch directory. They are representing the categories

category_list=os.walk( patch_dir).next()[1]
# print what we have as categories
print ('all classes:',category_list)
print '----------'
# go through all categories to calculate the number of patches per class
# 
for category in usedclassif:
    category_dir = os.path.join(patch_dir, category)
    #print  'the path into the categories is: ', category_dir
    sub_categories_dir_list = (os.listdir(category_dir))
    #print 'the sub categories are : ', sub_categories_dir_list
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
        #print  'the path into the sub categories is: '
        #print subCategory_dir
        image_files = (os.listdir(subCategory_dir))
        for filei in image_files:
            if filei.find('.bmp') > 0:
                classNumberInit[category]=classNumberInit[category]+1


total=0
print('number of patches init')
for f in usedclassif:
    print ('class:',f,classNumberInit[f])
    total=total+classNumberInit[f]
print('total:',total)
print '----------'

#define coeff
maxl=0
for f in usedclassif:
   if classNumberInit[f]>maxl and f !='back_ground':
      maxl=classNumberInit[f]
print ('max number of patches in : ',maxl)
print '----------'
#artificially clamp back-ground to maxl
classNumberInit['back_ground']=maxl
classConso={}
for f in usedclassif:
  classConso[f]=float(maxl)/classNumberInit[f]
for f in usedclassif:
    print (f,' {0:.2f}'.format (classConso[f]))
print '----------'

