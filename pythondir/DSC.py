# coding: utf-8
'''create dataset from patches using bg, bg and healthy number limited to max
 '''
import os
import cv2
#import dircache
import sys
import shutil
from scipy import misc
import numpy as np
from sklearn.cross_validation import train_test_split
import cPickle as pickle
#from sklearn.cross_validation import train_test_split
import random
import math
from math import *

#####################################################################
#define the working directory
HUG='HUG'   
#HUG='CHU'
subDir='TOPPATCH'

#extent='3d162'
extent='16_set1_13b2'
#extent='0'
#wbg = True # use back-ground or not
#hugeClass=['healthy','back_ground']
hugeClass=['healthy']
imageDepth=8191 
upSampling=1 # define thershold for up sampling in term of ration compared to max class
subDirc=subDir+'_'+extent
pickel_dirsource='pickle_ds69'
#pickel_dirsource='pickle_dsessai'
name_dir_patch='patches'
#output pickle dir with dataset   
augf=12
#typei='bmp' #can be jpg
typei='png' #can be jpg

#define the pattern set
pset=2 # 1 when with new patters superimposed, 2 idem but bg merged with healthy
#stdMean=True #use mean and normalization for each patch
#maxuse=True # True means that all class are upscalled to have same number of 
 #elements, False means that all classes are aligned with the minimum number
#useWeight=True #True means that no upscaling are done, but classes will 
#be weighted ( back_ground and healthy clamped to max) 

subdirc=subDir+'_'+extent
cwd=os.getcwd()
(cwdtop,tail)=os.path.split(cwd)
dirHUG=os.path.join(cwdtop,HUG)
dirHUG=os.path.join(dirHUG,subDirc)
print 'dirHUG', dirHUG
#input pickle dir with dataset to merge
#pickel_dirsourceToMerge='pickle_ds22'


#input patch directory
#patch_dirsource=os.path.join(dirHUG,'patches_norm_16_set0_1')
patch_dirsource=os.path.join(dirHUG,name_dir_patch)
print 'patch_dirsource',patch_dirsource
if not os.path.isdir(patch_dirsource):
    print 'directory ',patch_dirsource,' does not exist'
    sys.exit()

###############################################################

pickle_dir=os.path.join(cwdtop,pickel_dirsource) 

#pickle_dirToMerge=os.path.join(cwdtop,pickel_dirsourceToMerge)  

patch_dir=os.path.join(dirHUG,patch_dirsource)
print patch_dir

def remove_folder(path):
    """to remove folder"""
    # check if folder exists
    if os.path.exists(path):
         # remove if exists
         shutil.rmtree(path)

remove_folder(pickle_dir)
os.mkdir(pickle_dir)

if pset ==0:
        usedclassif = [
            'back_ground',
            'consolidation',
            'HC',
            'ground_glass',
            'healthy',
            'micronodules',
            'reticulation',
            'air_trapping',
            'cysts',
            'bronchiectasis'
            ]
            
        classif ={
            'back_ground':0,
            'consolidation':1,
            'HC':2,
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
        print 'with BG and merged patterns'
        usedclassif = [
        'back_ground',
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
#        'emphysema',
        'GGpret'
        ]
            
        classif ={
        'back_ground':0,
        'consolidation':1,
        'HC':2,
        'ground_glass':3,
        'healthy':4,
        'micronodules':5,
        'reticulation':6,
        'air_trapping':7,
        'cysts':8,
        'bronchiectasis':9,
#        'emphysema':10,
        'GGpret':10
        }

elif pset==2:
        print 'with BG merged in healthy and merged patterns'

        usedclassif = [
        'consolidation',
        'HC',
        'ground_glass',
        'healthy',
        'micronodules',
        'reticulation',
        'air_trapping',
        'cysts',
        'bronchiectasis',
#        'emphysema',
        'GGpret'
        ]
            
        classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'air_trapping':6,
        'cysts':7,
        'bronchiectasis':8,
#        'emphysema':10,
        'GGpret':9
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

def normi(tabi,n):
     """ normalise patches"""
    
     max_val=float(np.max(tabi))
     min_val=float(np.min(tabi))
    
     mm=max_val-min_val
     mm=max(mm,1.0)
#     print 'tabi1',min_val, max_val,imageDepth/float(max_val)
     tabi2=(tabi-min_val)*(n/mm)
     tabi2=tabi2.astype('uint16')

     return tabi2

class_weights={}

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
actualClasses=[]
classAugm={}

for f in usedclassif:
    classNumberInit[f]=0
    classNumberNewTr[f]=0
    classNumberNewV[f]=0
    classNumberNewTe[f]=0
    classAugm[f]=1

# list all directories under patch directory. They are representing the categories

category_list=os.walk( patch_dir).next()[1]


# print what we have as categories
print ('all actual classes:',category_list)
print '----------'

usedclassifFinal=[f for f in usedclassif if f in category_list]

# print what we have as categories and in used one
print ('all actual classes:',usedclassifFinal)
print '----------'
# go through all categories to calculate the number of patches per class
# 
for category in usedclassifFinal:
    category_dir = os.path.join(patch_dir, category)
#    print  'the path into the categories is: ', category_dir
    sub_categories_dir_list = (os.listdir(category_dir))
    #print 'the sub categories are : ', sub_categories_dir_list
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
#        print  'the path into the sub categories is: ',subCategory_dir
        #print subCategory_dir
        image_files = [name for name in os.listdir(subCategory_dir) if name.find('.'+typei) > 0 ]              
        for filei in image_files:               
                classNumberInit[category]=classNumberInit[category]+1


total=0
print('number of patches init')
for f in usedclassifFinal:
    print ('class:',f,classNumberInit[f])
    total=total+classNumberInit[f]
print('total:',total)
print '----------'
#ooo
#define coeff max
maxl=0
for f in usedclassifFinal:
   if classNumberInit[f]>maxl and f not in hugeClass:
#      print 'fmax', f
      maxl=classNumberInit[f]
print ('max number of patches in : ',maxl)
print '----------'
#define coeff min
minl=1000000
for f in usedclassifFinal:
   if classNumberInit[f]<minl:
      minl=classNumberInit[f]
print ('min number of patches in : ',minl)
print '----------'

print ' maximum number of patches ratio'  
classConso={}
for f in usedclassifFinal:
    classConso[f]=float(maxl)/max(1,classNumberInit[f])
#for f in usedclassifFinal:
    print (f,classif[f],' {0:.2f}'.format (classConso[f]))
print '----------'
print ' upsampling if ratio <' ,upSampling
print 'usedclassifFinal', usedclassifFinal
for f in usedclassifFinal:
#    print f, classConso[f], upSampling,maxl, float(maxl)/upSampling/classNumberInit[f]
    if int(classConso[f]) >upSampling:
#         print f
         classAugm[f]=float(maxl)/upSampling/classNumberInit[f]
    print (f,classif[f],' {0:.2f}'.format (classAugm[f]))
print '----------'


def genedata(category):
    feature=[]
    label=[]
    category_dir = os.path.join(patch_dir, category)
#    print  'the path into the categories is: ', category_dir
    sub_categories_dir_list = (os.listdir(category_dir))
    #print 'the sub categories are : ', sub_categories_dir_list
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
#        print  'the path into the sub categories is: ',subCategory_dir
        #print subCategory_dir
        image_files = [name for name in os.listdir(subCategory_dir) if name.find('.'+typei) > 0 ]              
        for filei in image_files:                      
            fileic=os.path.join(subCategory_dir,filei)
            image=cv2.imread(fileic,-1)                
            feature.append(image)
            label.append(classif[category])
    return feature,label

def geneaug(f,fea,lab):
        feature=[]
        label=[]     
        l=len(fea)
        for i in range (0,l):                                        
            # 1 append the array to the dataset list  
            labi=lab[i]
            image=fea[i]
            feature.append(image)
            label.append(labi)
            
#                        #2 created rotated copies of images
            image90 = np.rot90(image)  
            feature.append(image90)
            label.append(labi)
            
#                        #3 created rotated copies of images                        
            image180 = np.rot90(image90)
            feature.append(image180)
            label.append(labi)
      
#                        #4 created rotated copies of images                                          
            image270 = np.rot90(image180)                      
            feature.append(image270) 
            label.append(labi)
            
            #5 flip fimage left-right
            imagefliplr=np.fliplr(image)   
            feature.append(imagefliplr) 
            label.append(labi)
            
            #6 flip fimage left-right +rot 90
            image90 = np.rot90(imagefliplr)  
            feature.append(image90)
            label.append(labi)
            
#                        #7 flip fimage left-right +rot 180                   
            image180 = np.rot90(image90)
            feature.append(image180)
            label.append(labi)
      
#                        #8 flip fimage left-right +rot 270                                          
            image270 = np.rot90(image180)                      
            feature.append(image270)  
            label.append(labi)                                                             
                                        
            #9 flip fimage up-down
            imageflipud=np.flipud(image)                                   
            feature.append(imageflipud) 
            label.append(labi)
            
             #10 flip fimage up-down +rot90                               
            image90 = np.rot90(imageflipud)  
            feature.append(image90)
            label.append(labi)
            
#                         #11 flip fimage up-down +rot180                          
            image180 = np.rot90(image90)
            feature.append(image180)
            label.append(labi)
      
#                        #12 flip fimage up-down +rot270                                           
            image270 = np.rot90(image180)                      
            feature.append(image270) 
            label.append(labi)
        return feature,label
                                               

def hugedata(f,fea,lab,maxl):
    feature=[]
    label=[]  
    if maxl<=len(fea):
        feature = random.sample(fea, maxl)
        label=lab[0:maxl]
#        label=random.sample(lab, maxl)
    else:
        feature = fea
        label=lab
    return feature,label

def rn(x,n):
    """ Racine n système """
    if x == 0:
        return 0
    return math.exp(math.log(x) / n)


def updscaledata(fea,lab,ups):
    feature=[]
    label=[]  
    
    for i in range (0,len(fea)):
        img=fea[i]
        imgd=img
        imgm=img
        feature.append(img)
        label.append(lab[i])
#        print img.min(),img.max(),int(ups)
#        mini=img.min()
        maxi=img.max()
        maxscale= imageDepth/maxi
#        print mini,maxi,maxscale, int(ups)
        if int(ups)>1:
#            print int(ups)
            tm= rn(maxscale,int(ups)-1)/2+0.5
#            td= rn(mini,int(ups)-1)
        else:
            tm=1
#            td=1
#        print t
        for n in range (0, int(ups)-1):
            if n%2 ==0:
                imgm=imgm*tm
                imgm=imgm.astype('uint16')
#                print imgm.min(),imgm.max(),tm,td,int(ups)
                feature.append(imgm)
                label.append(lab[i])
            else:
                imgd=imgd/tm
                imgd=imgd.astype('uint16')
#                print imgd.min(),imgd.max(),tm,td,int(ups)
                feature.append(imgd)
                label.append(lab[i])
#        ooo
    return feature,label

def updscaledataSimple(fea,lab,ups):
    feature=[]
    label=[]      
    for i in range (0,len(fea)):
        img=fea[i]
        for n in range (0, int(ups)):
                feature.append(img)
                label.append(lab[i])
    return feature,label

# main program 
feature_d ={}
label_d={}
dataset_listTe ={}
print ('---')
for f in usedclassifFinal:  
     print('genedata from images work on :',f)
     feature_d[f],label_d[f]=genedata(f)

print ('---')
for f in usedclassifFinal:   
    print 'initial ',f,len(feature_d[f]),len(label_d[f])

for f in hugeClass:
     print('select random for huge data work on :',f, 'maximum :',maxl)
     feature_d[f],label_d[f]=hugedata(f,feature_d[f],label_d[f],maxl)

print ('---')
for f in usedclassifFinal:   
    print 'after maxl correction ',f,len(feature_d[f]),len(label_d[f])

features_train={}
features_test={}
labels_train={}
labels_test={}
print ('---')
for f in usedclassifFinal:
    print 'split data',f
    features_train[f], features_test[f], labels_train[f], labels_test[f] = train_test_split(feature_d[f],label_d[f],test_size=0.2, random_state=42)

print ('---')
for f in usedclassifFinal:
    print 'train',f,len(features_train[f]),len(labels_train[f])
    print 'test',f,len(features_test[f]),len(labels_test[f])
    print ('----')

features_aug_train={}
labels_aug_train={} 

usemhuege=[name for name in usedclassifFinal if name not in hugeClass]
for f in usemhuege:
     print('upscale training data on :',f)
     features_train[f],labels_train[f]=updscaledata(features_train[f],labels_train[f],classConso[f])
     print('upscale test data on :',f)
     features_test[f],labels_test[f]=updscaledataSimple(features_test[f],labels_test[f],classConso[f])

print ('---')
for f in usedclassifFinal:   
    print 'training after upscale ',f,len(features_train[f]),len(labels_train[f])
    print 'test after upscale ',f,len(features_test[f]),len(labels_test[f])

for f in usedclassifFinal:
    print('augmentated data on :',f)
    features_aug_train[f],labels_aug_train[f]=geneaug(f,features_train[f],labels_train[f])
    
for f in usedclassifFinal:
    print 'train augmented',f,len(features_aug_train[f]),len(labels_aug_train[f])
    
features_train_final=[]
features_test_final=[]
labels_train_final=[]
labels_test_final=[]

for f in usedclassifFinal:
    features_train_final=features_train_final+features_aug_train[f]
    features_test_final=features_test_final+features_test[f]
    labels_train_final=labels_train_final+labels_aug_train[f]
    labels_test_final=labels_test_final+labels_test[f]


print '---------------------------'
print ('training set:',len(features_train_final),len(labels_train_final))
print ('test set:',len(features_test_final),len(labels_test_final))


# transform dataset list into numpy array                   
X_train = np.array(features_train_final)
y_train = np.array(labels_train_final)

X_test = np.array(features_test_final)
y_test = np.array(labels_test_final)
 
print '-----------FINAL----------------'
print ('Xtrain :',X_train.shape)
print ('Xtest : ',X_test.shape)
print ('ytrain : ',y_train.shape)
print ('ytest : ',y_test.shape)

#    
for f in usedclassifFinal:
#    print f,classNumberNewTr[f]
#    if wbg:
        class_weights[classif[f]]=round(float(len(features_aug_train['healthy']))/len(features_aug_train[f]),3)
#    else:
#        class_weights[classif[f]]=round(float(classNumberNewTr['healthy'])/classNumberNewTr[f],3)
print '---------------'
#if wbg: 
#class_weights[classif['back_ground']]=0.1
print 'weights'
print class_weights

pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ))
pickle.dump(X_train, open( os.path.join(pickle_dir,"X_train.pkl"), "wb" ))
pickle.dump(X_test, open( os.path.join(pickle_dir,"X_val.pkl"), "wb" ))

pickle.dump(y_train, open( os.path.join(pickle_dir,"y_train.pkl"), "wb" ))
pickle.dump(y_test, open( os.path.join(pickle_dir,"y_val.pkl"), "wb" ))



recuperated_X_train = pickle.load( open( os.path.join(pickle_dir,"X_train.pkl"), "rb" ) )
#min_val=np.min(recuperated_X_train)
#max_val=np.max(recuperated_X_train)
#print 'recuperated_X_train', min_val, max_val

recuperated_class_weights = pickle.load( open(os.path.join(pickle_dir,"class_weights.pkl"), "rb" ) )
print 'recuparated weights'
print recuperated_class_weights