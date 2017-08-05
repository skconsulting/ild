# coding: utf-8
'''create dataset from patches 
support Hu

version 1.0
S. Kritter
5 august 2017
 '''
 
 
from param_pix_t import classif,usedclassif
from param_pix_t import thrpatch
from param_pix_t import remove_folder,norm

from param_pix_t import picklepath,perrorfile


import datetime
import os
import cv2
#import dircache
import sys
import shutil
from scipy import misc
from scipy.fftpack import fft,dct,idct
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import cPickle as pickle
#from sklearn.cross_validation import train_test_split
import random
import math
from math import *
print sklearn.__version__

#####################################################################
#define the working directory
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'

toppatch= 'TOPPATCH'
extendir='set1'

pickel_dirsource_root='pickle'
pickel_dirsource_e='train_set' #path for data fort training
pickel_dirsourcenum='1' #extensioon for path for data for training
extendir2=''

augf=3#augmentation factor

########################################################################
######################  end ############################################
########################################################################

#sepextend2='ROI'
if len (extendir2)>0:
    extendir2='_'+extendir2

pickel_dirsource=pickel_dirsource_root+'_'+pickel_dirsource_e+'_'+pickel_dirsourcenum+extendir2

patch_dir=os.path.join(topdir,pickel_dirsource)

print ('classification used:')
for f in usedclassif:
    print (f, classif[f])

print 'path to write data for training',patch_dir

#define the name of directory for patches
patchesdirnametop = 'th'+str(round(thrpatch,1))+'_'+toppatch+'_'+extendir

hugeClass=['healthy']
#hugeClass=['']


#input patch directory
patch_dirsource=os.path.join(topdir,patchesdirnametop)
patch_dirsource=os.path.join(patch_dirsource,patch_dirsource)
patch_dirsource=os.path.join(patch_dirsource,picklepath)
print 'path for pickle inputs',patch_dirsource
if not os.path.isdir(patch_dirsource):
    print 'directory ',patch_dirsource,' does not exist'
    sys.exit()
print '--------------------------------'
###############################################################
remove_folder(patch_dir)
os.mkdir(patch_dir)
eferror=os.path.join(patch_dir,perrorfile)
errorfile = open(eferror, 'w')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started ' +toppatch+' '+extendir+' at :'+todayn)
errorfile.write('numbe of loops :'+str(augf)+'\n')
errorfile.write('--------------------\n')



#define a dictionary with labels
#classif={}
#i=0

print '----------'
#define another dictionaries to calculate the number of label
patclass={}
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
    patclass[f]=[]

# list all directories under patch directory. They are representing the categories

category_list=os.walk( patch_dirsource).next()[1]


# print what we have as categories
print ('all classes in database:',category_list)
print '----------'

usedclassifFinal=[f for f in usedclassif if f in category_list]

# print what we have as categories and in used one
print ('all actual classes:',usedclassifFinal)

numclass= len (usedclassifFinal)
print 'number of classes:', numclass
print '----------'

# go through all categories to calculate the number of patches per class
#
for category in usedclassifFinal:
    category_dir = os.path.join(patch_dirsource, category)
#    print  'the path into the categories is: ', category_dir
    sub_categories_dir_list = (os.listdir(category_dir))
    #print 'the sub categories are : ', sub_categories_dir_list
    for subCategory in sub_categories_dir_list:
        subCategory_dir = os.path.join(category_dir, subCategory)
#        print  'the path into the sub categories is: ',subCategory_dir
        #print subCategory_dir
        image_files = [name for name in os.listdir(subCategory_dir) if name.find('.pkl') > 0 ]
        for filei in image_files:
                patclass[category]=patclass[category]+pickle.load(open(os.path.join(subCategory_dir,filei),'rb'))
        classNumberInit[category]=len(patclass[category])



total=0
print('number of patches init')
errorfile.write('number of patches init \n')
for f in usedclassifFinal:
    print ('class:',f,classNumberInit[f])
    errorfile.write('class ' +f+' '+str(classNumberInit[f])+'\n')
    total=total+classNumberInit[f]
print('total:',total)
errorfile.write('total ' +str(total)+'\n')
errorfile.write('--------------------\n')
print '----------'

#define coeff max
maxl=0
lif = [name for name in usedclassifFinal if name  not in hugeClass]
for f in lif:
   if classNumberInit[f]>maxl and f not in hugeClass:
#      print 'fmax', f
      maxl=classNumberInit[f]
      patmax=f
print ('max number of patches : '+str(maxl)+' for: '+ patmax)
errorfile.write ('max number of patches : '+str(maxl)+ ' for:' +patmax+'\n')
errorfile.write('--------------------\n')
for i in hugeClass:
    print ('number of patches : ',classNumberInit[i],' for: ', i)
print '----------'
#define coeff min
minl=100000000

for f in usedclassifFinal:
   if classNumberInit[f]<minl:
      minl=classNumberInit[f]
      patmin=f
print ('min number of patches : ',str(minl), ' for:' ,patmin)
errorfile.write ('min number of patches  : '+str(minl)+ ' for:' +patmin+'\n')
print '----------'
errorfile.write('--------------------\n')
print ' maximum number of patches ratio'
classConso={}

for f in usedclassifFinal:
    classConso[f]=float(maxl)/max(1,classNumberInit[f])
#for f in usedclassifFinal:
    print (f,classif[f],' {0:.2f}'.format (classConso[f]))
print '----------'

def genedata(category):
    feature=[]
    label=[]
    lc=classif[category]
    for p in patclass[category]:
            feature.append(norm(p))
            label.append(lc)
    return feature,label


def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image)
    elif tt==2:
    #2 180 deg
        imout = np.rot90(np.rot90(image))
    elif tt==3:
    #3 270 deg
        imout = np.rot90(np.rot90(np.rot90(image)))
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.rot90(np.fliplr(image)))
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.rot90(np.rot90(np.fliplr(image))))
    elif tt==8:
    # 8 flip fimage up-down
        imout = imout=np.flipud(image)
    elif tt==9:
    #9 flip fimage up-down +rot90
        imout = np.rot90(np.flipud(image))
    elif tt==10:
    #10 flip fimage up-down +rot180
        imout = np.rot90(np.rot90(np.flipud(image)))
    elif tt==11:
    #11 flip fimage up-down +rot270
        imout = np.rot90(np.rot90(np.rot90(np.flipud(image))))

    return imout


def genf(features_train,labels_train):
    feature=[]
    label=[]
    indexpatc={}
    for j in usedclassifFinal:
        indexpatc[j]=0
    for aug in range(augf):
            for numgen in range(maxl*numclass):
                    pat =usedclassifFinal[numgen%numclass]
                    numberscan=classNumberNewTr[pat]
                    indexpatc[pat] =  indexpatc[pat]%numberscan
                    indexpat=indexpatc[pat]
                    indexpatc[pat]=indexpatc[pat]+1
           
                    indexaug = random.randint(0, 11)
                    scan=geneaug(features_train[pat][indexpat],indexaug)
                    mask=classif[pat]
                    feature.append(scan)
                    label.append(mask)
    return feature,label

                

# main program
mindct=0
maxdct=0
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

features_train={}
features_test={}
labels_train={}
labels_test={}

print ('---')
for f in usedclassifFinal:
    print 'split data',f
    features_train[f], features_test[f], labels_train[f], labels_test[f] = train_test_split(feature_d[f],label_d[f],test_size=0.1, random_state=42)

print ('---')
errorfile.write('Number of data and label per pattern\n')
for f in usedclassifFinal:
    classNumberNewTr[f]=len(features_train[f])
    print 'train',f,len(features_train[f]),len(labels_train[f])
    errorfile.write('train '+f+' '+str(len(features_train[f]))+' '+str(len(labels_train[f]))+'\n')   
    print 'test',f,len(features_test[f]),len(labels_test[f])
    errorfile.write('test '+f+' '+str(len(features_test[f]))+' '+str(len(labels_test[f]))+'\n')
    print ('----')
    errorfile.write('--------------------\n')
errorfile.write('--------------------\n')


features_aug_train=[]
labels_aug_train=[]

features_aug_train,labels_aug_train= genf(features_train,labels_train)

print 'training after augmentation ',len(features_aug_train),len(labels_aug_train)


features_train_final=[]
features_test_final=[]
labels_train_final=[]
labels_test_final=[]

features_train_final=features_aug_train
labels_train_final=labels_aug_train

for f in usedclassifFinal:
    features_test_final=features_test_final+features_test[f]

    labels_test_final=labels_test_final+labels_test[f]


print '---------------------------'
print ('training set:',len(features_train_final),len(labels_train_final))
errorfile.write('training set: '+' '+str(len(features_train_final))+' '+str(len(labels_train_final))+'\n')   
print ('test set:',len(features_test_final),len(labels_test_final))
errorfile.write('test set: '+' '+str(len(features_test_final))+' '+str(len(labels_test_final))+'\n')   
errorfile.write('--------------------\n')


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
errorfile.write('Xtrain : '+' '+str(X_train.shape)+'\n')
errorfile.write('Xtest : '+' '+str(X_test.shape)+'\n') 
errorfile.write('ytrain : '+' '+str(y_train.shape)+'\n') 
errorfile.write('ytest : '+' '+str(y_test.shape)+'\n')   
errorfile.write('--------------------\n')
errorfile.close()

pickle.dump(X_train, open( os.path.join(patch_dir,"X_train.pkl"), "wb" ),protocol=-1)
pickle.dump(X_test, open( os.path.join(patch_dir,"X_val.pkl"), "wb" ),protocol=-1)

pickle.dump(y_train, open( os.path.join(patch_dir,"y_train.pkl"), "wb" ),protocol=-1)
pickle.dump(y_test, open( os.path.join(patch_dir,"y_val.pkl"), "wb" ),protocol=-1)



recuperated_X_train = pickle.load( open( os.path.join(patch_dir,"X_train.pkl"), "rb" ) )
min_val=np.min(recuperated_X_train)
max_val=np.max(recuperated_X_train)
print 'recuperated_X_train', min_val, max_val
