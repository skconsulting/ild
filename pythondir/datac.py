# coding: utf-8
'''create dataset from patches 
support Hu
includes back_ground
version 1.0
second step
S. Kritter
18 august 2017
 '''
 
from param_pix_t import classif,usedclassif
from param_pix_t import thrpatch,setdata
from param_pix_t import remove_folder,norm
from param_pix_t import picklepath,perrorfile

import datetime
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
import cPickle as pickle
import random


#####################################################################
#define the working directory for input patches
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
toppatch= 'TOPPATCH'
extendir='all'
extendir0='5'
#extendir0='essai'

extendir1=''

#define the directory to store data
pickel_dirsource_root='pickle'
pickel_dirsource_e='train' #path for data fort training
pickel_dirsourcenum=setdata #extensioon for path for data for training
extendir2='6'
#extendir2='essai'
extendir3=extendir1

augf=5#augmentation factor default 3
test_size=0.1 #split test training percent
########################################################################
######################  end ############################################
########################################################################

#sepextend2='ROI'
if len (extendir3)>0:
    extendir3='_'+extendir3
    
if len (extendir1)>0:
    extendir1='_'+extendir1

pickel_dirsource='th'+str(thrpatch)+'_'+pickel_dirsource_root+'_'+pickel_dirsource_e+'_'+pickel_dirsourcenum+'_'+extendir2+extendir3
print 'path to directory to store patches:',pickel_dirsource

patch_dir=os.path.join(topdir,pickel_dirsource)

print ('classification used:')
for f in usedclassif:
    print (f, classif[f])



#define the name of directory for patches
patchesdirnametop = 'th'+str(round(thrpatch,2))+'_'+toppatch+'_'+extendir+'_'+extendir0+extendir1

hugeClass=['healthy','back_ground']
#hugeClass=['']

#input patch directory
patch_dirsource=os.path.join(topdir,patchesdirnametop)
patch_dirsource=os.path.join(patch_dirsource,patch_dirsource)
patch_dirsource=os.path.join(patch_dirsource,picklepath)
print 'path for pickle inputs',patch_dirsource
print 'path to write data for training',patch_dir
if not os.path.isdir(patch_dirsource):
    print 'directory ',patch_dirsource,' does not exist'
    sys.exit()
print '--------------------------------'

###############################################################
if not os.path.exists(patch_dir):
    os.mkdir(patch_dir)
eferror=os.path.join(patch_dir,perrorfile)
errorfile = open(eferror, 'a')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started at :'+todayn)
errorfile.write('--------------------\n')
errorfile.write('numbe of loops :'+str(augf)+'\n')
errorfile.write('percentage split: :'+str(test_size)+'\n')
errorfile.write('path for pickle inputs'+patch_dirsource+'\n')
errorfile.write('path to write data for training'+patch_dir+'\n')
errorfile.write('pattern set :'+str(setdata)+'\n')
print 'number of loops:', augf
print 'percentage split:', test_size
errorfile.write('--------------------\n')
errorfile.close()


#define a dictionary with labels
patclass={}

print '----------'
#define another dictionaries to calculate the number of label
classNumberInit={}

for f in usedclassif:
    classNumberInit[f]=0
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
errorfile = open(eferror, 'a')
errorfile.write('number of patches init \n')
for f in usedclassifFinal:
    print ('class:',f,classNumberInit[f])
    errorfile.write('class ' +f+' '+str(classNumberInit[f])+'\n')
    total=total+classNumberInit[f]
print('total:',total)
errorfile.write('total ' +str(total)+'\n')
errorfile.write('--------------------\n')
errorfile.close()
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

errorfile = open(eferror, 'a')
errorfile.write ('max number of patches : '+str(maxl)+ ' for:' +patmax+'\n')
errorfile.write('--------------------\n')
errorfile.close()
for i in hugeClass:
    if i in usedclassif:
        print ('number of patches : ',classNumberInit[i],' for: ', i)
print '----------'
#define coeff min
minl=100000000

for f in usedclassifFinal:
   if classNumberInit[f]<minl:
      minl=classNumberInit[f]
      patmin=f
print ('min number of patches : ',str(minl), ' for:' ,patmin)
errorfile = open(eferror, 'a')
errorfile.write ('min number of patches  : '+str(minl)+ ' for:' +patmin+'\n')
print '----------'
errorfile.write('--------------------\n')
errorfile.close()
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
    avpavg=0
    nump=0
    for p in patclass[category]:
            pf = p.astype('float32')
            normp=norm(pf)
            feature.append(normp)
            avm=np.mean(normp)
            nump+=1
            avpavg=avpavg+avm
            label.append(lc)
#            print avm,avpavg
    avmr=avpavg/nump
    return feature,label,avmr


def geneaug(image,tt):
    if tt==0:
        imout=image
    elif tt==1:
    # 1 90 deg
        imout = np.rot90(image,1)
    elif tt==2:
    #2 180 deg
        imout = np.rot90( image,2)
    elif tt==3:
    #3 270 deg
        imout = np.rot90(image,3)
    elif tt==4:
    #4 flip fimage left-right
            imout=np.fliplr(image)
    elif tt==5:
    #5 flip fimage left-right +rot 90
        imout = np.rot90(np.fliplr(image))
    elif tt==6:
    #6 flip fimage left-right +rot 180
        imout = np.rot90(np.fliplr(image),2)
    elif tt==7:
    #7 flip fimage left-right +rot 270
        imout = np.rot90(np.fliplr(image),3)
  
    return imout


def genf(features_train,labels_train,maxl):
    feature=[]
    label=[]
    indexpatc={}
    maxindex=0
#    print numclass,maxl
    for j in usedclassifFinal:
        indexpatc[j]=0
    for aug in range(augf):
            for numgen in range(maxl*numclass): 
#            for numgen in range(20):    

                    pat =usedclassifFinal[numgen%numclass]
#                    print pat,numgen%numclass
                    numberscan=len(features_train[pat])
#                    print pat,len(features_train[pat])
                    if  pat in hugeClass:
                         indexpatc[pat] =  random.randint(0, numberscan-1)                       
                    else:                                                   
                        indexpatc[pat] =  indexpatc[pat]%numberscan
                        if indexpatc[pat]>maxindex:
                            maxindex=indexpatc[pat]
                            patmax=pat
                    
                    indexpat=indexpatc[pat]
                    indexpatc[pat]=indexpatc[pat]+1           
                    indexaug = random.randint(0, 7)
                    scan=geneaug(features_train[pat][indexpat],indexaug)
#                    print pat,indexpat,indexaug
                    
                    mask=classif[pat]
                    feature.append(scan)
                    label.append(mask)
                    if numgen%numclass==0:
                        pat='healthy'
                        numberscan=len(features_train[pat])
                        indexpat =  random.randint(0, numberscan-1)   
                        indexaug = random.randint(0, 7)
                        scan=geneaug(features_train[pat][indexpat],indexaug)
                        mask=classif[pat]
                        feature.append(scan)
                        label.append(mask)
#                        print mask,numgen
                    
    print 'max index',patmax,maxindex

    return feature,label

                
####################################################################################
# main program

feature_d ={}
label_d={}

print ('----------------')
avgp=0
nump=0
for f in usedclassifFinal:
     print('gene normalize data from images work on :',f)
     feature_d[f],label_d[f],avgpl=genedata(f)
#     print np.array(feature_d[f]).min(),np.array(feature_d[f]).max()
     avgp=avgp+avgpl
     nump+=1
     print ('---')

print 'average data scan',avgp/nump 
for f in usedclassifFinal:
    print 'initial ',f,len(feature_d[f]),len(label_d[f])

print ('----------------') 

feature_test ={}
label_test={}
feature_train ={}
label_train={}
print 'split data'
for f in usedclassifFinal:
    feature_train[f], feature_test[f], label_train[f], label_test[f] =  train_test_split(
            feature_d[f],label_d[f],test_size=test_size)

print 'after split '
maxltraining=0
maxltest=0
errorfile = open(eferror, 'a')

for f in usedclassifFinal:
    ltrain=len(feature_train[f])
    ltest=len(feature_test[f])
    print 'training ',f,ltrain,len(label_train[f])
    print 'test ',f,ltest,len(label_test[f])
    print '---'
    errorfile.write('split data training '+' '+f+' '+str(ltrain)+' '+str(len(label_train[f]))+'\n')   
    errorfile.write('split data test '+' '+f+' '+str(ltest)+' '+str(len(label_test[f]))+'\n')   
    errorfile.write('--\n')  
    if f not in hugeClass:
        if ltrain>maxltraining:
            maxltraining=ltrain
            maxpattrain=f
        if ltest>maxltest:
            maxltest=ltest
            maxpatest=f
print '------'
errorfile.close()
print 'max training',maxltraining, 'for: ',maxpattrain
print 'max test',maxltest, 'for:', maxpatest

print '----'
features_train_f=[]
labels_train_f=[]
features_test_f=[]
labels_test_f=[]
print 'augmentation training'
features_train_f,labels_train_f= genf(feature_train,label_train,maxltraining)
print 'augmentation test'
features_test_f,labels_test_f= genf(feature_test,label_test,maxltest)

print 'data training after augmentation ',len(features_train_f),len(labels_train_f)
print 'data test after augmentation ',len(features_test_f),len(labels_test_f)
print '----'
errorfile = open(eferror, 'a')
errorfile.write('data training after augmentation '+' '+str(len(features_train_f))+' '+str(len(labels_train_f))+'\n')   
errorfile.write('data test after augmentation '+' '+str(len(features_test_f))+' '+str(len(labels_test_f))+'\n')   
errorfile.write('--------------------\n')    
errorfile.close()

print ('------------')
errorfile = open(eferror, 'a')
errorfile.write('Number of data and label \n')
print ('Number of data and label ')
print 'train',len(features_train_f),len(labels_train_f)
errorfile.write('train '+' '+str(len(features_train_f))+' '+str(len(labels_train_f))+'\n')   
print 'test',len(features_test_f),len(labels_test_f)
errorfile.write('test '+' '+str(len(features_test_f))+' '+str(len(labels_test_f))+'\n')
print ('----')
errorfile.write('--------------------\n')
errorfile.close()
# transform dataset list into numpy array
X_train = np.array(features_train_f)
y_train = np.array(labels_train_f)

X_test = np.array(features_test_f)
y_test = np.array(labels_test_f)

print '-----------FINAL----------------'
print ('Xtrain :',X_train.shape)
print ('Xtest : ',X_test.shape)
print ('ytrain : ',y_train.shape)
print 'ytrain min max', y_train.min(),y_train.max()
print ('ytest : ',y_test.shape)
errorfile = open(eferror, 'a')
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
mean_val=np.mean(recuperated_X_train)
print 'recuperated_X_train', min_val, max_val,mean_val
