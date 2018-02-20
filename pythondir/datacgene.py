# coding: utf-8
'''create dataset from patches 
support Hu
norm patches using "norm"
split done for validation
version 1.0
second step
S. Kritter
18 august 2017
 '''
 
from param_pix_t import classif
from param_pix_t import thrpatch,setdata
from param_pix_t import norm
from param_pix_t import picklepath,perrorfile,generandom,geneaug

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

patchesdirnametop=[]

#######################   3b ########################
"""
#HUG
patchesdirnametop =  patchesdirnametop+['th0.95_TOPPATCH_all_HUG_ILD_TXT__3b']


#CHU
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP__3b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP_a_3b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP_med_3b']


#CHU2

patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_3_3b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_a3_3b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_med3_3b']


#CHU2new
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_3_3b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_a3_3b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_med3_3b']

############################## 1b   ###########################
"""
#HUG
patchesdirnametop = patchesdirnametop+['th0.8_TOPPATCH_all_HUG_ILD_TXT_med_1b']
#patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_HUG_ILD_TXT_a_1b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_HUG_ILD_TXT__1b']
#patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_HUG_ILD_TXT_med_1b']

#
##
###CHU
#patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP__1b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP_a_1b']
patchesdirnametop = patchesdirnametop+['th0.8_TOPPATCH_all_CHU_UIP_med_1b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP_med_1b']

#
#
##CHU2
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_a3_1b']
patchesdirnametop = patchesdirnametop+['th0.8_TOPPATCH_all_CHU2_UIP_med3_1b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_3_1b']


#CHU2new
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_a3_1b']
#patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP__1b']
patchesdirnametop = patchesdirnametop+['th0.8_TOPPATCH_all_CHU2new_UIP_med3_1b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_med3_1b']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_3_1b']
"""


############################## 3bm5  ###########################
#HUG
#patchesdirnametop = ['th0.95_TOPPATCH_all_HUG_ILD_TXT__3bm']
#patchesdirnametop = ['th0.95_TOPPATCH_all_HUG_ILD_TXT_a_1b']
#patchesdirnametop = ['th0.95_TOPPATCH_all_HUG_ILD_TXT_med_1b']

#CHU
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP__3bm5']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP_med_3bm5']

#CHU2
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_3_3bm5']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_a3_3bm5']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_med3_3bm5']

#CHU2new
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_3_3bm5']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_a3_3bm5']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_med3_3bm5']

############################## 3bm53  ###########################

#HUG
#patchesdirnametop = ['th0.95_TOPPATCH_all_HUG_ILD_TXT__3bm']
#patchesdirnametop = ['th0.95_TOPPATCH_all_HUG_ILD_TXT_a_1b']
#patchesdirnametop = ['th0.95_TOPPATCH_all_HUG_ILD_TXT_med_1b']

#CHU
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP__3bm53']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP_a_3bm53']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU_UIP_med_3bm53']

#CHU2
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_3_3bm53']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_a3_3bm53']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2_UIP_med3_3bm53']

#CHU2new
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_3_3bm53']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_a3_3bm53']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new_UIP_med3_3bm53']

##########################################################################################
"""
#patchesdirnametop = ['th0.95_TOPPATCH_val_REFVALnew_UIPJC__1b']
#patchesdirnametop = ['th0.95_TOPPATCH_val_REFnew_UIP__1b']
#patchesdirnametop = ['th0.8_TOPPATCH_val_REFVALnew_UIPJC__1b']


#print patchesdirnametop


#define the directory to store data
pickel_dirsource_root='pickle'
pickel_dirsource_e='train' #path for data fort training
#pickel_dirsource_e='val' #path for data fort training

dir_train='T'
pickel_dirsourcenum=setdata #extensioon for path for data for training
#extendir2='2'
extendir1='1'
extendir2='1b'
#extendir2='essai'

#augf=1#augmentation factor default 3
test_size=10 #split test training in percent 0 means no training, only val

#all in percent
maxshiftv=0
maxshifth=0
maxrot=7
maxresize=0
maxscaleint=0
maxmultint=0

listdirtop={}
for f in  patchesdirnametop:
    listdirtop[f]=classif
#
listforres={
        'consolidation':0,
#        'HC':1,
#        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
#        'bronchiectasis':6,
        'emphysema':7
#        'GGpret':8 
        }

for f in  patchesdirnametop:
#    print f
#    print f.find('th0.8_')
    if f.find('th0.8_')>=0:
        print 'apply restriction on:',f
        listdirtop[f]=listforres

"""
classif ={
        'consolidation':0,
        'HC':1,
        'ground_glass':2,
        'healthy':3,
        'micronodules':4,
        'reticulation':5,
        'bronchiectasis':6,
        'emphysema':7,
        'GGpret':8 
        }
"""

########################################################################
######################  end ############################################
########################################################################


if len (extendir2)>0:
    extendir2='_'+extendir2

pickel_dirsource='th'+str(thrpatch)+'_'+pickel_dirsource_root+'_'+pickel_dirsource_e+'_'+pickel_dirsourcenum+'_'+extendir1+extendir2
print 'path to directory to store patches:',pickel_dirsource

patch_dir=os.path.join(topdir,pickel_dirsource)
patch_dir_train=os.path.join(patch_dir,dir_train)

print ('classification used:')
for f in classif:
    print (f, classif[f])


hugeClass=['healthy']
#hugeClass=['']

#input patch directory
print 'path for pickle inputs',patchesdirnametop
print 'path to write data for training',patch_dir

print '--------------------------------'

###############################################################
if not os.path.exists(patch_dir):
    os.mkdir(patch_dir)
if not os.path.exists(patch_dir_train):
    os.mkdir(patch_dir_train)
    
eferror=os.path.join(patch_dir,perrorfile)
errorfile = open(eferror, 'w')
tn = datetime.datetime.now()
todayn = str(tn.month)+'-'+str(tn.day)+'-'+str(tn.year)+' - '+str(tn.hour)+'h '+str(tn.minute)+'m'+'\n'
errorfile.write('started at :'+todayn)
errorfile.write('--------------------\n')
#errorfile.write('numbe of loops :'+str(augf)+'\n')
errorfile.write('pattern to augment: \n')
errorfile.write('percentage split: :'+str(test_size)+'\n')
errorfile.write('path for pickle inputs'+str(patchesdirnametop)+'\n')
errorfile.write('path to write data for training'+patch_dir+'\n')
errorfile.write('pattern set :'+str(setdata)+'\n')
#print 'number of loops:', augf
print 'percentage split:', test_size
errorfile.write('--------------------\n')
errorfile.close()


#define a dictionary with labels
patclass={}

print '----------'
#define another dictionaries to calculate the number of label
classNumberInit={}

for f in classif:
    classNumberInit[f]=0
    patclass[f]=[]

# list all directories under patch directory. They are representing the categories
category_list=[]
# list all directories under patch directory. They are representing the categories
for f in  patchesdirnametop:
    print f
    patch_dirsource=os.path.join(topdir,f)
    patch_dirsource=os.path.join(patch_dirsource,picklepath)
    category_list=category_list+os.walk( patch_dirsource).next()[1]
# print what we have as categories
#print ('all classes in database:',category_list)
#print '----------'
category_list=list(set(category_list))
print ('all classes in database:',category_list)
print '----------'

usedclassifFinal=[f for f in classif if f in category_list]

# print what we have as categories and in used one
print ('all actual classes:',usedclassifFinal)

numclass= len (usedclassifFinal)
print 'number of classes:', numclass
print '----------'

# go through all categories to calculate the number of patches per class

for f in patchesdirnametop:
        print 'work on: ',f
        patch_dirsource=os.path.join(topdir,f)
        if not os.path.exists(patch_dirsource):
            print patch_dirsource, 'does not exist'
            sys.exit()
        patch_dirsource=os.path.join(patch_dirsource,picklepath)
#        print patch_dirsource
        for category in listdirtop[f]:
#            if category=='bronchiectasis':
#                print category
                category_dir = os.path.join(patch_dirsource, category)
                if os.path.exists(category_dir):
                    sub_categories_dir_list = (os.listdir(category_dir))
    #                print 'the sub categories are : ', sub_categories_dir_list
                    for subCategory in sub_categories_dir_list:
                        subCategory_dir = os.path.join(category_dir, subCategory)
    #                    print subCategory_dir
                        image_files = [name for name in os.listdir(subCategory_dir) if name.find('.pkl') > 0 ]
#                        print image_files
                        for filei in image_files:
                            aa=pickle.load(open(os.path.join(subCategory_dir,filei),'rb'))
#                            print filei,len (aa)
                            patclass[category]=patclass[category]+aa
for category in usedclassifFinal:
    classNumberInit[category]=len(patclass[category])  

for f in usedclassifFinal:
    if classNumberInit[f]==0:
        usedclassifFinal.remove(f)
print ('all final classes:',usedclassifFinal)    
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
    if i in classif:
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
    if classNumberInit[f]>0:
        classConso[f]=float(maxl)/max(1,classNumberInit[f])
        print (f,classif[f],' {0:.2f}'.format (classConso[f]))
    else:
        classConso[f]='NONE'
        print (f,classif[f],classConso[f])

def genedata(category):
    feature=[]
    label=[]
    lc=classif[category]
    for p in patclass[category]:
#        pf = p.astype('float32')
#        normp=norm(pf)        
        feature.append(p)
        label.append(lc)
    return feature,label

def genf(features_p,maxl):
    feature=[]
    label=[]
    numpat={}
    for pat in usedclassifFinal:
        numpat[pat]=0
    for pat in usedclassifFinal:
        numberscan1=len(features_p[pat])
        numberscan=min(numberscan1,maxl)
        for i in range(0,numberscan):
                numpat[pat]+=1
                if  pat in hugeClass:
                     indexpatc =  random.randint(0, numberscan1-1)  
                     scan=features_p[pat][indexpatc]
                else:                
                    keepaenh=0
                    scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,
                                maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
                    scan=geneaug(features_p[pat][i],scaleint,multint,rotimg,resiz,shiftv,shifth)

                mask=classif[pat]
                feature.append(norm(scan))
                label.append(mask)
        if  pat not in hugeClass:        
                for i in range(0,maxl-numberscan):
                    numpat[pat]+=1
                    keepaenh=0
                    scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,
                           maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
                    scan=geneaug(features_p[pat][i%numberscan],scaleint,multint,rotimg,resiz,shiftv,shifth)
                    mask=classif[pat]
                    feature.append(norm(scan))
                    label.append(mask)
    
    return feature,label,numpat

                
####################################################################################
# main program

feature_d ={}
label_d={}

print ('----------------')
avgp=0
totalp=[]
for f in usedclassifFinal:
     print('gene normalize data from images work on :',f)
     feature_d[f],label_d[f]=genedata(f)
     totalp=totalp+feature_d[f]
     print ('---')
print 'total number of patches:',len(totalp)

for f in usedclassifFinal:
    print 'initial ',f,len(feature_d[f]),len(label_d[f])
print ('----------------') 

feature_test ={}
label_test={}
feature_train ={}
label_train={}
print 'split data',
if test_size>0:
    for f in usedclassifFinal:
        feature_train[f], feature_test[f], label_train[f], label_test[f] =  train_test_split(
            feature_d[f],label_d[f],test_size=test_size/100.)
else:
    for f in usedclassifFinal:
        feature_test[f]=feature_d[f]
        label_test[f]=label_d[f]
        
if test_size>0:       
    for category in usedclassifFinal:
        dirpat=os.path.join(patch_dir_train,category)
        if not os.path.exists(dirpat):
            os.mkdir(dirpat)
        pickle.dump(feature_train[category], open( os.path.join(dirpat,"pat.pkl"), "wb" ),protocol=-1)

print 'after split '
maxltraining=0
maxltest=0
errorfile = open(eferror, 'a')
maxpatest=['None']
maxpattrain=['None']

for f in usedclassifFinal:
    if test_size>0: 
        ltrain=len(feature_train[f])
        print 'training ',f,ltrain,len(label_train[f])
    
    ltest=len(feature_test[f])
    print 'test ',f,ltest,len(label_test[f])
    print '---'
    if test_size>0:
        errorfile.write('split data training '+' '+f+' '+str(ltrain)+' '+str(len(label_train[f]))+'\n')   
    errorfile.write('split data test '+' '+f+' '+str(ltest)+' '+str(len(label_test[f]))+'\n')   
    errorfile.write('--\n')  
    if f not in hugeClass:
        if test_size>0:
            if ltrain>maxltraining:
                maxltraining=ltrain
                maxpattrain=f
        if ltest>maxltest:
                maxltest=ltest
                maxpatest=f
print '------'
errorfile.close()
if test_size>0:print 'max training',maxltraining, 'for: ',maxpattrain
print 'max test',maxltest, 'for:', maxpatest

print '----'

print 'augmentation test'
features_test_f,labels_test_f,numpat= genf(feature_test,maxltest)
print '----'
errorfile = open(eferror, 'a')
print 'after augmentation test'
for f in usedclassifFinal:
    print f,numpat[f]
    errorfile.write('data training after augmentation '+' '+str(numpat[f])+' '+f+' \n')     
errorfile.close()
print '----'
#print 'data training after augmentation ',len(features_train_f),len(labels_train_f)
#if test_size>0: print 'data test after augmentation ',len(features_test_f),len(labels_test_f)
#print '----'
#errorfile = open(eferror, 'a')
##errorfile.write('data training after augmentation '+' '+str(len(features_train_f))+' '+str(len(labels_train_f))+'\n')   
#if test_size>0: errorfile.write('data test after augmentation '+' '+str(len(features_test_f))+' '+str(len(labels_test_f))+'\n')   
#errorfile.write('--------------------\n')    
#errorfile.close()
#
#print ('------------')
#errorfile = open(eferror, 'a')
#errorfile.write('Number of data and label \n')
#print ('Number of data and label ')
##print 'train',len(features_train_f),len(labels_train_f)
##errorfile.write('train '+' '+str(len(features_train_f))+' '+str(len(labels_train_f))+'\n')   
#if test_size>0: print 'test',len(features_test_f),len(labels_test_f)
#if test_size>0: errorfile.write('test '+' '+str(len(features_test_f))+' '+str(len(labels_test_f))+'\n')
#print ('----')
#errorfile.write('--------------------\n')
#errorfile.close()
## transform dataset list into numpy array
##X_train = np.array(features_train_f)
##y_train = np.array(labels_train_f)

X_test = np.array(features_test_f)
y_test = np.array(labels_test_f)

print '-----------FINAL----------------'
#print ('Xtrain :',X_train.shape)

print ('X_test : ',X_test.shape)
print ('y_test : ',y_test.shape)

errorfile = open(eferror, 'a')
errorfile.write('X_test : '+' '+str(X_test.shape)+'\n') 
errorfile.write('y_test : '+' '+str(y_test.shape)+'\n')   
errorfile.write('--------------------\n')
errorfile.close()

#pickle.dump(X_train, open( os.path.join(patch_dir,"X_train.pkl"), "wb" ),protocol=-1)
#pickle.dump(y_train, open( os.path.join(patch_dir,"y_train.pkl"), "wb" ),protocol=-1)

pickle.dump(X_test, open( os.path.join(patch_dir,"X_val.pkl"), "wb" ),protocol=-1)
pickle.dump(y_test, open( os.path.join(patch_dir,"y_val.pkl"), "wb" ),protocol=-1)

    
X_train=[]
features_train_f=[]
features_test_f=[]
y_train=[]
labels_train_f=[]
labels_train_f=[]
print 'recuperate X_test'
recuperated_X_train = pickle.load( open( os.path.join(patch_dir,"X_val.pkl"), "rb" ) )
min_val=np.min(recuperated_X_train)
max_val=np.max(recuperated_X_train)
mean_val=np.mean(recuperated_X_train)
print 'recuperated_X_train', min_val, max_val,mean_val
recuperated_X_train=[]
