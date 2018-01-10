# coding: utf-8
'''create dataset from patches for training, all patches with augmentation
support Hu
no split for validation data, done at training

version 1.0
second step
S. Kritter
18 august 2017
 '''
from param_pix_t import classif
from param_pix_t import thrpatch,setdata
from param_pix_t import norm,normi
from param_pix_t import picklepath,perrorfile,generandom,geneaug
import datetime
import os
import sys

import cv2
import numpy as np
#from sklearn.model_selection import train_test_split
import cPickle as pickle
import random
from random import shuffle
#####################################################################
#define the working directory for input patches
topdir='C:/Users/sylvain/Documents/boulot/startup/radiology/traintool'
#toppatch= 'TOPPATCH'
#extendir='all'
#extendir='JC'
patchesdirnametop = ['th0.95_TOPPATCH_all_HUG']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2']
patchesdirnametop = patchesdirnametop+['th0.95_TOPPATCH_all_CHU2new']
#extendir0='5'
#extendir0='essai'
#extendir0='5'
extendir1=''

#define the directory to store data
pickel_dirsource_root='pickle'
pickel_dirsource_e='train' #path for data fort training
pickel_dirsourcenum=setdata #extensioon for path for data for training
extendir2='0'
#extendir2='essai'
extendir3=extendir1

augf=5#augmentation factor default 3

#all in percent
maxshiftv=0
maxshifth=0
maxrot=7 #7
maxresize=0
maxscaleint=20
maxmultint=20
notToAug=['ground_glass']
notToAug=['']

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
for f in classif:
    print (f, classif[f])



#define the name of directory for patches
#patchesdirnametop = 'th'+str(round(thrpatch,2))+'_'+toppatch+'_'+extendir+'_'+extendir0+extendir1

hugeClass=['healthy','back_ground']
#hugeClass=['']

#input patch directory
#patch_dirsource=os.path.join(topdir,patchesdirnametop)
#patch_dirsource=os.path.join(patch_dirsource,patch_dirsource)
#patch_dirsource=os.path.join(patch_dirsource,picklepath)
print 'path for pickle inputs',patchesdirnametop
print 'path to write data for training',patch_dir
#if not os.path.isdir(patch_dirsource):
#    print 'directory ',patch_dirsource,' does not exist'
#    sys.exit()
#print '--------------------------------'

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
errorfile.write('path for pickle inputs'+str(patchesdirnametop)+'\n')
errorfile.write('path to write data for training'+patch_dir+'\n')
errorfile.write('pattern set :'+str(setdata)+'\n')
print 'number of loops:', augf

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

# go through all categories to calculate the number of patches per class
for category in usedclassifFinal:
    for f in patchesdirnametop:
        patch_dirsource=os.path.join(topdir,f)
        patch_dirsource=os.path.join(patch_dirsource,picklepath)
        category_dir = os.path.join(patch_dirsource, category)
    #    print  'the path into the categories is: ', category_dir
        if os.path.exists(category_dir):
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
    if classNumberInit[category]==0:
            print category, 'is empty'
            usedclassifFinal.remove(category)

print ('all non empty actual classes:',usedclassifFinal)
numclass= len (usedclassifFinal)
print 'number of classes:', numclass
print '----------'

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
    classConso[f]=float(maxl)/max(1,classNumberInit[f])
#for f in usedclassifFinal:
    print (f,classif[f],' {0:.2f}'.format (classConso[f]))
print '----------'


def genedata(category):
    """include normalisation"""
    feature=[]
    label=[]
    lc=classif[category]
    for p in patclass[category]:
        
#        kernel=(3,3)
##        imageview=cv2.blur(imageview,kernel)
#        
#        p1=cv2.medianBlur(p,kernel[0])
#        cv2.imwrite('orig.bmp',normi(p))
#        cv2.imwrite('blur.bmp',normi(p1))
#        cv2.imwrite('blurnorm.bmp',normi(norm(p1)))
#
#        ooo
        feature.append(norm(p))
        label.append(lc)
    return feature,label


def genfnoopt(features_train,labels_train,maxl):
    feature=[]
    label=[]
    numpat={}
    for pat in usedclassifFinal:
        numpat[pat]=0
    for pat in usedclassifFinal:
        numberscan1=len(features_train[pat])
        print pat,numberscan1,maxl
        numberscan=min(numberscan1,maxl)
#        print pat,numberscan,maxl
        for num in range(0,augf):
#            print 'init',num,numpat
            for i in range(0,numberscan):
                numpat[pat]+=1
                if num==0:
                     if  pat in hugeClass:
                         indexpatc =  random.randint(0, len(features_train[pat])-1)  
                         scan=features_train[pat][indexpatc]
                     else:
                         scan=features_train[pat][i]
                else:
                    if pat in notToAug:
                        keepaenh=0
                    else:
                        keepaenh=1
                    scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,
                                maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
                    if  pat in hugeClass:
                         indexpatc =  random.randint(0, numberscan-1)  
                         scan=geneaug(features_train[pat][indexpatc],scaleint,multint,rotimg,resiz,shiftv,shifth)
                    else:
                         scan=geneaug(features_train[pat][i],scaleint,multint,rotimg,resiz,shiftv,shifth)
#                tt=(scan,mask)
#                featurelab.append(tt)
                mask=labels_train[pat][i]
                feature.append(scan)
                label.append(mask)
            if  pat not in hugeClass:        
                    for i in range(0,maxl-numberscan):
                        numpat[pat]+=1
                        if pat in notToAug:
                            keepaenh=0
                        else:
                            keepaenh=1
                        scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,
                               maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
                        scan=geneaug(features_train[pat][i%numberscan],scaleint,multint,rotimg,resiz,shiftv,shifth)
                        mask=labels_train[pat][i%numberscan]  
                        feature.append(scan)
                        label.append(mask)
#            print 'end',num,numpat
    
            """
            for i in range(numberscan):
                pat1 = 'healthy'
                numpat[pat1]+=1
                indexpatc =  random.randint(0,  len(features_train[pat1]) -1)  
                scan=features_train[pat1][indexpatc]
                mask=classif[pat1]
                feature.append(scan)
                label.append(mask)
            print 'fend',num,numpat
            """
#    shuffle(featurelab)
#    unzipped = zip(*featurelab )
#    feature=(unzipped[0])
#    label=(unzipped[1])
#    print numpat
#    print len(feature)
    
    for i in range(len(feature)/len(usedclassifFinal)):
                pat1 = 'healthy'
                numpat[pat1]+=1
                indexpatc =  random.randint(0,  len(features_train[pat1]) -1)  
                scan=features_train[pat1][indexpatc]
                mask=classif[pat1]
                feature.append(scan)
                label.append(mask)
    
    return feature,label,numpat


def genf(features_train,labels_train,maxl):
    feature=[]
    label=[]
    indexpatc={}
    maxindex=0
    featurelab=[]
    numpat={}

#    print numclass,maxl
    for j in usedclassifFinal:
        indexpatc[j]=0
        numpat[j]=0
    for aug in range(augf):
        for numgen in range(maxl*numclass):    
            pat =usedclassifFinal[numgen%numclass]
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
#                    indexaug = random.randint(0, 7)
            if pat in notToAug:
#                print pat,'do not augment'
                keepaenh=0
            else:
                keepaenh=1
#                print pat,'augm'
            scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
            if numgen ==0:
                
                print  scaleint,multint,rotimg,resiz,shiftv,shifth
#                    scan=geneaug(features_train[pat][indexpat],indexaug)
            scan=geneaug(features_train[pat][indexpat],scaleint,multint,rotimg,resiz,shiftv,shifth)
#                    print pat,indexpat,indexaug
            
            mask=classif[pat]
            numpat[pat]=+1
#                    print pat,mask
            tt=(scan,mask)
            featurelab.append(tt)
#                    feature.append(scan)
#                    label.append(mask)
    print 'max index',patmax,maxindex
    
    shuffle(featurelab)
    unzipped = zip(*featurelab )
    feature=(unzipped[0])
    label=(unzipped[1])
    return feature,label,numpat

                
####################################################################################
# main program

feature_d ={}
label_d={}
minp=10
maxp=0
print ('----------------')
for f in usedclassifFinal:
     print('genedata from images work on :',f)
     feature_d[f],label_d[f]=genedata(f)
     try:
         minf=np.array(feature_d[f]).min()
         maxf=np.array(feature_d[f]).max()
         if minf<minp:
             minp=minf
         if maxf>maxp:
             maxp=maxf
     except:
                continue
    
     print ('---')
print 'min max of data',minp,maxp
print 'initial'
for f in usedclassifFinal:
    print f,len(feature_d[f]),len(label_d[f])

print ('----------------') 

print '----'
features_train_f=[]
labels_train_f=[]
print 'augmentation training'
#features_train_f,labels_train_f= genf(feature_d,label_d,maxl)
features_train_f,labels_train_f,numpat= genfnoopt(feature_d,label_d,maxl)
print '----'
print 'after augmentation'
for f in usedclassifFinal:
    print f,numpat[f]

print 'data training after augmentation ',len(features_train_f),len(labels_train_f)
print 'which is maxl * num class * augf',maxl*numclass*augf

print '----'
errorfile = open(eferror, 'a')
errorfile.write('data training after augmentation '+' '+str(len(features_train_f))+' '+str(len(labels_train_f))+'\n')   
  
errorfile.write('--------------------\n')    
errorfile.close()

print ('------------')
errorfile = open(eferror, 'a')
errorfile.write('Number of data and label \n')
print ('Number of data and label ')
print 'train',len(features_train_f),len(labels_train_f)
errorfile.write('train '+' '+str(len(features_train_f))+' '+str(len(labels_train_f))+'\n')   


print ('----')
errorfile.write('--------------------\n')
errorfile.close()
# transform dataset list into numpy array
X_train = np.array(features_train_f)
y_train = np.array(labels_train_f)


print '-----------FINAL----------------'
print ('Xtrain :',X_train.shape)

print ('ytrain : ',y_train.shape)
print 'ytrain min max', y_train.min(),y_train.max()

errorfile = open(eferror, 'a')
errorfile.write('Xtrain : '+' '+str(X_train.shape)+'\n')

errorfile.write('ytrain : '+' '+str(y_train.shape)+'\n') 

errorfile.write('END------------------------------------------------------\n')
errorfile.close()

pickle.dump(X_train, open( os.path.join(patch_dir,"X_train.pkl"), "wb" ),protocol=-1)

pickle.dump(y_train, open( os.path.join(patch_dir,"y_train.pkl"), "wb" ),protocol=-1)


recuperated_X_train = pickle.load( open( os.path.join(patch_dir,"X_train.pkl"), "rb" ) )
min_val=np.min(recuperated_X_train)
max_val=np.max(recuperated_X_train)
mean_val=np.mean(recuperated_X_train)
print 'recuperated_X_train', min_val, max_val,mean_val
