# -*- coding: utf-8 -*-
"""
Created on Tue August 21 15:04:39 2017
split database in training and validation
Create validation data for training 
use ROI data to generate dat: 1 pattern after another + patchassembly
only one validation set for all training sets
2nd step
@author: sylvain

"""

#from __future__ import print_function
from param_pix import cwdtop,image_rows,image_cols

from param_pix import remove_folder,normi,fidclass
from param_pix import classif,generandom,geneaug

import cPickle as pickle
#import cv2
import collections
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
#import shutil
#import math
import cv2
import keras
print ' keras.backend.image_data_format :',keras.backend.image_data_format()
#######################################################################################################

nameHug='IMAGEDIR'
toppatch= 'TOPROI' #for scan classified ROI
toppatch= 'TOPVAL' #for scan classified ROI
extendir='jc'  #for scan classified ROI
#extendir='ILD6'  #for scan classified ROI
rand=False # False means no random but all, only when valshare =100
valshare=30 #percentage for validation set
numturn=1#number of turn for validation
pickel_dirsource_root='TRAIN_SET' #path for data fort training
pickel_dirsource='pickle' #path for data fort training
pickel_dirsourcenum='train_val' #extensioon for path for data for training
extendir2='1'
#extendir2='ild6'
calculOnly=True
calculOnly=False
bigpat='healthy'

#all in percent
maxshiftv=0
maxshifth=0
maxrot=7 #7
maxresize=0
maxscaleint=0
maxmultint=0
notToAug=['']
##############################################################
validationdir='V'
#sepextend2='ROI'
if len (extendir2)>0:
    extendir2='_'+extendir2
#path for cnn training data recording
pickle_dir=os.path.join(cwdtop,pickel_dirsource_root)
pickle_dir=os.path.join(pickle_dir,pickel_dirsource+'_'+pickel_dirsourcenum+extendir2)


print 'path to write data for training',pickle_dir

if not os.path.exists(pickle_dir):
   os.mkdir(pickle_dir)

path_HUG=os.path.join(cwdtop,nameHug)
patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
#patchesdirnametopt = toppatch+'_T_'+extendir
#patchesdirnametopv = toppatch+'_V_'+extendir
#patchtoppatht=os.path.join(path_HUG,patchesdirnametopt)
#patchtoppathv=os.path.join(path_HUG,patchesdirnametopv)

#remove_folder(patchtoppatht)
#os.mkdir(patchtoppatht)
#remove_folder(patchtoppathv)
#os.mkdir(patchtoppathv)

print 'work on :',patchtoppath , 'for scan data input '
#print 'creation :',patchtoppatht , 'for training data input'
#print 'creation :',patchtoppathv , 'for valid data input'

patchpicklename='picklepatches.pkl'
roipicklepath = 'roipicklepatches'
picklepatches='picklepatches'
picklepathdir =os.path.join(patchtoppath,roipicklepath) # path scan classified by ROI
#picklepathdirt =os.path.join(patchtoppatht,roipicklepath) # path scan classified by ROI for training
#picklepathdirv =os.path.join(patchtoppathv,roipicklepath) # path scan classified by ROI for vali
print 'source',picklepathdir
#print 'destination training',picklepathdirt
#print 'destination validation',picklepathdirv


#remove_folder(picklepathdirt)
#remove_folder(picklepathdirv)
#os.mkdir(picklepathdirv)
#shutil.copytree(picklepathdir, picklepathdirt)
        
def get_class_weights(y):
    counter = collections.Counter(y)
    majority = max(counter.values())
  
#    for cls, count in counter.items():
#        print cls, count
    return  {cls: float(majority/count) for cls, count in counter.items()}

def numbclasses(y):

    y_train=np.array(y)
    uniquelbls = np.unique(y_train)

    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes in this data:', int(nb_classes)) 
    nb_classes = len( classif)
    print ('number of classes in this set:', int(nb_classes)) 
    y_flatten=y_train.flatten()
    class_weights= get_class_weights(y_flatten)
    
    return class_weights


def readclasses(ls,indexpat,scaleint,multint,rotimg,resiz,shiftv,shifth):

    readpkl=pickle.load(open( ls[indexpat], "rb" ))

    scanr=readpkl[0]
    scan=geneaug(scanr,scaleint,multint,rotimg,resiz,shiftv,shifth,False)
    maskr=readpkl[1]
    mask=geneaug(maskr,0,0,rotimg,resiz,shiftv,shifth,True)
    minr=maskr.min()
    manr=maskr.max()
    minm=mask.min()
    manm=mask.max()

    mansr=scanr.max()
    mansm=scan.max()
    if manm==0 or manr==0:
        print 'error label nul'
        sys.exit()
    if mansr==0 or mansm==0:
        print 'error scan nul'
        sys.exit()
    if minr != minm or manr!=manm or manr==0 or manm==0:
        print 'error label'
        print minr,minm,manr,manm,ls[indexpat]
#        cv2.imwrite('images.bmp',normi(scanr))
#        cv2.imwrite('imaget.bmp',normi(scan))
#        cv2.imwrite('labels.bmp',normi( maskr))
#        cv2.imwrite('labelt.bmp',normi( mask))
        sys.exit()

    return scan, mask  

def readclasses2(num_classes,X_testi,y_testi):
 
    DIM_ORDERING=keras.backend.image_data_format()
    
    y_test = np.array(y_testi)

    lytest=y_test.shape[0]
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)
#    ytestr=np.zeros((lytest,int(num_classes),image_rows, image_cols),np.uint8)

    for i in range (lytest):
        for j in range (0,image_rows):
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
            
    if DIM_ORDERING == 'channels_first':
            X_test = np.asarray(np.expand_dims(X_testi,1)) 
            ytestr=np.moveaxis(ytestr,-1,1)
    else:
            X_test = np.asarray(np.expand_dims(X_testi,3))  

    return  X_test,  ytestr  

    
def batch_generator(numsamples,listroi,ls,numgenerate,numgeneratef,augm,te,classnumber):
        image_list = []
        mask_list = []
#        numgen=0
        print 'generation of images for ',te, 'number of samples: ',numsamples
        for numgen in range(numsamples):
            for pat in listroi:
                if pat in notToAug:
                    keepaenh=0
                else:
                    keepaenh=1
                if not augm:
                    keepaenh=0                 
                scaleint,multint,rotimg,resiz,shiftv,shifth=generandom(maxscaleint,
                            maxmultint,maxrot,maxresize,maxshiftv,maxshifth,keepaenh)
                numgeneratef[pat]+=1
                numberscan=classnumber[pat]
                indexpat=numgenerate[pat]
                img,mask=readclasses(ls[pat],indexpat,scaleint,multint,rotimg,resiz,shiftv,shifth) 
                if  img.max() ==0:
                    print ('error')
                    sys.exit()
                numgenerate[pat]=(numgenerate[pat]+1)%numberscan
#                print numgen, pat,n,ls[pat]
                image_list.append(img)
                mask_list.append(mask)
                
#        print '--------------'
        return  image_list,  mask_list,numgeneratef

###########################################################"

listroi=[name for name in os.listdir(picklepathdir)]

numclass=len(listroi)
print '-----------'
print'number of classes in scan:', numclass
print '-----------'
num_classes=len(classif)
print '-----------'
print'number of classes in setdata:', num_classes
print '-----------'
print listroi

listscaninroi={}
listscaninroisub={}
classnumberval={}
classnumbertrain={}

listtoroi=[]
totalimages=0
totalimagesval=0
totalimagestrain=0
maximage=0
maximageval=0
maximagetrain=0
ptmaxtrain='non'
for c in listroi:
    print 'work on ',c
    dirsource=os.path.join(picklepathdir,c)    
    listscaninroi[c]=os.listdir(dirsource) 
    numberscanorig=len(listscaninroi[c]) 
    for u in range(0,numberscanorig):
        listscaninroi[c][u]=os.path.join(dirsource,listscaninroi[c][u])
    totalimages+=numberscanorig
    if numberscanorig>maximage:
        maximage=numberscanorig
        maximagesc=c

    num_to_select=max(valshare*numberscanorig/100,2)
    listscaninroisub[c]= random.sample(listscaninroi[c], num_to_select)
#    print c,listscaninroisub[c],num_to_select

    for p in listscaninroisub[c]:
        listscaninroi[c].remove(p)

    classnumberval[c]=len(listscaninroisub[c])

    if classnumberval[c]>maximageval:
        if c !=bigpat:
            maximageval=classnumberval[c]
            ptmaxval=c
    totalimagesval+=classnumberval[c]
    
    classnumbertrain[c]=len(listscaninroi[c])
#    print c,numberscan
    if classnumbertrain[c]>maximagetrain:
        if c !=bigpat:
            maximagetrain=classnumbertrain[c]
            ptmaxtrain=c
    totalimagestrain+=classnumbertrain[c]
    
 
print 'number total of scan images:',totalimages
print 'maximum data in one pat in source:',maximage,' in: ',maximagesc
print 'number total of scan images in validation:',totalimagesval
print 'number of turns:',numturn
print 'maximum data in one pat in val:',maximageval,' in ',ptmaxval
print 'number total of scan images in training:',totalimagestrain
print 'maximum data in one pat in train:',maximagetrain,' in ',ptmaxtrain
print '-----------'

#print 'val', listscaninroisub['air_trapping']
#print 'test',listscaninroi['air_trapping']
numgenerateval={}
numgeneratevalf={}

numgeneratetrain={}
numgeneratetrainf={}
print 'number of images per pattern in validation:'
for pat in listroi:
    numgenerateval[pat]=0
    numgeneratevalf[pat]=0
    print pat,classnumberval[pat]
print '-----------'
print 'number of images per pattern in training:'
for pat in listroi:
    numgeneratetrain[pat]=0
    numgeneratetrainf[pat]=0
    print pat,classnumbertrain[pat]
print '-----------'

#validation"
numsamples=maximageval*numturn
image_list,mask_list,numgeneratevalf =  batch_generator(numsamples,listroi,listscaninroisub,
                                                      numgenerateval,numgeneratevalf,False,'validation',classnumberval)
X_test,  Y_test  =readclasses2(num_classes,image_list,mask_list)
print 'number of images validation generated',len(X_test)
for pat in listroi:
    print pat,numgeneratevalf[pat]
print 'which is :',valshare,'% of ',totalimages,'multiplied by number of classes and number of turns:',maximage*numclass*numturn
print '-----------'
class_weights=numbclasses(mask_list)
#train
#numsamples=maximagetrain*numturn
#image_list,mask_list,numgeneratetrainf =  batch_generator(numsamples,listroi,listscaninroi,
#                                                          numgeneratetrain,numgeneratetrainf,True,'training',classnumbertrain)
#
##print class_weights
#X_train,  Y_train  =readclasses2(num_classes,image_list,mask_list)
#print 'number of images training generated',len(X_train)
#
#for pat in listroi:
#    print pat,numgeneratetrainf[pat]
#print 'which is :',100-valshare,'% of ',totalimages,'multiplied by number of classes and number of turns:',maximagetrain*numclass*numturn

if calculOnly==True:
    sys.exit()
print '-----------'
print 'weights:'
setvalue=[]
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
#   class_weights[key]=math.log10(value)+0.1
   setvalue.append(key)
print('-' * 30)
print 'after adding for non existent :'
for numw in range(num_classes):
    if numw not in setvalue:
        class_weights[numw]=1
#class_weights[0]=0.05
for key,value in class_weights.items():
   print key, fidclass (key,classif), value;
print('-' * 30)

print 'shape X_test :',X_test.shape
print 'shape y_test :',Y_test.shape
#print 'shape X_train :',X_train.shape
#print 'shape Y_train :',Y_train.shape
print '-----------'
diri=os.path.join(pickle_dir,validationdir)
remove_folder(diri)
os.mkdir(diri)
pickle.dump(X_test, open( os.path.join(diri,"X_test.pkl"), "wb" ),protocol=-1)
pickle.dump(Y_test, open( os.path.join(diri,"Y_test.pkl"), "wb" ),protocol=-1)
#pickle.dump(X_train, open( os.path.join(diri,"X_train.pkl"), "wb" ),protocol=-1)
#pickle.dump(Y_train, open( os.path.join(diri,"Y_train.pkl"), "wb" ),protocol=-1)
pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ),protocol=-1)
#print class_weights

debug=True
if debug:
        print 'debug test'
        xt=  pickle.load(open( os.path.join(diri,"X_test.pkl"), "rb" ))        
        yt= pickle.load(open( os.path.join(diri,"Y_test.pkl"), "rb" ))
        
        DIM_ORDERING=keras.backend.image_data_format()
        print DIM_ORDERING
        if DIM_ORDERING == 'channels_first':
            xt=np.squeeze(xt,1)
            yt=np.moveaxis(yt,1,-1)
        else:
            xt=np.squeeze(xt,-1)
        print 'xt', xt.shape
                   
        xcol=30
        ycol=20
        for i in range(3):
            numtosee=i
            print 'numtosee',numtosee
            print 'xtest shape', xt.shape
            print 'type xt', type(xt[numtosee][0][0])
            print 'xt min max',xt.min(),xt.max()
            print 'xt[0][0][0]',xt[numtosee][0][0]
            print 'xt[0][350][160]',xt[numtosee][ycol][xcol]
            print 'yt', yt.shape
        
            print 'yt[0][0][0]',yt[numtosee][0][0]
            print 'yt[0][350][160]',yt[numtosee][ycol][xcol]
            print 'xt numtosee min max', xt[numtosee].min(), xt[numtosee].max()
            print 'yt numtosee min max',yt[numtosee].min(), yt[numtosee].max()
#            print xt[numtosee][:,:,0].shape
#            cv2.imwrite(str(i)+'image.bmp',normi(xt[numtosee][:,:,0]))
#            cv2.imwrite(str(i)+'label.bmp',normi( np.argmax(yt[numtosee],axis=2)))
            plt.figure(figsize = (5, 5))
            #    plt.subplot(1,3,1)
            #    plt.title('image')
            
            #    plt.imshow( np.asarray(crpim) )
            plt.subplot(1,2,1)
            plt.title(str(i)+'image')
            plt.imshow( normi(xt[numtosee]*10).astype(np.uint8) )
            plt.subplot(1,2,2)
            plt.title(str(i)+'label')
            plt.imshow( np.argmax(yt[numtosee],axis=2) )
            plt.show()
#        print 'debug train'
#        xt=  pickle.load(open( os.path.join(diri,"X_train.pkl"), "rb" ))        
#        yt= pickle.load(open( os.path.join(diri,"Y_train.pkl"), "rb" ))
#        DIM_ORDERING=keras.backend.image_data_format()
#        print DIM_ORDERING
#        if DIM_ORDERING == 'channels_first':
#            xt=np.squeeze(xt,1)
#            yt=np.moveaxis(yt,1,-1)
#        else:
#            xt=np.squeeze(xt,-1)
#        print 'xt', xt.shape
#                   
#        xcol=30
#        ycol=20
#        for i in range(3):
#            numtosee=i
#            print 'numtosee',numtosee
#            print 'xt train', xt.shape
#            print 'type xt', type(xt[numtosee][0][0])
#            print 'xt min max',xt.min(),xt.max()
#            print 'xt[0][0][0]',xt[numtosee][0][0]
#            print 'xt[0][350][160]',xt[numtosee][ycol][xcol]
#            print 'yt', yt.shape
#        
#            print 'yt[0][0][0]',yt[numtosee][0][0]
#            print 'yt[0][350][160]',yt[numtosee][ycol][xcol]
#            print 'xt numtosee  min max', xt[numtosee].min(), xt[numtosee].max()
#            print 'yt numtosee min max',yt[numtosee].min(), yt[numtosee].max()
##            print xt[numtosee][:,:,0].shape
#            if  xt[numtosee].max()==0:
#                cv2.imwrite(str(i)+'image.bmp',normi(xt[numtosee]))
#                cv2.imwrite(str(i)+'label.bmp',normi( np.argmax(yt[numtosee],axis=2)))
#            plt.figure(figsize = (5, 5))
#            #    plt.subplot(1,3,1)
#            #    plt.title('image')
#            
#            #    plt.imshow( np.asarray(crpim) )
#            plt.subplot(1,2,1)
#            plt.title(str(i)+'image')
#            plt.imshow( normi(xt[numtosee]).astype(np.uint8) )
#            plt.subplot(1,2,2)
#            plt.title(str(i)+'label')
#            plt.imshow( np.argmax(yt[numtosee],axis=2) )
#            plt.show()
            
