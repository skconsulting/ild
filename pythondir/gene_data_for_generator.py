# -*- coding: utf-8 -*-
"""
Created on Tue August 21 15:04:39 2017
Create validation data for training 
use ROI data to generate dat: 1 pattern after another + patchassembly
only one validation set for all training sets
2nd step
@author: sylvain

"""

#from __future__ import print_function
from param_pix import cwdtop,image_rows,image_cols

from param_pix import remove_folder,normi,fidclass
from param_pix import classif

import cPickle as pickle
#import cv2
import collections
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
#import math
#import cv2
import keras
print ' keras.backend.image_data_format :',keras.backend.image_data_format()
#######################################################################################################

nameHug='IMAGEDIR'
toppatch= 'TOPVAL' #for scan classified ROI
extendir='0'  #for scan classified ROI
#extendir='essai'  #for scan classified ROI
rand=False # False means no random but all, only when valshare =100
valshare=100 #percentage for validation set

pickel_dirsource_root='TRAIN_SET' #path for data fort training
pickel_dirsource='pickle' #path for data fort training
pickel_dirsourcenum='train_val' #extensioon for path for data for training
extendir2='0'
calculOnly=True
calculOnly=False
##############################################################
validationdir='V'
#sepextend2='ROI'
if len (extendir2)>0:
    extendir2='_'+extendir2
#path for cnn training data recording
pickle_dir=os.path.join(cwdtop,pickel_dirsource_root)
pickle_dir=os.path.join(pickle_dir,pickel_dirsource+'_'+pickel_dirsourcenum+extendir2)


print 'path to write data for training',pickle_dir

remove_folder(pickle_dir)
os.mkdir(pickle_dir)


path_HUG=os.path.join(cwdtop,nameHug)
patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
print 'work on :',patchtoppath , 'for scan data input '
patchpicklename='picklepatches.pkl'
roipicklepath = 'roipicklepatches'
picklepatches='picklepatches'
picklepathdir =os.path.join(patchtoppath,roipicklepath) # path scan classified by ROI

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

def readclasses(pat,namepat,indexpat,indexaug):

    patpick=os.path.join(picklepathdir,pat)
    patpick=os.path.join(patpick,namepat[indexpat])
    readpkl=pickle.load(open(patpick, "rb"))
    scanr=readpkl[0]

    scan=geneaug(scanr,indexaug)
    maskr=readpkl[1]

    mask=geneaug(maskr,indexaug)

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


def gen_random_image(numgen,numclass,listroi,listscaninroi,numgenerate):

    pat =listroi[numgen%numclass]
    numgenerate[pat]+=1
    numberscan=classnumber[pat]
    indexpat =  random.randint(0, numberscan-1)  
    indexaug = random.randint(0, 11)
    scan,mask=readclasses(pat,listscaninroi[pat],indexpat,indexaug)  
    
    numgen+=1
    return scan,mask,numgen,numgenerate

    
def batch_generator(numsamples,numclass,num_classes,listroi,listscaninroi, rand,numgenerate):
        image_list = []
        mask_list = []
        numgen=0
        print 'generation of images for validation'
        if rand:
            for i in range (numsamples):
    
                img, mask,numgen,numgenerate = gen_random_image(numgen,numclass,listroi,listscaninroi,numgenerate)
                image_list.append(img)
                mask_list.append(mask)
        else:
            for pat in listroi:
                numberscan=classnumber[pat]
                for n in range(numberscan):
                    numgenerate[pat]+=1
                    img,mask=readclasses(pat,listscaninroi[pat],n,0) 
#                    print pat,n
                    image_list.append(img)
                    mask_list.append(mask)
                    numgen+=1
                    if numgen==numsamples:
                        break
                if numgen==numsamples:
                        break
                
                
        print '--------------'
        return  image_list,  mask_list ,numgenerate

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

listscaninroi={}
classnumber={}

listtoroi=[]
totalimages=0
maximage=0


for c in listroi:
    listscaninroi[c]=os.listdir(os.path.join(picklepathdir,c))
#    print listscaninroi[c]
#    for k in listscaninroi[c]:
        
    numberscan=len(listscaninroi[c])
    classnumber[c]=numberscan
#    print c,numberscan
    if numberscan>maximage:
        maximage=numberscan
        ptmax=c
    totalimages+=numberscan
 
print 'number total of scan images:',totalimages
print 'maximum data in one pat:',maximage,' in ',ptmax
print '-----------'
numgenerate={}
print 'number of images per pattern:'
for pat in listroi:
    numgenerate[pat]=0
    print pat,classnumber[pat]
print '-----------'

numsamples=int(1.0*totalimages*valshare/100.)

image_list,mask_list,numgenerate =  batch_generator(numsamples,numclass,num_classes,listroi,listscaninroi,rand,numgenerate)
#print len(mask_list)
print 'number of images per pattern:'
for pat in listroi:
    print pat,numgenerate[pat]
class_weights=numbclasses(mask_list)
#print class_weights
X_test,  Y_test  =readclasses2(num_classes,image_list,mask_list)
print 'number of images generated',len(X_test)
print 'which is :',valshare,'% of ',totalimages
if calculOnly==True:
    sys.exit()

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

#X_test, Y_test= readclasses2(num_classes,x_test,y_test)

print 'shape y_train :',X_test.shape
print 'shape y_test :',Y_test.shape
print '-----------'
diri=os.path.join(pickle_dir,validationdir)
remove_folder(diri)
os.mkdir(diri)
pickle.dump(X_test, open( os.path.join(diri,"X_test.pkl"), "wb" ),protocol=-1)
pickle.dump(Y_test, open( os.path.join(diri,"Y_test.pkl"), "wb" ),protocol=-1)
pickle.dump(class_weights, open( os.path.join(pickle_dir,"class_weights.pkl"), "wb" ),protocol=-1)
#print class_weights

debug=True
if debug:
        print 'debug'
        xt=  pickle.load(open( os.path.join(diri,"X_test.pkl"), "rb" ))        
        yt= pickle.load(open( os.path.join(diri,"Y_test.pkl"), "rb" ))
        print 'xt', xt.shape
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
            print 'xt', xt.shape
            print 'type xt', type(xt[numtosee][0][0])

            print 'xt[0][0][0]',xt[numtosee][0][0]
            print 'xt[0][350][160]',xt[numtosee][ycol][xcol]
            print 'yt', yt.shape
        
            print 'yt[0][0][0]',yt[numtosee][0][0]
            print 'yt[0][350][160]',yt[numtosee][ycol][xcol]
            print 'xt min max', xt[numtosee].min(), xt[numtosee].max()
            print 'yt min max',yt[numtosee].min(), yt[numtosee].max()
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
