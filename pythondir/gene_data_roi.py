# -*- coding: utf-8 -*-
"""
Created on Tue May 02 15:04:39 2017

@author: sylvain
prepare data for segmentation ROI
"""

#from __future__ import print_function
from param_pix import *
import os
import cv2
import numpy as np
from keras.utils import np_utils
import cPickle as pickle
import collections

nameHug='HUG'
toppatch= 'TOPPATCH'
#extension for output dir
extendir='ILD4'
pklnum=1
#extendir='pix1'
pickel_dirsource_root='pickle'
pickel_dirsource=pickel_dirsource_root+'_'+extendir

cwd=os.getcwd()

(cwdtop,tail)=os.path.split(cwd)
pickle_dir=os.path.join(cwdtop,pickel_dirsource)
if not os.path.exists(pickle_dir):
    os.mkdir(pickle_dir)

def get_class_weights(y):
    counter = collections.Counter(y)
    majority = max(counter.values())
#    for cls, count in counter.items():
#        print cls, count
    return  {cls: min(float(majority/count),1000) for cls, count in counter.items()}

#remove_folder(pickle_dir)


path_HUG=os.path.join(cwdtop,nameHug)
patchesdirnametop = toppatch+'_'+extendir
patchtoppath=os.path.join(path_HUG,patchesdirnametop)
patchpicklename='picklepatches.pkl'
picklepath = 'picklepatches'
picklepathdir =os.path.join(patchtoppath,picklepath)

def countclasses():
    for category in usedpatient:
        category_dir = os.path.join(picklepathdir, category)
        print  'the patients are:  ', category
        image_files = [name for name in os.listdir(category_dir) if name.find('.pkl') > 0 ]
        for filei in image_files:
            usedpatient[category]=usedpatient[category]+1

def readclasses1(usedpatient):
    patch_list=[]
    label_list=[]

    for category in usedpatient:
        category_dir = os.path.join(picklepathdir, category)
        print  'work on: ', category
        image_files = [name for name in os.listdir(category_dir) if name.find('.pkl') > 0 ]
        for filei in image_files:
            pos=filei.find('_')
            numslice=filei[0:pos]
#            usedpatient[category]=usedpatient[category]+1
            usedpatientlist[category].append(numslice)
            readpkl=pickle.load(open(os.path.join(category_dir,filei), "rb"))
            for i in range (len(readpkl[0])):
                                
                scan=readpkl[0][i]
                mask=readpkl[1][i]    
                           
                scanr=cv2.resize(scan,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
                maskr=cv2.resize(mask,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
#                print scanr.min(),scanr.max()
                scanm=norm(scanr)
#                print scanm.min(),scanm.max()
                patch_list.append(scanm)
                label_list.append(maskr)
  
#   y_train = np.array(label_list)
    y_train = np.array(label_list)
    uniquelbls = np.unique(y_train)
    for pat in uniquelbls:
#        print pat
        print  fidclass(pat,classif)
    
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes :', int(nb_classes)) 
    return patch_list, label_list   

def numbclasses(y):
    y_train = np.array(y)
    uniquelbls = np.unique(y_train)
    for pat in uniquelbls:
#        print pat
        print  fidclass(pat,classif)
    
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes :', int(nb_classes)) 
    y_flatten=y_train.flatten()
    class_weights= get_class_weights(y_flatten)
    return int(nb_classes),class_weights

def readclasses2(num_classes,X_trainl,y_trainl):
    
    X_train = np.asarray(np.expand_dims(X_trainl,3))           

    y_train = np.array(y_trainl)
    lyt=y_train.shape[0]-1
    
    ytrainr=np.zeros((y_train.shape[0],image_rows, image_cols,int(num_classes)),np.uint8)

    for i in range (lyt):
        for j in range (0,image_rows):
                ytrainr[i][j] = np_utils.to_categorical(y_train[i][j], num_classes)        

#    print class_weights
    return X_train, ytrainr       
   
def readclasses(usedpatient):
    patch_list=[]
    label_list=[]

    for category in usedpatient:
        category_dir = os.path.join(picklepathdir, category)
        print  'work on: ', category
        image_files = [name for name in os.listdir(category_dir) if name.find('.pkl') > 0 ]
        for filei in image_files:
            pos=filei.find('_')
            numslice=filei[0:pos]
#            usedpatient[category]=usedpatient[category]+1
            usedpatientlist[category].append(numslice)
            readpkl=pickle.load(open(os.path.join(category_dir,filei), "rb"))
            for i in range (len(readpkl[0])):
                                
                scan=readpkl[0][i]
                mask=readpkl[1][i]    
                           
                scanr=cv2.resize(scan,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
                maskr=cv2.resize(mask,(image_cols, image_rows),interpolation=cv2.INTER_LINEAR)
#                print scanr.min(),scanr.max()
                scanm=norm(scanr)
#                print scanm.min(),scanm.max()
                patch_list.append(scanm)
                label_list.append(maskr)
  
    X_train = np.asarray(np.expand_dims(patch_list,3))           

    y_train = np.array(label_list)
    uniquelbls = np.unique(y_train)
    for pat in uniquelbls:
#        print pat
        print  fidclass(pat,classif)
    
    nb_classes = int( uniquelbls.shape[0])
    print ('number of classes :', int(nb_classes)) 
    ytrainr=np.zeros((y_train.shape[0],image_rows, image_cols,int(nb_classes)),np.uint8)

    for i in range (y_train.shape[0]-1):
        for j in range (0,image_rows):
                ytrainr[i][j] = np_utils.to_categorical(y_train[i][j], nb_classes)        
#                if  y_train[i][j][120]!=0:
#                    print y_train[i][j][120]
#                    print ytrainr[i][j][120]
#                    print argmax(ytrainr[i][j][120])
#    ooo
#    print y_train.shape
#    for i in range(y_train.shape[0]):
    y_flatten=y_train.flatten()
    class_weights= get_class_weights(y_flatten)
#    print class_weights
    return X_train, ytrainr   ,class_weights            


usedpatient={}
usedpatientlist={}

listpatient=[name for name in os.listdir(picklepathdir)]
for c in listpatient:
    usedpatient[c]=0
    usedpatientlist[c]=[]
    
countclasses()
print ('patients found:')
print ( usedpatient)
print('----------------------------')

list_patient=[]
for k,value in usedpatient.items():   
    list_patient.append(k)
#print list_patient
spl=np.array_split(list_patient,pklnum)

X_trainl={}
y_trainl={}
y_trainlist=[]

for i in range(pklnum):
    listp=spl[i]
    print 'set number :',i,' ',listp
    print('-' * 30)
    X_trainl[i],y_trainl[i]=readclasses1(listp)
#    print len(X_trainl[i]),len(y_trainl[i])
 
    y_trainlist=y_trainlist+y_trainl[i]

#print len(y_trainlist)
num_classes,class_weights=numbclasses(y_trainlist)
print "weights"
for key,value in class_weights.items():
   print key, value 
print('-' * 30)
#
for i in range(pklnum):
    diri=os.path.join(pickle_dir,str(i))
    remove_folder(diri)
    os.mkdir(diri)
    X_train,y_train=readclasses2(num_classes,X_trainl[i],y_trainl[i])
    pickle.dump(X_train, open( os.path.join(diri,"X_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(y_train, open( os.path.join(diri,"y_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(class_weights, open( os.path.join(diri,"class_weights.pkl"), "wb" ),protocol=-1)
    
