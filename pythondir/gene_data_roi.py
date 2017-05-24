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
import keras
import theano
from keras.utils import np_utils
import cPickle as pickle
import collections
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split


print keras.__version__
print theano.__version__
nameHug='HUG'
toppatch= 'TOPPATCH'
#extension for output dir
extendir='ILD_TXT'
#extendir='ILD6'

extendir2=''
pklnum=1
pickel_dirsource_root='pickle'

##############################################################

sepextend2=''
if len (extendir2)>0:
    sepextend2='_'

pickel_dirsource=pickel_dirsource_root+'_'+extendir+sepextend2+extendir2


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
    return  {cls: float(majority/count) for cls, count in counter.items()}

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
                scanm=norm(scan)
#                o=normi(scan)
#                on=normi(scanm)
#                n=normi(mask)
#                cv2.imshow('scan',o)
#                cv2.imshow('scann',on)
#                cv2.imshow('mask',n)
#                cv2.waitKey(0)
#                cv2.destroyAllWindows()
                patch_list.append(scanm)
                label_list.append(mask)
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
#    class_weights[0]=class_weights[0]/200.0
#    class_weights[1]=class_weights[1]/100.0
    
    return int(nb_classes),class_weights

def readclasses2(num_classes,X_trainl,y_trainl):

    X_traini, X_testi, y_traini, y_testi = train_test_split(X_trainl,
                                        y_trainl,test_size=0.2, random_state=42)
    
    X_train = np.asarray(np.expand_dims(X_traini,1)) 
#    X_train = np.repeat(X_train,3,axis=3) 
    X_test = np.asarray(np.expand_dims(X_testi,1))    
#    X_test = np.repeat(X_test,3,axis=3)        
#    X_train = np.asarray(X_traini)
#    X_test = np.asarray(X_testi) 
    y_train = np.array(y_traini)
    y_test = np.array(y_testi)
     
    lytrain=y_train.shape[0]
    lytest=y_test.shape[0]
    
    ytrainr=np.zeros((lytrain,image_rows, image_cols,int(num_classes)),np.uint8)
    ytestr=np.zeros((lytest,image_rows, image_cols,int(num_classes)),np.uint8)

    for i in range (lytrain):
        for j in range (0,image_rows):
            ytrainr[i][j] = np_utils.to_categorical(y_train[i][j], num_classes)
    
    ytrainr=np.moveaxis(ytrainr,3,1)
    for i in range (lytest):
        for j in range (0,image_rows):
            ytestr[i][j] = np_utils.to_categorical(y_test[i][j], num_classes)
#    print class_weights
    ytestr=np.moveaxis(ytestr,3,1)
    return X_train, X_test, ytrainr, ytestr       
   
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
    y_trainlist=y_trainlist+y_trainl[i]

num_classes,class_weights=numbclasses(y_trainlist)
print "weights"
for key,value in class_weights.items():
   print key, fidclass (key,classif), value
print('-' * 30)
for i in range(pklnum):
    diri=os.path.join(pickle_dir,str(i))
    remove_folder(diri)
    os.mkdir(diri)
    X_train, X_test, y_train, y_test =readclasses2(num_classes,X_trainl[i],y_trainl[i])
    print 'shape X_train :',X_train.shape
    print 'shape X_test :',X_test.shape
    print 'shape y_train :',y_train.shape
    print 'shape y_test :',y_test.shape
    pickle.dump(X_train, open( os.path.join(diri,"X_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(y_train, open( os.path.join(diri,"y_train.pkl"), "wb" ),protocol=-1)
    pickle.dump(X_test, open( os.path.join(diri,"X_test.pkl"), "wb" ),protocol=-1)
    pickle.dump(y_test, open( os.path.join(diri,"y_test.pkl"), "wb" ),protocol=-1)
    pickle.dump(class_weights, open( os.path.join(diri,"class_weights.pkl"), "wb" ),protocol=-1)
    
